"""
Optimal MoE Architecture for Language Modeling (2024-2025)

Implementation based on latest research:
- 8 experts with top-2 routing
- SwiGLU activation
- Auxiliary-loss-free load balancing with dynamic bias
- Router z-loss for stability
- Modern components: RMSNorm, RoPE, Grouped Query Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        x_norm = x / rms
        return self.weight * x_norm


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin for max sequence length
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with 4-8x KV compression"""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads  # Repetition factor
        
        # Q, K, V projections
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
        # RoPE
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        
        # QK normalization for stability
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match Q heads"""
        batch, n_kv_heads, seq_len, head_dim = x.shape
        if self.n_rep == 1:
            return x
        return x[:, :, None, :, :].expand(batch, n_kv_heads, self.n_rep, seq_len, head_dim).reshape(
            batch, n_kv_heads * self.n_rep, seq_len, head_dim
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply RoPE
        q, k = self.rope(q, k)
        
        # Repeat KV to match Q heads
        k = self.repeat_kv(k)
        v = self.repeat_kv(v)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            attn_weights = attn_weights + mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn_output)


class SwiGLUExpert(nn.Module):
    """Single expert with SwiGLU activation"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: gate(x) * SiLU(up(x))
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


class TopKRouter(nn.Module):
    """
    Top-K router with auxiliary-loss-free load balancing
    Uses dynamic bias terms instead of auxiliary loss
    """
    def __init__(self, dim: int, num_experts: int, top_k: int = 2, 
                 bias_update_speed: float = 0.001):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.bias_update_speed = bias_update_speed
        
        # Router weights
        self.gate = nn.Linear(dim, num_experts, bias=False)
        
        # Dynamic bias for load balancing (not trained, updated algorithmically)
        self.register_buffer("expert_bias", torch.zeros(num_experts))
        
        # Track expert utilization for bias updates
        self.register_buffer("expert_counts", torch.zeros(num_experts))
        self.register_buffer("total_tokens", torch.tensor(0.0))

    def compute_router_z_loss(self, logits: torch.Tensor, z_loss_coeff: float = 1e-5) -> torch.Tensor:
        """
        Router z-loss to prevent numerical instability
        Constrains logit magnitude entering the gating network
        """
        log_z = torch.logsumexp(logits, dim=-1)
        z_loss = torch.mean(log_z ** 2)
        return z_loss_coeff * z_loss

    def update_expert_bias(self, expert_indices: torch.Tensor, batch_tokens: int):
        """
        Update expert bias based on load imbalance
        This replaces the auxiliary loss for load balancing
        """
        if not self.training:
            return
        
        # Count tokens routed to each expert
        with torch.no_grad():
            counts = torch.bincount(expert_indices.flatten(), minlength=self.num_experts).float()
            self.expert_counts += counts
            self.total_tokens += batch_tokens
            
            # Calculate current load and target load
            current_load = self.expert_counts / (self.total_tokens + 1e-8)
            target_load = 1.0 / self.num_experts
            
            # Update bias to push away from overused experts
            load_diff = current_load - target_load
            self.expert_bias -= self.bias_update_speed * load_diff

    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            training: Whether in training mode
            
        Returns:
            expert_indices: Top-k expert indices [batch_size, seq_len, top_k]
            expert_weights: Normalized weights [batch_size, seq_len, top_k]
            z_loss: Router z-loss for stability
            routing_info: Dictionary with routing statistics
        """
        batch_size, seq_len, dim = x.shape
        
        # Compute router logits
        logits = self.gate(x)  # [batch, seq_len, num_experts]
        
        # Apply bias for load balancing (only affects selection, not weighting)
        logits_for_selection = logits + self.expert_bias.unsqueeze(0).unsqueeze(0)
        
        # Compute z-loss before softmax
        z_loss = self.compute_router_z_loss(logits)
        
        # Select top-k experts (using biased logits)
        top_k_logits, expert_indices = torch.topk(logits_for_selection, self.top_k, dim=-1)
        
        # Compute weights using ORIGINAL logits (not biased)
        # Gather the original logits for selected experts
        batch_idx = torch.arange(batch_size, device=x.device)[:, None, None]
        seq_idx = torch.arange(seq_len, device=x.device)[None, :, None]
        selected_logits = logits[batch_idx, seq_idx, expert_indices]
        
        # Normalize weights with softmax
        expert_weights = F.softmax(selected_logits, dim=-1)
        
        # Update bias for load balancing
        if training:
            self.update_expert_bias(expert_indices, batch_size * seq_len)
        
        # Gather routing statistics
        routing_info = {
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'expert_utilization': self.expert_counts / (self.total_tokens + 1e-8),
            'expert_bias': self.expert_bias.clone()
        }
        
        return expert_indices, expert_weights, z_loss, routing_info


class MixtureOfExpertsLayer(nn.Module):
    """
    MoE layer with 8 experts, top-2 routing, and auxiliary-loss-free balancing
    """
    def __init__(self, dim: int, num_experts: int = 8, top_k: int = 2, 
                 expert_hidden_dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Default: 3.5-4x expansion as per research
        if expert_hidden_dim is None:
            expert_hidden_dim = int(dim * 3.5)
        
        # Create experts
        self.experts = nn.ModuleList([
            SwiGLUExpert(dim, expert_hidden_dim) for _ in range(num_experts)
        ])
        
        # Router
        self.router = TopKRouter(dim, num_experts, top_k)

    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            
        Returns:
            output: MoE output [batch_size, seq_len, dim]
            routing_info: Dictionary with routing statistics and losses
        """
        batch_size, seq_len, dim = x.shape
        
        # Route tokens to experts
        expert_indices, expert_weights, z_loss, routing_info = self.router(x, training)
        
        # Reshape for expert processing
        x_flat = x.view(-1, dim)  # [batch * seq_len, dim]
        expert_indices_flat = expert_indices.view(-1, self.top_k)  # [batch * seq_len, top_k]
        expert_weights_flat = expert_weights.view(-1, self.top_k)  # [batch * seq_len, top_k]
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find all tokens routed to this expert
            expert_mask = (expert_indices_flat == expert_idx)
            token_indices, k_indices = torch.where(expert_mask)
            
            if len(token_indices) == 0:
                continue
            
            # Get tokens and weights for this expert
            expert_tokens = x_flat[token_indices]
            expert_token_weights = expert_weights_flat[token_indices, k_indices]
            
            # Process through expert
            expert_output = self.experts[expert_idx](expert_tokens)
            
            # Weight and accumulate
            weighted_output = expert_output * expert_token_weights.unsqueeze(-1)
            output.index_add_(0, token_indices, weighted_output)
        
        # Reshape back
        output = output.view(batch_size, seq_len, dim)
        
        # Add z_loss to routing info
        routing_info['z_loss'] = z_loss
        
        return output, routing_info


class TransformerBlock(nn.Module):
    """Transformer block with GQA and MoE"""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, 
                 num_experts: int = 8, top_k: int = 2, use_moe: bool = True):
        super().__init__()
        self.use_moe = use_moe
        
        # Pre-normalization
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        
        # Attention
        self.attention = GroupedQueryAttention(dim, n_heads, n_kv_heads)
        
        # FFN or MoE
        if use_moe:
            self.ffn = MixtureOfExpertsLayer(dim, num_experts, top_k)
        else:
            # Standard dense FFN for first layers if needed
            hidden_dim = int(dim * 3.5)
            self.ffn = nn.Sequential(
                nn.Linear(dim, hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_dim, dim, bias=False)
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                training: bool = True) -> Tuple[torch.Tensor, dict]:
        # Attention with residual
        attn_out = self.attention(self.attn_norm(x), mask)
        x = x + attn_out
        
        # FFN with residual
        routing_info = {}
        if self.use_moe:
            ffn_out, routing_info = self.ffn(self.ffn_norm(x), training)
        else:
            ffn_out = self.ffn(self.ffn_norm(x))
        
        x = x + ffn_out
        
        return x, routing_info


class MoELanguageModel(nn.Module):
    """
    Complete MoE Language Model with optimal 2024-2025 architecture
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        n_kv_heads: int = 2,  # 4x compression
        num_experts: int = 8,
        top_k: int = 2,
        max_seq_len: int = 2048,
        skip_first_n_moe: int = 0,  # Skip MoE in first N layers
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        
        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim, n_heads, n_kv_heads, num_experts, top_k,
                use_moe=(i >= skip_first_n_moe)  # Skip MoE in first N layers
            )
            for i in range(n_layers)
        ])
        
        # Final norm and output
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie embeddings
        self.output.weight = self.token_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following research recommendations"""
        if isinstance(module, nn.Linear):
            # Reduced scale initialization for stability
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.006)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.006)

    def forward(self, input_ids: torch.Tensor, 
                training: bool = True) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            training: Whether in training mode
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            aux_info: Dictionary with routing info and losses
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        x = self.token_emb(input_ids)
        
        # Create causal mask
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Process through transformer blocks
        total_z_loss = 0.0
        routing_stats = []
        
        for layer in self.layers:
            x, routing_info = layer(x, mask, training)
            if routing_info:
                if 'z_loss' in routing_info:
                    total_z_loss += routing_info['z_loss']
                routing_stats.append(routing_info)
        
        # Final norm and output projection
        x = self.norm(x)
        logits = self.output(x)
        
        # Compile auxiliary information
        aux_info = {
            'total_z_loss': total_z_loss,
            'routing_stats': routing_stats,
            'num_moe_layers': len(routing_stats)
        }
        
        return logits, aux_info

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Simple autoregressive generation"""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                logits, _ = self.forward(input_ids, training=False)
                
                # Get logits for last token
                logits = logits[:, -1, :] / temperature
                
                # Optional top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


def count_parameters(model: nn.Module) -> dict:
    """Count total and active parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    
    # Estimate active parameters (with top-2 of 8 experts, 25% are active)
    expert_params = sum(p.numel() for name, p in model.named_parameters() if 'experts' in name)
    non_expert_params = total_params - expert_params
    
    # With top-2 of 8 experts, we use 2/8 = 25% of expert parameters
    active_expert_params = expert_params * (2.0 / 8.0)
    active_params = non_expert_params + active_expert_params
    
    return {
        'total': total_params,
        'active': active_params,
        'expert': expert_params,
        'non_expert': non_expert_params,
        'efficiency': active_params / total_params
    }


if __name__ == "__main__":
    # Test the model
    print("Testing MoE Language Model...")
    
    model = MoELanguageModel(
        vocab_size=50257,  # GPT-2 vocab size
        dim=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=2,
        num_experts=8,
        top_k=2,
        skip_first_n_moe=0
    )
    
    # Print parameter counts
    param_info = count_parameters(model)
    print(f"\nParameter counts:")
    print(f"  Total: {param_info['total']:,} ({param_info['total']/1e6:.2f}M)")
    print(f"  Active: {param_info['active']:,} ({param_info['active']/1e6:.2f}M)")
    print(f"  Expert: {param_info['expert']:,} ({param_info['expert']/1e6:.2f}M)")
    print(f"  Efficiency: {param_info['efficiency']:.2%}")
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    
    print(f"\nTesting forward pass with input shape: {input_ids.shape}")
    logits, aux_info = model(input_ids, training=True)
    print(f"Output logits shape: {logits.shape}")
    print(f"Total z-loss: {aux_info['total_z_loss']:.6f}")
    print(f"Number of MoE layers: {aux_info['num_moe_layers']}")
    
    print("\n✓ Model architecture test passed!")