"""
Training script for MoE Language Model
Implements 2024-2025 best practices from research
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb
import os
import json
from pathlib import Path
from typing import Optional, Dict
import time
import numpy as np
from tqdm import tqdm

from moe_model import MoELanguageModel, count_parameters


class TextDataset(Dataset):
    """Simple text dataset for language modeling"""
    def __init__(self, data_path: str, seq_len: int = 2048, vocab_size: int = 50257):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # For demonstration, we'll create synthetic data
        # In practice, load your tokenized dataset here
        print(f"Loading data from {data_path}...")
        
        if os.path.exists(data_path):
            # Load real data if available
            self.tokens = np.load(data_path)
        else:
            # Generate synthetic data for testing
            print("Data file not found, generating synthetic data for testing...")
            self.tokens = np.random.randint(0, vocab_size, size=(1000000,), dtype=np.int64)
        
        print(f"Loaded {len(self.tokens):,} tokens")

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        
        # Get input and target sequences
        chunk = self.tokens[start:end]
        input_ids = torch.from_numpy(chunk[:-1].astype(np.int64))
        target_ids = torch.from_numpy(chunk[1:].astype(np.int64))
        
        return input_ids, target_ids


class CosineScheduler:
    """Cosine learning rate schedule with linear warmup"""
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float,
        min_lr: float,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        """Update learning rate"""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class MoETrainer:
    """Trainer for MoE Language Model"""
    def __init__(
        self,
        model: MoELanguageModel,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config: Optional[Dict] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or self.default_config()
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self.setup_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = CosineScheduler(
            self.optimizer,
            warmup_steps=self.config['warmup_steps'],
            max_steps=self.config['max_steps'],
            max_lr=self.config['learning_rate'],
            min_lr=self.config['learning_rate'] * 0.1,
        )
        
        # Setup mixed precision training
        self.use_amp = self.config['use_amp'] and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Setup logging
        self.setup_logging()

    @staticmethod
    def default_config() -> Dict:
        """Default configuration following research best practices"""
        return {
            # Model
            'vocab_size': 50257,
            'dim': 512,
            'n_layers': 8,
            'n_heads': 8,
            'n_kv_heads': 2,
            'num_experts': 8,
            'top_k': 2,
            'skip_first_n_moe': 0,
            
            # Training
            'batch_size': 8,  # Start small, will scale up
            'gradient_accumulation_steps': 8,  # Effective batch = 8 * 8 = 64 sequences
            'max_steps': 100000,
            'learning_rate': 3e-4,
            'weight_decay': 0.1,
            'beta1': 0.9,
            'beta2': 0.95,
            'grad_clip': 1.0,
            'warmup_steps': 2000,
            
            # Mixed precision
            'use_amp': True,
            
            # Logging and checkpointing
            'log_interval': 100,
            'eval_interval': 1000,
            'save_interval': 5000,
            'checkpoint_dir': './checkpoints',
            
            # WandB
            'use_wandb': False,
            'wandb_project': 'moe-language-model',
            'wandb_run_name': None,
        }

    def setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup AdamW optimizer with weight decay"""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Don't apply weight decay to biases and normalization layers
                if 'bias' in name or 'norm' in name or 'ln' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': self.config['weight_decay']},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ],
            lr=self.config['learning_rate'],
            betas=(self.config['beta1'], self.config['beta2']),
            eps=1e-8,
        )
        
        return optimizer

    def setup_logging(self):
        """Setup logging and checkpointing"""
        # Create checkpoint directory
        Path(self.config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        
        # Initialize WandB if enabled
        if self.config['use_wandb']:
            wandb.init(
                project=self.config['wandb_project'],
                name=self.config['wandb_run_name'],
                config=self.config,
            )

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, 
                    aux_info: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute total loss including language modeling and auxiliary losses
        """
        # Language modeling loss
        lm_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
        
        # Z-loss from router (already scaled)
        z_loss = aux_info.get('total_z_loss', torch.tensor(0.0, device=logits.device))
        
        # Total loss
        total_loss = lm_loss + z_loss
        
        return {
            'total_loss': total_loss,
            'lm_loss': lm_loss,
            'z_loss': z_loss,
        }

    def train_step(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> Dict:
        """Single training step"""
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        
        # Forward pass with mixed precision
        with autocast(enabled=self.use_amp, dtype=torch.bfloat16 if self.use_amp else torch.float32):
            logits, aux_info = self.model(input_ids, training=True)
            losses = self.compute_loss(logits, target_ids, aux_info)
            loss = losses['total_loss'] / self.config['gradient_accumulation_steps']
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        return {
            'lm_loss': losses['lm_loss'].item(),
            'z_loss': losses['z_loss'].item(),
            'total_loss': losses['total_loss'].item(),
            'routing_stats': aux_info.get('routing_stats', []),
        }

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Evaluate on validation set"""
        if not self.val_dataset:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        for input_ids, target_ids in self.val_loader:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            with autocast(enabled=self.use_amp, dtype=torch.bfloat16 if self.use_amp else torch.float32):
                logits, aux_info = self.model(input_ids, training=False)
                losses = self.compute_loss(logits, target_ids, aux_info)
            
            total_loss += losses['lm_loss'].item() * input_ids.size(0)
            total_tokens += input_ids.size(0)
        
        self.model.train()
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return {
            'val_loss': avg_loss,
            'val_perplexity': perplexity,
        }

    def log_metrics(self, metrics: Dict, step: int):
        """Log metrics to console and WandB"""
        # Console logging
        log_str = f"Step {step} | "
        log_str += " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                               for k, v in metrics.items() if k != 'routing_stats'])
        print(log_str)
        
        # WandB logging
        if self.config['use_wandb']:
            wandb.log(metrics, step=step)

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state': {
                'current_step': self.scheduler.current_step,
            },
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config,
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.current_step = checkpoint['scheduler_state']['current_step']
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Loaded checkpoint from {path} at step {self.global_step}")

    def train(self):
        """Main training loop"""
        print("\n" + "="*50)
        print("Starting MoE Training")
        print("="*50)
        
        # Print model info
        param_info = count_parameters(self.model)
        print(f"\nModel parameters:")
        print(f"  Total: {param_info['total']:,} ({param_info['total']/1e6:.2f}M)")
        print(f"  Active: {param_info['active']:,} ({param_info['active']/1e6:.2f}M)")
        print(f"  Efficiency: {param_info['efficiency']:.2%}")
        
        print(f"\nTraining configuration:")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Gradient accumulation: {self.config['gradient_accumulation_steps']}")
        print(f"  Effective batch size: {self.config['batch_size'] * self.config['gradient_accumulation_steps']}")
        print(f"  Learning rate: {self.config['learning_rate']}")
        print(f"  Max steps: {self.config['max_steps']}")
        print(f"  Mixed precision: {self.use_amp}")
        print()
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Training metrics
        step_losses = []
        step_times = []
        start_time = time.time()
        
        # Progress bar
        pbar = tqdm(total=self.config['max_steps'], desc="Training")
        pbar.update(self.global_step)
        
        while self.global_step < self.config['max_steps']:
            for input_ids, target_ids in self.train_loader:
                # Training step
                step_metrics = self.train_step(input_ids, target_ids)
                step_losses.append(step_metrics['total_loss'])
                
                # Gradient accumulation
                if (self.global_step + 1) % self.config['gradient_accumulation_steps'] == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    # Learning rate schedule
                    lr = self.scheduler.step()
                    
                    # Update progress
                    self.global_step += 1
                    pbar.update(1)
                    
                    # Compute step time
                    step_time = time.time() - start_time
                    step_times.append(step_time)
                    start_time = time.time()
                
                # Logging
                if self.global_step % self.config['log_interval'] == 0 and self.global_step > 0:
                    avg_loss = np.mean(step_losses[-self.config['log_interval']:])
                    avg_time = np.mean(step_times[-self.config['log_interval']:])
                    
                    metrics = {
                        'train_loss': avg_loss,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'step_time': avg_time,
                        'tokens_per_sec': (self.config['batch_size'] * self.config['gradient_accumulation_steps'] * 
                                         input_ids.size(1)) / avg_time,
                    }
                    
                    # Add routing statistics
                    if step_metrics['routing_stats']:
                        last_layer_stats = step_metrics['routing_stats'][-1]
                        if 'expert_utilization' in last_layer_stats:
                            util = last_layer_stats['expert_utilization'].cpu().numpy()
                            metrics['expert_util_mean'] = float(util.mean())
                            metrics['expert_util_std'] = float(util.std())
                            metrics['expert_util_max'] = float(util.max())
                            metrics['expert_util_min'] = float(util.min())
                    
                    self.log_metrics(metrics, self.global_step)
                
                # Evaluation
                if self.global_step % self.config['eval_interval'] == 0 and self.global_step > 0:
                    eval_metrics = self.evaluate()
                    if eval_metrics:
                        self.log_metrics(eval_metrics, self.global_step)
                        
                        # Save best model
                        if eval_metrics['val_loss'] < self.best_val_loss:
                            self.best_val_loss = eval_metrics['val_loss']
                            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pt')
                            self.save_checkpoint(best_path)
                
                # Checkpointing
                if self.global_step % self.config['save_interval'] == 0 and self.global_step > 0:
                    checkpoint_path = os.path.join(
                        self.config['checkpoint_dir'], 
                        f'checkpoint_step_{self.global_step}.pt'
                    )
                    self.save_checkpoint(checkpoint_path)
                
                # Check if training is complete
                if self.global_step >= self.config['max_steps']:
                    break
            
            self.epoch += 1
            
            if self.global_step >= self.config['max_steps']:
                break
        
        pbar.close()
        print("\n" + "="*50)
        print("Training completed!")
        print("="*50)
        
        # Save final model
        final_path = os.path.join(self.config['checkpoint_dir'], 'final_model.pt')
        self.save_checkpoint(final_path)


def main():
    """Main training function"""
    # Configuration
    config = MoETrainer.default_config()
    
    # Override with custom settings if needed
    config.update({
        'batch_size': 4,
        'gradient_accumulation_steps': 16,  # Effective batch = 64 sequences
        'max_steps': 10000,
        'learning_rate': 3e-4,
        'use_wandb': False,  # Set to True to enable WandB logging
        'checkpoint_dir': './checkpoints',
    })
    
    print("Loading datasets...")
    # Load datasets (using synthetic data for demonstration)
    train_dataset = TextDataset('train_data.npy', seq_len=512)
    val_dataset = TextDataset('val_data.npy', seq_len=512)
    
    print("\nInitializing model...")
    # Initialize model
    model = MoELanguageModel(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        n_kv_heads=config['n_kv_heads'],
        num_experts=config['num_experts'],
        top_k=config['top_k'],
        skip_first_n_moe=config['skip_first_n_moe'],
    )
    
    # Initialize trainer
    trainer = MoETrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()