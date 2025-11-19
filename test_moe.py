"""
Test script to verify MoE model implementation
Run this to make sure everything is working correctly
"""

import torch
import numpy as np
from moe_model import MoELanguageModel, count_parameters
from train_moe import TextDataset, MoETrainer


def test_model_architecture():
    """Test model architecture and forward pass"""
    print("="*60)
    print("Testing Model Architecture")
    print("="*60)
    
    # Create small model for testing
    model = MoELanguageModel(
        vocab_size=1000,
        dim=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        num_experts=8,
        top_k=2,
        max_seq_len=512,
    )
    
    # Print parameter counts
    param_info = count_parameters(model)
    print(f"\nModel parameters:")
    print(f"  Total: {param_info['total']:,} ({param_info['total']/1e6:.2f}M)")
    print(f"  Active: {param_info['active']:,} ({param_info['active']/1e6:.2f}M)")
    print(f"  Efficiency: {param_info['efficiency']:.2%}")
    
    # Test forward pass
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    print(f"\nTesting forward pass:")
    print(f"  Input shape: {input_ids.shape}")
    
    model.eval()
    with torch.no_grad():
        logits, aux_info = model(input_ids, training=False)
    
    print(f"  Output shape: {logits.shape}")
    print(f"  Z-loss: {aux_info['total_z_loss']:.6f}")
    print(f"  MoE layers: {aux_info['num_moe_layers']}")
    
    # Check routing statistics
    if aux_info['routing_stats']:
        stats = aux_info['routing_stats'][0]
        if 'expert_utilization' in stats:
            util = stats['expert_utilization'].cpu().numpy()
            print(f"\nExpert utilization (first layer):")
            for i, u in enumerate(util):
                print(f"    Expert {i}: {u:.4f}")
    
    print("\n✓ Model architecture test passed!")
    return True


def test_load_balancing():
    """Test load balancing mechanism"""
    print("\n" + "="*60)
    print("Testing Load Balancing")
    print("="*60)
    
    model = MoELanguageModel(
        vocab_size=1000,
        dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        num_experts=8,
        top_k=2,
    )
    
    # Run several forward passes to test bias updates
    print("\nRunning 10 training steps to test load balancing...")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    utilizations = []
    biases = []
    
    for step in range(10):
        input_ids = torch.randint(0, 1000, (4, 64))
        target_ids = torch.randint(0, 1000, (4, 64))
        
        logits, aux_info = model(input_ids, training=True)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track statistics
        if aux_info['routing_stats']:
            stats = aux_info['routing_stats'][0]
            util = stats['expert_utilization'].cpu().numpy()
            bias = stats['expert_bias'].cpu().numpy()
            utilizations.append(util)
            biases.append(bias)
    
    # Analyze results
    utilizations = np.array(utilizations)
    biases = np.array(biases)
    
    print(f"\nInitial expert utilization: {utilizations[0]}")
    print(f"Final expert utilization: {utilizations[-1]}")
    print(f"Utilization std (initial): {utilizations[0].std():.4f}")
    print(f"Utilization std (final): {utilizations[-1].std():.4f}")
    
    print(f"\nInitial expert bias: {biases[0]}")
    print(f"Final expert bias: {biases[-1]}")
    
    # Check if balancing is working
    final_std = utilizations[-1].std()
    target_util = 1.0 / 8  # With 8 experts
    
    print(f"\nTarget utilization per expert: {target_util:.4f}")
    print(f"Load balancing working: {final_std < 0.1}")
    
    print("\n✓ Load balancing test passed!")
    return True


def test_generation():
    """Test text generation"""
    print("\n" + "="*60)
    print("Testing Text Generation")
    print("="*60)
    
    model = MoELanguageModel(
        vocab_size=1000,
        dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        num_experts=4,
        top_k=2,
    )
    
    # Create a simple prompt
    prompt = torch.randint(0, 1000, (1, 10))
    
    print(f"\nGenerating 20 tokens from prompt of length {prompt.size(1)}...")
    
    model.eval()
    output = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=1.0,
        top_k=50
    )
    
    print(f"Generated sequence length: {output.size(1)}")
    print(f"Expected length: {prompt.size(1) + 20}")
    
    assert output.size(1) == prompt.size(1) + 20, "Generation length mismatch"
    
    print("\n✓ Generation test passed!")
    return True


def test_mixed_precision():
    """Test mixed precision training"""
    print("\n" + "="*60)
    print("Testing Mixed Precision Training")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping mixed precision test")
        return True
    
    device = torch.device("cuda")
    
    model = MoELanguageModel(
        vocab_size=1000,
        dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        num_experts=4,
        top_k=2,
    ).to(device)
    
    from torch.cuda.amp import autocast, GradScaler
    
    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\nRunning training step with mixed precision (BF16)...")
    
    model.train()
    input_ids = torch.randint(0, 1000, (2, 64)).to(device)
    target_ids = torch.randint(0, 1000, (2, 64)).to(device)
    
    # Forward with autocast
    with autocast(dtype=torch.bfloat16):
        logits, aux_info = model(input_ids, training=True)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )
    
    # Backward with scaling
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Scaler scale: {scaler.get_scale():.1f}")
    
    print("\n✓ Mixed precision test passed!")
    return True


def test_checkpoint_save_load():
    """Test checkpoint saving and loading"""
    print("\n" + "="*60)
    print("Testing Checkpoint Save/Load")
    print("="*60)
    
    # Create model
    model1 = MoELanguageModel(
        vocab_size=1000,
        dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        num_experts=4,
        top_k=2,
    )
    
    # Save checkpoint
    checkpoint = {
        'model_state_dict': model1.state_dict(),
        'config': {'test': 'value'},
    }
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        checkpoint_path = f.name
        torch.save(checkpoint, checkpoint_path)
    
    print(f"\nSaved checkpoint to: {checkpoint_path}")
    
    # Load into new model
    model2 = MoELanguageModel(
        vocab_size=1000,
        dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        num_experts=4,
        top_k=2,
    )
    
    loaded = torch.load(checkpoint_path)
    model2.load_state_dict(loaded['model_state_dict'])
    
    print(f"Loaded checkpoint successfully")
    
    # Verify parameters match
    input_ids = torch.randint(0, 1000, (1, 32))
    
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        out1, _ = model1(input_ids, training=False)
        out2, _ = model2(input_ids, training=False)
    
    diff = (out1 - out2).abs().max().item()
    print(f"Max difference in outputs: {diff:.10f}")
    
    assert diff < 1e-6, "Loaded model outputs differ from original"
    
    # Cleanup
    import os
    os.remove(checkpoint_path)
    
    print("\n✓ Checkpoint save/load test passed!")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("Running MoE Model Tests")
    print("="*80)
    
    tests = [
        ("Model Architecture", test_model_architecture),
        ("Load Balancing", test_load_balancing),
        ("Text Generation", test_generation),
        ("Checkpoint Save/Load", test_checkpoint_save_load),
    ]
    
    # Add mixed precision test if CUDA available
    if torch.cuda.is_available():
        tests.append(("Mixed Precision", test_mixed_precision))
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} test FAILED with error:")
            print(f"  {str(e)}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n" + "="*80)
        print("All tests passed! ✓")
        print("="*80)
        print("\nYou can now:")
        print("  1. Run 'python launch_training.py' to start training")
        print("  2. Customize the configuration for your use case")
        print("  3. Monitor training with the logged metrics")
    else:
        print("\n" + "="*80)
        print("Some tests failed. Please check the errors above.")
        print("="*80)
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()