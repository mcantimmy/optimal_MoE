#!/usr/bin/env python3
"""
Compare SOTA configurations and show their characteristics
"""

import torch
from moe_model import MoELanguageModel, count_parameters
from sota_config import get_config


def compare_configs():
    """Compare all SOTA configurations"""

    configs = [
        'SOTA_STANDARD',
        'SOTA_EXPERT_CHOICE',
        'SOTA_MULTI_TOKEN',
        'SOTA_FINE_GRAINED',
        'SOTA_MAX_PERFORMANCE',
        'SOTA_SMALL_FAST',
    ]

    print("="*80)
    print("SOTA MoE Configuration Comparison")
    print("="*80)
    print()

    results = []

    for config_name in configs:
        config = get_config(config_name)

        # Create model
        model = MoELanguageModel(
            vocab_size=config['vocab_size'],
            dim=config['dim'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            n_kv_heads=config['n_kv_heads'],
            num_experts=config['num_experts'],
            top_k=config['top_k'],
            skip_first_n_moe=config['skip_first_n_moe'],
            use_shared_expert=config.get('use_shared_expert', False),
            expert_choice=config.get('expert_choice', False),
            expert_dropout=config.get('expert_dropout', 0.0),
            num_predict_tokens=config.get('num_predict_tokens', 1),
        )

        # Count parameters
        param_info = count_parameters(model)

        # Calculate memory estimate (rough)
        param_memory_mb = param_info['total'] * 4 / 1e6  # FP32

        # Collect results
        results.append({
            'name': config_name,
            'config': config,
            'params': param_info,
            'memory_mb': param_memory_mb,
        })

    # Print comparison table
    print(f"{'Config':<25} {'Model Size':<15} {'Total Params':<15} {'Active Params':<15} {'Features':<30}")
    print("-" * 100)

    for result in results:
        config = result['config']
        params = result['params']

        model_size = f"{config['dim']}d×{config['n_layers']}L"
        total_params = f"{params['total']/1e6:.1f}M"
        active_params = f"{params['active']/1e6:.1f}M"

        # Features
        features = []
        if config.get('use_shared_expert', False):
            features.append('Shared')
        if config.get('expert_choice', False):
            features.append('ExpertChoice')
        if config.get('expert_dropout', 0.0) > 0:
            features.append(f'Dropout{config["expert_dropout"]:.1f}')
        if config.get('num_predict_tokens', 1) > 1:
            features.append(f'MT{config["num_predict_tokens"]}')
        features_str = ','.join(features) if features else 'Base'

        print(f"{result['name']:<25} {model_size:<15} {total_params:<15} {active_params:<15} {features_str:<30}")

    print()
    print("="*80)
    print("Detailed Comparison")
    print("="*80)
    print()

    for result in results:
        config = result['config']
        params = result['params']

        print(f"\n{result['name']}")
        print("-" * 60)
        print(f"  Model: {config['dim']}d × {config['n_layers']}L")
        print(f"  Experts: {config['num_experts']} (top-{config['top_k']})")
        print(f"  Total Params: {params['total']:,} ({params['total']/1e6:.1f}M)")
        print(f"  Active Params: {params['active']:,} ({params['active']/1e6:.1f}M)")
        print(f"  Efficiency: {params['efficiency']:.1%}")
        print(f"  Est. Memory: {result['memory_mb']:.0f} MB (FP32)")
        print()
        print(f"  SOTA Features:")
        print(f"    Shared Expert: {config.get('use_shared_expert', False)}")
        print(f"    Expert Choice: {config.get('expert_choice', False)}")
        print(f"    Expert Dropout: {config.get('expert_dropout', 0.0)}")
        print(f"    Multi-token: {config.get('num_predict_tokens', 1)} tokens")
        print()
        print(f"  Training:")
        print(f"    Batch: {config['batch_size']} × {config['gradient_accumulation_steps']} = {config['batch_size'] * config['gradient_accumulation_steps']}")
        print(f"    Steps: {config['max_steps']:,}")
        print(f"    LR: {config['learning_rate']}")

    print()
    print("="*80)
    print("Recommendations")
    print("="*80)
    print()
    print("For Quick Experiments:")
    print("  → SOTA_SMALL_FAST (6L, 384d, 4 experts)")
    print()
    print("For Production (Balanced):")
    print("  → SOTA_STANDARD (12L, 768d, 8 experts + shared)")
    print()
    print("For Maximum Speed:")
    print("  → SOTA_EXPERT_CHOICE (8L, 512d, Expert Choice routing)")
    print()
    print("For Maximum Quality:")
    print("  → SOTA_FINE_GRAINED (12L, 768d, 16 experts + shared)")
    print()
    print("For Maximum Performance:")
    print("  → SOTA_MAX_PERFORMANCE (16L, 1024d, all features)")
    print()


if __name__ == "__main__":
    compare_configs()
