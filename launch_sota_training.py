#!/usr/bin/env python3
"""
SOTA MoE Training Launcher with all 2024-2025 improvements
Supports: Expert Choice, Shared Experts, Multi-token Prediction, Expert Dropout
"""

import argparse
import torch
from train_moe import MoETrainer, TextDataset
from moe_model import MoELanguageModel, count_parameters
from sota_config import get_config


def parse_args():
    parser = argparse.ArgumentParser(description='Launch SOTA MoE Language Model Training')

    # Quick config selection
    parser.add_argument('--config', type=str, default='SOTA_STANDARD',
                       choices=['SOTA_STANDARD', 'SOTA_EXPERT_CHOICE', 'SOTA_MULTI_TOKEN',
                               'SOTA_FINE_GRAINED', 'SOTA_MAX_PERFORMANCE', 'SOTA_SMALL_FAST'],
                       help='Pre-configured SOTA setup')

    # Model architecture
    parser.add_argument('--vocab-size', type=int, default=None, help='Vocabulary size')
    parser.add_argument('--dim', type=int, default=None, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=None, help='Number of layers')
    parser.add_argument('--n-heads', type=int, default=None, help='Number of attention heads')
    parser.add_argument('--n-kv-heads', type=int, default=None, help='Number of KV heads (for GQA)')
    parser.add_argument('--num-experts', type=int, default=None, help='Number of experts')
    parser.add_argument('--top-k', type=int, default=None, help='Number of experts to route to')
    parser.add_argument('--skip-first-n-moe', type=int, default=None, help='Skip MoE in first N layers')

    # SOTA Features
    parser.add_argument('--use-shared-expert', action='store_true', help='Use shared experts')
    parser.add_argument('--no-shared-expert', action='store_true', help='Disable shared experts')
    parser.add_argument('--expert-choice', action='store_true', help='Use Expert Choice routing (2x speedup)')
    parser.add_argument('--expert-dropout', type=float, default=None, help='Expert-level dropout rate')
    parser.add_argument('--num-predict-tokens', type=int, default=None, help='Multi-token prediction (1=standard)')

    # Training
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size per GPU')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=None, help='Gradient accumulation')
    parser.add_argument('--max-steps', type=int, default=None, help='Maximum training steps')
    parser.add_argument('--learning-rate', type=float, default=None, help='Peak learning rate')
    parser.add_argument('--weight-decay', type=float, default=None, help='Weight decay')
    parser.add_argument('--warmup-steps', type=int, default=None, help='Warmup steps')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping norm')

    # Data
    parser.add_argument('--train-data', type=str, default='train_data.npy', help='Training data path')
    parser.add_argument('--val-data', type=str, default='val_data.npy', help='Validation data path')
    parser.add_argument('--seq-len', type=int, default=512, help='Sequence length')

    # Logging
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--eval-interval', type=int, default=1000, help='Evaluation interval')
    parser.add_argument('--save-interval', type=int, default=5000, help='Save interval')

    # WandB
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='moe-sota', help='WandB project')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='WandB run name')

    # Mixed precision
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision training')

    # Resume training
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load base configuration
    config = get_config(args.config)

    print("="*60)
    print(f"SOTA MoE Language Model Training - {args.config}")
    print("="*60)

    # Override config with command line arguments
    if args.vocab_size is not None:
        config['vocab_size'] = args.vocab_size
    if args.dim is not None:
        config['dim'] = args.dim
    if args.n_layers is not None:
        config['n_layers'] = args.n_layers
    if args.n_heads is not None:
        config['n_heads'] = args.n_heads
    if args.n_kv_heads is not None:
        config['n_kv_heads'] = args.n_kv_heads
    if args.num_experts is not None:
        config['num_experts'] = args.num_experts
    if args.top_k is not None:
        config['top_k'] = args.top_k
    if args.skip_first_n_moe is not None:
        config['skip_first_n_moe'] = args.skip_first_n_moe

    # SOTA features
    if args.use_shared_expert:
        config['use_shared_expert'] = True
    if args.no_shared_expert:
        config['use_shared_expert'] = False
    if args.expert_choice:
        config['expert_choice'] = True
    if args.expert_dropout is not None:
        config['expert_dropout'] = args.expert_dropout
    if args.num_predict_tokens is not None:
        config['num_predict_tokens'] = args.num_predict_tokens

    # Training parameters
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    if args.max_steps is not None:
        config['max_steps'] = args.max_steps
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.weight_decay is not None:
        config['weight_decay'] = args.weight_decay
    if args.warmup_steps is not None:
        config['warmup_steps'] = args.warmup_steps

    # Add other config options
    config['grad_clip'] = args.grad_clip
    config['use_amp'] = not args.no_amp
    config['log_interval'] = args.log_interval
    config['eval_interval'] = args.eval_interval
    config['save_interval'] = args.save_interval
    config['checkpoint_dir'] = args.checkpoint_dir
    config['use_wandb'] = args.use_wandb
    config['wandb_project'] = args.wandb_project
    config['wandb_run_name'] = args.wandb_run_name or f"{args.config}_{config['dim']}d_{config['n_layers']}l"
    config['beta1'] = 0.9
    config['beta2'] = 0.95

    print("\nConfiguration:")
    print("-" * 60)
    print(f"Model: {config['dim']}d × {config['n_layers']}L")
    print(f"Experts: {config['num_experts']} (top-{config['top_k']})")
    print(f"Shared Expert: {config.get('use_shared_expert', False)}")
    print(f"Expert Choice: {config.get('expert_choice', False)}")
    print(f"Expert Dropout: {config.get('expert_dropout', 0.0)}")
    print(f"Multi-token: {config.get('num_predict_tokens', 1)} tokens")
    print(f"Batch: {config['batch_size']} × {config['gradient_accumulation_steps']} = {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"Steps: {config['max_steps']} (warmup: {config['warmup_steps']})")
    print("-" * 60)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"\nCUDA: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\nWARNING: CUDA not available, training on CPU")
    print()

    # Load datasets
    print("Loading datasets...")
    train_dataset = TextDataset(args.train_data, seq_len=args.seq_len, vocab_size=config['vocab_size'])
    val_dataset = TextDataset(args.val_data, seq_len=args.seq_len, vocab_size=config['vocab_size'])
    print()

    # Initialize model with SOTA features
    print("Initializing SOTA MoE model...")
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

    # Print parameter counts
    param_info = count_parameters(model)
    print("\nModel Statistics:")
    print(f"  Total params: {param_info['total']:,} ({param_info['total']/1e6:.1f}M)")
    print(f"  Active params: {param_info['active']:,} ({param_info['active']/1e6:.1f}M)")
    print(f"  Expert params: {param_info['expert']:,} ({param_info['expert']/1e6:.1f}M)")
    print(f"  Efficiency: {param_info['efficiency']:.1%}")
    print()

    # Initialize trainer
    trainer = MoETrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
        print()

    # Print SOTA features summary
    print("\n" + "="*60)
    print("SOTA Features Enabled:")
    print("="*60)
    features = []
    if config.get('use_shared_expert', False):
        features.append("✓ Shared Experts (improves quality)")
    if config.get('expert_choice', False):
        features.append("✓ Expert Choice Routing (2x speedup)")
    if config.get('expert_dropout', 0.0) > 0:
        features.append(f"✓ Expert Dropout ({config['expert_dropout']:.1%})")
    if config.get('num_predict_tokens', 1) > 1:
        features.append(f"✓ Multi-token Prediction ({config['num_predict_tokens']} tokens)")
    features.append("✓ Entropy Regularization (routing diversity)")
    features.append("✓ Auxiliary-loss-free Load Balancing")
    features.append("✓ Router Z-loss (stability)")
    features.append("✓ Grouped Query Attention (KV compression)")

    for feature in features:
        print(f"  {feature}")
    print("="*60 + "\n")

    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        import os
        interrupt_path = os.path.join(config['checkpoint_dir'], 'interrupted.pt')
        trainer.save_checkpoint(interrupt_path)
        print(f"Checkpoint saved to {interrupt_path}")


if __name__ == "__main__":
    main()
