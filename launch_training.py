#!/usr/bin/env python3
"""
Simple launcher script for MoE training
Run with: python launch_training.py
"""

import argparse
import torch
from train_moe import MoETrainer, TextDataset
from moe_model import MoELanguageModel


def parse_args():
    parser = argparse.ArgumentParser(description='Launch MoE Language Model Training')
    
    # Model architecture
    parser.add_argument('--vocab-size', type=int, default=50257, help='Vocabulary size')
    parser.add_argument('--dim', type=int, default=512, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n-kv-heads', type=int, default=2, help='Number of KV heads (for GQA)')
    parser.add_argument('--num-experts', type=int, default=8, help='Number of experts')
    parser.add_argument('--top-k', type=int, default=2, help='Number of experts to route to')
    parser.add_argument('--skip-first-n-moe', type=int, default=0, help='Skip MoE in first N layers')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=16, help='Gradient accumulation')
    parser.add_argument('--max-steps', type=int, default=10000, help='Maximum training steps')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Peak learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--warmup-steps', type=int, default=2000, help='Warmup steps')
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
    parser.add_argument('--wandb-project', type=str, default='moe-language-model', help='WandB project')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='WandB run name')
    
    # Mixed precision
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision training')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Build configuration from arguments
    config = {
        # Model
        'vocab_size': args.vocab_size,
        'dim': args.dim,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'n_kv_heads': args.n_kv_heads,
        'num_experts': args.num_experts,
        'top_k': args.top_k,
        'skip_first_n_moe': args.skip_first_n_moe,
        
        # Training
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'max_steps': args.max_steps,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'beta1': 0.9,
        'beta2': 0.95,
        'grad_clip': args.grad_clip,
        'warmup_steps': args.warmup_steps,
        
        # Mixed precision
        'use_amp': not args.no_amp,
        
        # Logging
        'log_interval': args.log_interval,
        'eval_interval': args.eval_interval,
        'save_interval': args.save_interval,
        'checkpoint_dir': args.checkpoint_dir,
        
        # WandB
        'use_wandb': args.use_wandb,
        'wandb_project': args.wandb_project,
        'wandb_run_name': args.wandb_run_name,
    }
    
    print("="*60)
    print("MoE Language Model Training")
    print("="*60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("WARNING: CUDA not available, training on CPU")
    print()
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = TextDataset(args.train_data, seq_len=args.seq_len, vocab_size=args.vocab_size)
    val_dataset = TextDataset(args.val_data, seq_len=args.seq_len, vocab_size=args.vocab_size)
    print()
    
    # Initialize model
    print("Initializing model...")
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