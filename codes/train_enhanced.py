#!/usr/bin/env python3
"""
Enhanced Training Script with WandB, AdamW, and Cosine Annealing

This script demonstrates how to use the enhanced training features:
1. WandB monitoring and logging
2. AdamW optimizer
3. Cosine annealing learning rate scheduling
4. Step-based checkpoint saving

Usage Examples:

1. Train with WandB logging:
   python train_enhanced.py \
     --model SYMUNET_PRETRAIN \
     --dataset=UCMerced \
     --scale 4 \
     --ext=img \
     --use_wandb \
     --wandb_project "My-SR-Project" \
     --wandb_name "symunet_pretrain_v1" \
     --epochs 300 \
     --batch_size 4 \
     --optimizer ADAMW \
     --scheduler cosine \
     --cosine_t_max 300 \
     --cosine_eta_min 5e-5 \
     --lr 2e-4 \
     --loss "1*L1+0.005*FFT" \
     --symunet_pretrain_width 48 \
     --symunet_pretrain_enc_blk_nums 4,6,6 \
     --symunet_pretrain_dec_blk_nums 6,6,4 \
     --symunet_pretrain_restormer_heads 1,2,4 \
     --symunet_pretrain_restormer_middle_heads 8 \
     --save_every_n_steps 50 \
     --save enhanced_symunet_pretrain

2. Resume training with new scheduler:
   python train_enhanced.py \
     --resume 1 \
     --save enhanced_symunet_pretrain \
     --scheduler cosine \
     --optimizer ADAMW

3. Test-only mode with WandB logging:
   python train_enhanced.py \
     --test_only \
     --use_wandb \
     --model SYMUNET_PRETRAIN \
     --save enhanced_symunet_pretrain
"""

from option import args
import data
import model
import utils
import loss
import trainer
import torch

args.use_wandb = False


def get_epoch_checkpoint_interval(parsed_args):
    interval = getattr(parsed_args, 'save_every_n_epochs', 0)
    if interval > 0:
        return interval
    return max(0, getattr(parsed_args, 'save_every_n_steps', 0))


if __name__ == '__main__':
    if torch.cuda.is_available() and not args.cpu:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    # Print enhanced training configuration
    print("🚀 Enhanced Training Configuration:")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Scale: {args.scale}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Scheduler: {getattr(args, 'scheduler', 'step')}")
    print(f"Learning Rate: {args.lr}")
    print(f"Loss: {args.loss}")
    print(f"WandB Enabled: {getattr(args, 'use_wandb', False)}")
    print(f"AMP Enabled: {getattr(args, 'amp', False)}")
    print("Validation Every: disabled (always validate every epoch)")
    print(f"Save Every N Epochs: {get_epoch_checkpoint_interval(args)}")
    print(f"DataLoader Threads: {args.n_threads}")

    # Model-specific parameters
    model_prefix = args.model.lower()
    width = getattr(args, f'{model_prefix}_width', 32)
    enc_blks = getattr(args, f'{model_prefix}_enc_blk_nums', [4,6])
    dec_blks = getattr(args, f'{model_prefix}_dec_blk_nums', [6,4])
    heads = getattr(args, f'{model_prefix}_restormer_heads', [1, 2, 4])
    middle_heads = getattr(args, f'{model_prefix}_restormer_middle_heads', 8)

    print(f"\n📐 Model Architecture:")
    print(f"Width: {width}")
    print(f"Encoder Blocks: {enc_blks}")
    print(f"Decoder Blocks: {dec_blks}")
    print(f"Restormer Heads: {heads}")
    print(f"Middle Heads: {middle_heads}")

    # Initialize checkpoint
    checkpoint = utils.checkpoint(args)
    if checkpoint.ok:
        # Create dataloaders
        dataloaders = data.create_dataloaders(args)

        # Create model
        sr_model = model.Model(args, checkpoint)

        # Create loss function
        sr_loss = loss.Loss(args, checkpoint) if not args.test_only else None

        # Create trainer with enhanced features
        t = trainer.Trainer(args, dataloaders, sr_model, sr_loss, checkpoint)

        # Training loop
        print("\n🏃 Starting training...")
        while not t.terminate():
            t.train()
            current_epoch = t.scheduler.last_epoch
            save_every_n_epochs = get_epoch_checkpoint_interval(args)
            if save_every_n_epochs > 0 and current_epoch % save_every_n_epochs == 0:
                t.save_epoch_checkpoint(current_epoch)
            t.test()

        print("✅ Training completed!")
        checkpoint.done()
    else:
        print("❌ Checkpoint initialization failed!")
