"""
WandB monitoring utilities for Super-Resolution training
"""

import os
import torch
import wandb
import numpy as np
from datetime import datetime


class WandbLogger:
    def __init__(self, args, model, loss):
        self.args = args
        self.model = model
        self.loss = loss

        # Initialize wandb if enabled
        if args.use_wandb:
            self.init_wandb(args, model, loss)

    def init_wandb(self, args, model, loss):
        """Initialize wandb logging"""
        try:
            # Create experiment name if not provided
            exp_name = args.wandb_name or f"{args.model}_{args.dataset}_scale{args.scale[0]}_{datetime.now().strftime('%m%d_%H%M')}"

            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=exp_name,
                config={
                    'model': args.model,
                    'dataset': args.dataset,
                    'scale': args.scale[0],
                    'optimizer': args.optimizer,
                    'scheduler': getattr(args, 'scheduler', 'step'),
                    'learning_rate': args.lr,
                    'batch_size': args.batch_size,
                    'epochs': args.epochs,
                    'patch_size': args.patch_size,
                    'loss': args.loss,
                    # SymUNet parameters
                    'symunet_width': getattr(args, f'{args.model.lower()}_width', 64),
                    'symunet_enc_blk_nums': getattr(args, f'{args.model.lower()}_enc_blk_nums', [2, 2, 2]),
                    'symunet_dec_blk_nums': getattr(args, f'{args.model.lower()}_dec_blk_nums', [2, 2, 2]),
                    'symunet_restormer_heads': getattr(args, f'{args.model.lower()}_restormer_heads', [1, 2, 4]),
                }
            )

            # Log model architecture
            wandb.watch(model, log='all', log_freq=100)

            print(f"✅ WandB initialized: {args.wandb_project}/{exp_name}")

        except Exception as e:
            print(f"❌ Failed to initialize WandB: {e}")
            print("Continuing without WandB logging...")

    def log_training_step(self, step, epoch, batch_idx, total_batches, loss_dict, learning_rate, time_data, time_model):
        """Log training step metrics to WandB"""
        if not self.args.use_wandb:
            return

        # Calculate progress
        progress = (epoch - 1) * total_batches + batch_idx + 1
        total_steps = self.args.epochs * total_batches

        # Prepare metrics
        metrics = {
            'train/epoch': epoch,
            'train/progress': progress,
            'train/learning_rate': learning_rate,
            'train/time_data': time_data,
            'train/time_model': time_model,
        }

        # Add loss metrics
        if isinstance(loss_dict, dict):
            for key, value in loss_dict.items():
                if isinstance(value, (int, float)):
                    metrics[f'train/{key}'] = float(value)

        # Log to wandb
        wandb.log(metrics, step=progress)

    def log_validation(self, epoch, psnr_value, ssim_value=None, best_psnr=None):
        """Log validation metrics to WandB"""
        if not self.args.use_wandb:
            return

        metrics = {
            'val/epoch': epoch,
            'val/psnr': psnr_value,
        }

        if ssim_value is not None:
            metrics['val/ssim'] = ssim_value

        if best_psnr is not None:
            metrics['val/best_psnr'] = best_psnr

        wandb.log(metrics, step=epoch)

    def log_images(self, epoch, lr_img, hr_img, sr_img, save_images=True):
        """Log sample images to WandB"""
        if not self.args.use_wandb or not save_images:
            return

        try:
            # Convert tensors to numpy arrays
            lr_np = self.tensor_to_numpy(lr_img[0])  # First image in batch
            hr_np = self.tensor_to_numpy(hr_img[0])
            sr_np = self.tensor_to_numpy(sr_img[0])

            # Create comparison image
            comparison = np.hstack([lr_np, sr_np, hr_np])

            # Log to wandb
            wandb.log({
                'sample_images/epoch': epoch,
                'sample_images/comparison': wandb.Image(
                    comparison,
                    caption=f'LR | SR | HR (Epoch {epoch})'
                )
            }, step=epoch)

        except Exception as e:
            print(f"⚠️ Failed to log images to WandB: {e}")

    def log_model_summary(self):
        """Log model summary to WandB"""
        if not self.args.use_wandb:
            return

        try:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            wandb.log({
                'model/total_parameters': total_params,
                'model/trainable_parameters': trainable_params,
            })

        except Exception as e:
            print(f"⚠️ Failed to log model summary: {e}")

    def save_model_artifact(self, model_path, model_name='SymUNet-SR'):
        """Save model as WandB artifact"""
        if not self.args.use_wandb:
            return

        try:
            artifact = wandb.Artifact(model_name, type='model')
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

        except Exception as e:
            print(f"⚠️ Failed to save model artifact: {e}")

    def finish(self):
        """Finish WandB run"""
        if self.args.use_wandb:
            try:
                wandb.finish()
                print("✅ WandB run finished")
            except Exception as e:
                print(f"⚠️ Failed to finish WandB run: {e}")

    @staticmethod
    def tensor_to_numpy(tensor):
        """Convert tensor to numpy array for visualization"""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu()
            if tensor.dim() == 4:  # BCHW
                tensor = tensor[0]  # Take first image
            tensor = tensor.permute(1, 2, 0)  # CHW to HWC

            # Convert to numpy
            if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
                tensor = tensor.numpy()
                # Ensure values are in [0, 1] range
                tensor = np.clip(tensor, 0, 1)
            else:
                tensor = tensor.numpy()
                tensor = np.clip(tensor, 0, 255).astype(np.uint8)

        return tensor