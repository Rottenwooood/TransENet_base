#!/usr/bin/env python3
"""
快速批量训练脚本 - 预设几种常用的实验配置

这个脚本提供了一些预设的实验配置，方便快速开始：
1. 学习率对比实验
2. 优化器对比实验
3. 模型宽度对比实验
4. 损失函数对比实验

使用方法:
python quick_batch.py --preset lr_comparison
python quick_batch.py --preset all
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


class QuickBatchTrainer:
    def __init__(self):
        self.base_dir = Path.cwd()
        self.experiment_dir = self.base_dir.parent / "experiment"
        self.experiment_dir.mkdir(exist_ok=True)

    def get_presets(self):
        """获取预设配置"""
        return {
            "lr_comparison": {
                "name": "学习率对比实验",
                "description": "比较不同学习率对模型性能的影响",
                "experiments": [
                    {
                        "name": "lr_1e-4",
                        "config": {
                            "model": "SYMUNET_PRETRAIN",
                            "dataset": "UCMerced",
                            "scale": 4,
                            "epochs": 200,
                            "batch_size": 4,
                            "optimizer": "ADAMW",
                            "scheduler": "cosine",
                            "lr": 1e-4,
                            "cosine_t_max": 200,
                            "loss": "1*L1+0.005*FFT",
                            "symunet_pretrain_width": 48,
                            "symunet_pretrain_enc_blk_nums": "4,6,6",
                            "symunet_pretrain_dec_blk_nums": "6,6,4",
                            "save_every_n_steps": 50,
                            "save": "lr_1e-4"
                        }
                    },
                    {
                        "name": "lr_2e-4",
                        "config": {
                            "model": "SYMUNET_PRETRAIN",
                            "dataset": "UCMerced",
                            "scale": 4,
                            "epochs": 200,
                            "batch_size": 4,
                            "optimizer": "ADAMW",
                            "scheduler": "cosine",
                            "lr": 2e-4,
                            "cosine_t_max": 200,
                            "loss": "1*L1+0.005*FFT",
                            "symunet_pretrain_width": 48,
                            "symunet_pretrain_enc_blk_nums": "4,6,6",
                            "symunet_pretrain_dec_blk_nums": "6,6,4",
                            "save_every_n_steps": 50,
                            "save": "lr_2e-4"
                        }
                    },
                    {
                        "name": "lr_5e-4",
                        "config": {
                            "model": "SYMUNET_PRETRAIN",
                            "dataset": "UCMerced",
                            "scale": 4,
                            "epochs": 200,
                            "batch_size": 4,
                            "optimizer": "ADAMW",
                            "scheduler": "cosine",
                            "lr": 5e-4,
                            "cosine_t_max": 200,
                            "loss": "1*L1+0.005*FFT",
                            "symunet_pretrain_width": 48,
                            "symunet_pretrain_enc_blk_nums": "4,6,6",
                            "symunet_pretrain_dec_blk_nums": "6,6,4",
                            "save_every_n_steps": 50,
                            "save": "lr_5e-4"
                        }
                    }
                ]
            },
            "optimizer_comparison": {
                "name": "优化器对比实验",
                "description": "比较AdamW和Adam优化器的性能",
                "experiments": [
                    {
                        "name": "adamw_cosine",
                        "config": {
                            "model": "SYMUNET_PRETRAIN",
                            "dataset": "UCMerced",
                            "scale": 4,
                            "epochs": 200,
                            "batch_size": 4,
                            "optimizer": "ADAMW",
                            "scheduler": "cosine",
                            "lr": 2e-4,
                            "cosine_t_max": 200,
                            "loss": "1*L1+0.005*FFT",
                            "symunet_pretrain_width": 48,
                            "symunet_pretrain_enc_blk_nums": "4,6,6",
                            "symunet_pretrain_dec_blk_nums": "6,6,4",
                            "save_every_n_steps": 50,
                            "save": "adamw_cosine"
                        }
                    },
                    {
                        "name": "adam_cosine",
                        "config": {
                            "model": "SYMUNET_PRETRAIN",
                            "dataset": "UCMerced",
                            "scale": 4,
                            "epochs": 200,
                            "batch_size": 4,
                            "optimizer": "ADAM",
                            "scheduler": "cosine",
                            "lr": 2e-4,
                            "cosine_t_max": 200,
                            "loss": "1*L1+0.005*FFT",
                            "symunet_pretrain_width": 48,
                            "symunet_pretrain_enc_blk_nums": "4,6,6",
                            "symunet_pretrain_dec_blk_nums": "6,6,4",
                            "save_every_n_steps": 50,
                            "save": "adam_cosine"
                        }
                    },
                    {
                        "name": "adamw_step",
                        "config": {
                            "model": "SYMUNET_PRETRAIN",
                            "dataset": "UCMerced",
                            "scale": 4,
                            "epochs": 200,
                            "batch_size": 4,
                            "optimizer": "ADAMW",
                            "scheduler": "step",
                            "lr": 2e-4,
                            "lr_decay": 100,
                            "loss": "1*L1+0.005*FFT",
                            "symunet_pretrain_width": 48,
                            "symunet_pretrain_enc_blk_nums": "4,6,6",
                            "symunet_pretrain_dec_blk_nums": "6,6,4",
                            "save_every_n_steps": 50,
                            "save": "adamw_step"
                        }
                    }
                ]
            },
            "width_comparison": {
                "name": "模型宽度对比实验",
                "description": "比较不同模型宽度对性能的影响",
                "experiments": [
                    {
                        "name": "width_32",
                        "config": {
                            "model": "SYMUNET_PRETRAIN",
                            "dataset": "UCMerced",
                            "scale": 4,
                            "epochs": 200,
                            "batch_size": 4,
                            "optimizer": "ADAMW",
                            "scheduler": "cosine",
                            "lr": 2e-4,
                            "cosine_t_max": 200,
                            "loss": "1*L1+0.005*FFT",
                            "symunet_pretrain_width": 32,
                            "symunet_pretrain_enc_blk_nums": "4,6,6",
                            "symunet_pretrain_dec_blk_nums": "6,6,4",
                            "save_every_n_steps": 50,
                            "save": "width_32"
                        }
                    },
                    {
                        "name": "width_48",
                        "config": {
                            "model": "SYMUNET_PRETRAIN",
                            "dataset": "UCMerced",
                            "scale": 4,
                            "epochs": 200,
                            "batch_size": 4,
                            "optimizer": "ADAMW",
                            "scheduler": "cosine",
                            "lr": 2e-4,
                            "cosine_t_max": 200,
                            "loss": "1*L1+0.005*FFT",
                            "symunet_pretrain_width": 48,
                            "symunet_pretrain_enc_blk_nums": "4,6,6",
                            "symunet_pretrain_dec_blk_nums": "6,6,4",
                            "save_every_n_steps": 50,
                            "save": "width_48"
                        }
                    },
                    {
                        "name": "width_64",
                        "config": {
                            "model": "SYMUNET_PRETRAIN",
                            "dataset": "UCMerced",
                            "scale": 4,
                            "epochs": 200,
                            "batch_size": 4,
                            "optimizer": "ADAMW",
                            "scheduler": "cosine",
                            "lr": 2e-4,
                            "cosine_t_max": 200,
                            "loss": "1*L1+0.005*FFT",
                            "symunet_pretrain_width": 64,
                            "symunet_pretrain_enc_blk_nums": "4,6,6",
                            "symunet_pretrain_dec_blk_nums": "6,6,4",
                            "save_every_n_steps": 50,
                            "save": "width_64"
                        }
                    }
                ]
            },
            "loss_comparison": {
                "name": "损失函数对比实验",
                "description": "比较不同损失函数组合的效果",
                "experiments": [
                    {
                        "name": "loss_l1_only",
                        "config": {
                            "model": "SYMUNET_PRETRAIN",
                            "dataset": "UCMerced",
                            "scale": 4,
                            "epochs": 200,
                            "batch_size": 4,
                            "optimizer": "ADAMW",
                            "scheduler": "cosine",
                            "lr": 2e-4,
                            "cosine_t_max": 200,
                            "loss": "1*L1",
                            "symunet_pretrain_width": 48,
                            "symunet_pretrain_enc_blk_nums": "4,6,6",
                            "symunet_pretrain_dec_blk_nums": "6,6,4",
                            "save_every_n_steps": 50,
                            "save": "loss_l1_only"
                        }
                    },
                    {
                        "name": "loss_l1_fft_005",
                        "config": {
                            "model": "SYMUNET_PRETRAIN",
                            "dataset": "UCMerced",
                            "scale": 4,
                            "epochs": 200,
                            "batch_size": 4,
                            "optimizer": "ADAMW",
                            "scheduler": "cosine",
                            "lr": 2e-4,
                            "cosine_t_max": 200,
                            "loss": "1*L1+0.005*FFT",
                            "symunet_pretrain_width": 48,
                            "symunet_pretrain_enc_blk_nums": "4,6,6",
                            "symunet_pretrain_dec_blk_nums": "6,6,4",
                            "save_every_n_steps": 50,
                            "save": "loss_l1_fft_005"
                        }
                    },
                    {
                        "name": "loss_l1_fft_01",
                        "config": {
                            "model": "SYMUNET_PRETRAIN",
                            "dataset": "UCMerced",
                            "scale": 4,
                            "epochs": 200,
                            "batch_size": 4,
                            "optimizer": "ADAMW",
                            "scheduler": "cosine",
                            "lr": 2e-4,
                            "cosine_t_max": 200,
                            "loss": "1*L1+0.01*FFT",
                            "symunet_pretrain_width": 48,
                            "symunet_pretrain_enc_blk_nums": "4,6,6",
                            "symunet_pretrain_dec_blk_nums": "6,6,4",
                            "save_every_n_steps": 50,
                            "save": "loss_l1_fft_01"
                        }
                    }
                ]
            }
        }

    def list_presets(self):
        """列出所有预设配置"""
        presets = self.get_presets()

        print("🚀 Available Presets:")
        print("=" * 60)

        for key, preset in presets.items():
            print(f"\n📦 {key}")
            print(f"   Name: {preset['name']}")
            print(f"   Description: {preset['description']}")
            print(f"   Experiments: {len(preset['experiments'])}")

    def run_preset(self, preset_key: str, use_wandb: bool = True, dry_run: bool = False):
        """运行预设实验"""
        presets = self.get_presets()

        if preset_key not in presets:
            print(f"❌ Preset '{preset_key}' not found")
            return

        preset = presets[preset_key]
        experiments = preset['experiments']

        print(f"\n🧪 Running preset: {preset['name']}")
        print(f"📋 Description: {preset['description']}")
        print(f"🔢 Experiments: {len(experiments)}")

        # 显示实验列表
        print(f"\n📋 Experiment List:")
        for i, exp in enumerate(experiments, 1):
            print(f"   {i}. {exp['name']}")

        # 询问确认
        if not dry_run:
            response = input(f"\nDo you want to run {len(experiments)} experiments? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("❌ Cancelled")
                return

        # 执行实验
        for i, experiment in enumerate(experiments, 1):
            exp_name = experiment['name']
            config = experiment['config']

            # 添加WandB配置
            if use_wandb:
                config['use_wandb'] = True
                config['wandb_project'] = f"SymUNet-{preset_key}"
                config['wandb_name'] = exp_name

            print(f"\n{'='*60}")
            print(f"🔬 Experiment {i}/{len(experiments)}: {exp_name}")
            print(f"{'='*60}")

            # 显示配置
            print(f"📋 Configuration:")
            for key, value in config.items():
                print(f"   {key}: {value}")

            if dry_run:
                print(f"🔍 Dry run - would run:")
                cmd = self.build_command(config)
                print(f"   Command: {cmd}")
                continue

            # 执行训练
            try:
                cmd = self.build_command(config)
                print(f"\n🚀 Starting training...")

                result = subprocess.run(cmd, shell=True, check=True)
                print(f"✅ Experiment {exp_name} completed successfully")

            except subprocess.CalledProcessError as e:
                print(f"❌ Experiment {exp_name} failed: {e}")
            except KeyboardInterrupt:
                print(f"⚠️ Training interrupted by user")
                break

        print(f"\n🎉 Preset '{preset_key}' completed!")

    def build_command(self, config: dict) -> str:
        """构建训练命令"""
        cmd_parts = ["python", "train_enhanced.py"]

        for key, value in config.items():
            if isinstance(value, bool):
                if value:
                    cmd_parts.append(f"--{key}")
            else:
                cmd_parts.append(f"--{key}")
                cmd_parts.append(str(value))

        return " ".join(cmd_parts)

    def run_all_presets(self, use_wandb: bool = True, dry_run: bool = False):
        """运行所有预设实验"""
        presets = self.get_presets()

        print(f"🚀 Running ALL presets ({len(presets)} total)")
        print(f"⚠️ This will take approximately {len(presets) * 3:.0f} hours")

        if not dry_run:
            response = input("Do you want to continue? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("❌ Cancelled")
                return

        for preset_key in presets.keys():
            print(f"\n{'#'*80}")
            print(f"🎯 Processing preset: {preset_key}")
            print(f"{'#'*80}")

            self.run_preset(preset_key, use_wandb, dry_run)


def main():
    parser = argparse.ArgumentParser(description="快速批量训练脚本")
    parser.add_argument("--preset", type=str, help="要运行的预设配置")
    parser.add_argument("--list", action="store_true", help="列出所有可用预设")
    parser.add_argument("--all", action="store_true", help="运行所有预设")
    parser.add_argument("--no-wandb", action="store_true", help="禁用WandB")
    parser.add_argument("--dry-run", action="store_true", help="试运行（不实际执行）")

    args = parser.parse_args()

    trainer = QuickBatchTrainer()

    if args.list:
        trainer.list_presets()
    elif args.all:
        trainer.run_all_presets(use_wandb=not args.no_wandb, dry_run=args.dry_run)
    elif args.preset:
        trainer.run_preset(args.preset, use_wandb=not args.no_wandb, dry_run=args.dry_run)
    else:
        print("🚀 Quick Batch Training")
        print("\nUsage:")
        print("  python quick_batch.py --list                    # 列出所有预设")
        print("  python quick_batch.py --preset lr_comparison     # 运行学习率对比实验")
        print("  python quick_batch.py --all                     # 运行所有预设实验")
        print("  python quick_batch.py --preset optimizer_comparison --dry-run  # 试运行")
        print("\nAvailable presets:")
        trainer.list_presets()


if __name__ == "__main__":
    main()
