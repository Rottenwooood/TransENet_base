#!/usr/bin/env python3
"""
批量训练脚本 - 支持超参数管理和串行训练

这个脚本允许你：
1. 定义超参数网格
2. 自动生成实验配置
3. 串行执行多个训练任务
4. 管理实验结果
5. 比较不同超参数的性能

使用方法:
python batch_train.py --config experiments_config.json
python batch_train.py --quick # 使用默认配置
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from itertools import product


class ExperimentManager:
    def __init__(self, config_file: str = None):
        self.experiments_dir = Path("../experiment")
        self.experiments_dir.mkdir(exist_ok=True)
        self.results_file = self.experiments_dir / "experiment_results.csv"
        self.config_file = config_file

        # 加载配置
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """加载实验配置"""
        if self.config_file and os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        # 默认配置
        return {
            "base_config": {
                "model": "SYMUNET_PRETRAIN",
                "dataset": "UCMerced",
                "scale": 4,
                "epochs": 300,
                "batch_size": 4,
                "ext": "img",
                "patch_size": 192,
                "resume": 0
            },
            "hyperparameter_grid": {
                "optimizer": ["ADAMW", "ADAM"],
                "scheduler": ["cosine", "step"],
                "lr": [1e-4, 2e-4, 5e-4],
                "loss": [
                    "1*L1",
                    "1*L1+0.005*FFT",
                    "1*L1+0.01*FFT"
                ],
                "symunet_pretrain_width": [32, 48, 64],
                "symunet_pretrain_enc_blk_nums": [
                    "2,2,2",
                    "4,4,4",
                    "4,6,6"
                ],
                "symunet_pretrain_dec_blk_nums": [
                    "2,2,2",
                    "4,4,4",
                    "6,6,4"
                ]
            },
            "experiment_prefix": "batch_exp",
            "max_experiments": 20,  # 限制最大实验数量
            "use_wandb": True,
            "wandb_project": "SymUNet-Batch",
            "save_every_n_steps": 50,
            "run_name_pattern": "{prefix}_lr{lr}_opt{optimizer}_sch{scheduler}_w{symunet_pretrain_width}"
        }

    def generate_experiments(self) -> List[Dict[str, Any]]:
        """生成实验配置网格"""
        base_config = self.config["base_config"]
        grid = self.config["hyperparameter_grid"]

        # 检查并处理成对参数
        paired_experiments, remaining_grid = self.handle_paired_parameters(grid)

        # 如果有成对参数，使用成对生成逻辑
        if paired_experiments:
            experiments = []
            for i, (enc_config, dec_config) in enumerate(paired_experiments):
                # 处理剩余参数
                if remaining_grid:
                    keys = list(remaining_grid.keys())
                    values = [remaining_grid[key] for key in keys]
                    other_combinations = list(product(*values))

                    for other_comb in other_combinations:
                        exp_config = base_config.copy()
                        exp_name = self.generate_paired_experiment_name(i, dict(zip(keys, other_comb)))

                        # 设置成对参数
                        exp_config["symunet_pretrain_enc_blk_nums"] = enc_config
                        exp_config["symunet_pretrain_dec_blk_nums"] = dec_config

                        # 设置其他参数
                        for key, value in zip(keys, other_comb):
                            exp_config[key] = value

                        # 添加WandB配置
                        if self.config["use_wandb"]:
                            exp_config["use_wandb"] = True
                            exp_config["wandb_project"] = self.config["wandb_project"]
                            exp_config["wandb_name"] = exp_name

                        # 添加其他配置
                        exp_config["save_every_n_steps"] = self.config["save_every_n_steps"]
                        exp_config["save"] = exp_name

                        experiments.append({
                            "id": len(experiments) + 1,
                            "name": exp_name,
                            "config": exp_config
                        })
                else:
                    # 只有成对参数
                    exp_config = base_config.copy()
                    exp_name = self.generate_paired_experiment_name(i, enc_config, dec_config, {})

                    # 设置成对参数
                    exp_config["symunet_pretrain_enc_blk_nums"] = enc_config
                    exp_config["symunet_pretrain_dec_blk_nums"] = dec_config

                    # 添加WandB配置
                    if self.config["use_wandb"]:
                        exp_config["use_wandb"] = True
                        exp_config["wandb_project"] = self.config["wandb_project"]
                        exp_config["wandb_name"] = exp_name

                    # 添加其他配置
                    exp_config["save_every_n_steps"] = self.config["save_every_n_steps"]
                    exp_config["save"] = exp_name

                    experiments.append({
                        "id": len(experiments) + 1,
                        "name": exp_name,
                        "config": exp_config
                    })
        else:
            # 使用原有的组合逻辑（当没有成对参数时）
            keys = list(grid.keys())
            values = [grid[key] for key in keys]

            all_combinations = list(product(*values))

            # 限制实验数量
            if len(all_combinations) > self.config["max_experiments"]:
                import random
                random.seed(42)
                all_combinations = random.sample(all_combinations, self.config["max_experiments"])

            experiments = []
            for i, combination in enumerate(all_combinations):
                exp_config = base_config.copy()
                exp_name = self.generate_experiment_name(i, dict(zip(keys, combination)))

                # 添加超参数
                for key, value in zip(keys, combination):
                    exp_config[key] = value

                # 添加WandB配置
                if self.config["use_wandb"]:
                    exp_config["use_wandb"] = True
                    exp_config["wandb_project"] = self.config["wandb_project"]
                    exp_config["wandb_name"] = exp_name

                # 添加其他配置
                exp_config["save_every_n_steps"] = self.config["save_every_n_steps"]
                exp_config["save"] = exp_name

                experiments.append({
                    "id": i + 1,
                    "name": exp_name,
                    "config": exp_config
                })

        return experiments

    def handle_paired_parameters(self, grid: Dict[str, Any]) -> tuple:
        """处理成对参数，返回成对配置和剩余参数"""
        enc_key = None
        dec_key = None

        # 查找编码器/解码器参数
        for key in grid.keys():
            if 'enc_blk_nums' in key:
                enc_key = key
            elif 'dec_blk_nums' in key:
                dec_key = key

        if not (enc_key and dec_key):
            return [], grid  # 没有找到成对参数，返回空列表和原始grid

        enc_values = grid[enc_key]
        dec_values = grid[dec_key]

        # 验证成对参数
        if len(enc_values) != len(dec_values):
            raise ValueError(
                f"❌ 编码器和解码器参数数量不匹配！\n"
                f"   {enc_key}: {len(enc_values)} 个值 -> {enc_values}\n"
                f"   {dec_key}: {len(dec_values)} 个值 -> {dec_values}\n"
                f"   解决方案：确保两个参数列表长度相等，并且一一对应配对。"
            )

        # 检查是否成对配置
        paired_configs = []
        for i, (enc_val, dec_val) in enumerate(zip(enc_values, dec_values)):
            enc_depths = enc_val.split(',')
            dec_depths = dec_val.split(',')

            if len(enc_depths) != len(dec_depths):
                raise ValueError(
                    f"❌ 成对参数第 {i+1} 配置深度不匹配！\n"
                    f"   编码器: {enc_val} ({len(enc_depths)} 层)\n"
                    f"   解码器: {dec_val} ({len(dec_depths)} 层)\n"
                    f"   解决方案：确保编码器和解码器有相同的层数。"
                )

            # 验证是否为合理的配对
            is_reasonable_pair = self.is_reasonable_encoder_decoder_pair(enc_val, dec_val)
            if not is_reasonable_pair:
                print(f"⚠️ 警告：第 {i+1} 对配置可能不是最佳配对：")
                print(f"   编码器: {enc_val} -> 解码器: {dec_val}")
                print(f"   建议使用对称配置，如：")
                print(f"   编码器: {enc_val} -> 解码器: {'-'.join(reversed(enc_depths))}")
                print(f"   继续使用当前配置...\n")

            paired_configs.append((enc_val, dec_val))

        # 移除成对参数，创建剩余参数grid
        remaining_grid = {k: v for k, v in grid.items() if k != enc_key and k != dec_key}

        return paired_configs, remaining_grid

    def is_reasonable_encoder_decoder_pair(self, enc_config: str, dec_config: str) -> bool:
        """检查编码器-解码器配对是否合理"""
        enc_depths = [int(x) for x in enc_config.split(',')]
        dec_depths = [int(x) for x in dec_config.split(',')]

        # 理想情况：解码器应该是编码器的反向
        expected_dec = list(reversed(enc_depths))

        # 如果完全匹配，认为是合理的
        return dec_depths == expected_dec

    def generate_paired_experiment_name(self, pair_idx: int, other_params: Dict[str, Any]) -> str:
        """生成成对参数的实验名称"""
        pattern = self.config["run_name_pattern"]
        name = pattern.format(
            prefix=self.config["experiment_prefix"],
            **other_params
        )
        # 清理特殊字符
        name = name.replace(".", "p").replace("+", "p").replace("*", "x")
        return f"{pair_idx+1:03d}_{name}"
        
    def generate_experiment_name(self, index: int, params: Dict[str, Any]) -> str:
        """生成实验名称"""
        pattern = self.config["run_name_pattern"]
        name = pattern.format(
            prefix=self.config["experiment_prefix"],
            **params
        )
        # 清理特殊字符
        name = name.replace(".", "p").replace("+", "p").replace("*", "x")
        return f"{self.config['experiment_prefix']}_{index+1:03d}_{name}"

    def run_experiment(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个实验"""
        exp_id = experiment["id"]
        exp_name = experiment["name"]
        config = experiment["config"]

        print(f"\n{'='*80}")
        print(f"🧪 Running Experiment {exp_id}/{len(self.all_experiments)}: {exp_name}")
        print(f"{'='*80}")
        print(f"📋 Configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")

        start_time = time.time()

        # 构建训练命令
        cmd = self.build_training_command(config)

        # 执行训练
        try:
            print(f"\n🚀 Starting training...")
            result = subprocess.run(cmd, shell=True, capture_output=False, text=True)

            end_time = time.time()
            duration = end_time - start_time

            # 解析结果
            success = result.returncode == 0
            status = "SUCCESS" if success else "FAILED"

            print(f"\n✅ Experiment {exp_name} completed in {duration/3600:.2f} hours")
            print(f"Status: {status}")

            return {
                "id": exp_id,
                "name": exp_name,
                "status": status,
                "duration": duration,
                "config": config,
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.fromtimestamp(end_time).isoformat()
            }

        except Exception as e:
            print(f"❌ Experiment {exp_name} failed: {e}")
            return {
                "id": exp_id,
                "name": exp_name,
                "status": "ERROR",
                "error": str(e),
                "config": config,
                "start_time": datetime.fromtimestamp(start_time).isoformat()
            }

    def build_training_command(self, config: Dict[str, Any]) -> str:
        """构建训练命令"""
        cmd_parts = ["python", "train_enhanced.py"]

        # 添加参数
        for key, value in config.items():
            if key.startswith("_"):  # 跳过内部参数
                continue

            if isinstance(value, bool):
                if value:
                    cmd_parts.append(f"--{key}")
            elif isinstance(value, list):
                cmd_parts.append(f"--{key}")
                cmd_parts.append(str(value[0]) if len(value) == 1 else " ".join(map(str, value)))
            else:
                cmd_parts.append(f"--{key}")
                cmd_parts.append(str(value))

        return " ".join(cmd_parts)

    def save_results(self, results: List[Dict[str, Any]]):
        """保存实验结果"""
        if not results:
            return

        # 转换为DataFrame
        df_data = []
        for result in results:
            row = {
                "experiment_id": result["id"],
                "experiment_name": result["name"],
                "status": result["status"],
                "duration_hours": result.get("duration", 0) / 3600,
                "start_time": result.get("start_time", ""),
                "end_time": result.get("end_time", "")
            }

            # 添加配置参数
            for key, value in result.get("config", {}).items():
                row[f"config_{key}"] = str(value)

            df_data.append(row)

        df = pd.DataFrame(df_data)

        # 保存到CSV
        if self.results_file.exists():
            # 如果文件存在，追加数据
            df.to_csv(self.results_file, mode='a', header=False, index=False)
        else:
            # 如果文件不存在，创建新文件
            df.to_csv(self.results_file, index=False)

        print(f"\n📊 Results saved to {self.results_file}")

        # 显示结果摘要
        self.print_results_summary(results)

    def print_results_summary(self, results: List[Dict[str, Any]]):
        """打印结果摘要"""
        print(f"\n{'='*80}")
        print(f"📊 EXPERIMENT RESULTS SUMMARY")
        print(f"{'='*80}")

        successful = [r for r in results if r["status"] == "SUCCESS"]
        failed = [r for r in results if r["status"] != "SUCCESS"]

        print(f"✅ Successful experiments: {len(successful)}")
        print(f"❌ Failed experiments: {len(failed)}")
        print(f"📈 Success rate: {len(successful)/len(results)*100:.1f}%")

        if successful:
            total_time = sum(r.get("duration", 0) for r in successful)
            avg_time = total_time / len(successful)
            print(f"⏱️ Average training time: {avg_time/3600:.2f} hours")
            print(f"🕐 Total training time: {total_time/3600:.2f} hours")

        # 显示每个实验的结果
        print(f"\n📋 Individual Results:")
        print(f"{'ID':<4} {'Name':<40} {'Status':<10} {'Duration (h)':<12}")
        print("-" * 80)
        for result in results:
            duration_h = result.get("duration", 0) / 3600
            print(f"{result['id']:<4} {result['name'][:39]:<40} {result['status']:<10} {duration_h:<12.2f}")

    def run_all_experiments(self, experiment_ids: List[int] = None):
        """运行所有实验"""
        self.all_experiments = self.generate_experiments()

        # 过滤实验ID
        if experiment_ids:
            self.all_experiments = [exp for exp in self.all_experiments if exp["id"] in experiment_ids]

        print(f"🧪 Generated {len(self.all_experiments)} experiments")
        print(f"📁 Results will be saved to {self.experiments_dir}")

        # 显示所有实验配置
        print(f"\n📋 Experiment List:")
        for exp in self.all_experiments:
            print(f"   {exp['id']}: {exp['name']}")

        # 询问用户确认
        if not self.confirm_execution():
            print("❌ Experiment execution cancelled")
            return

        # 执行实验
        results = []
        for experiment in self.all_experiments:
            result = self.run_experiment(experiment)
            results.append(result)

            # 保存中间结果
            self.save_results([result])

        # 保存最终结果
        self.save_results(results)

        print(f"\n🎉 All experiments completed!")

    def confirm_execution(self) -> bool:
        """确认执行实验"""
        total_experiments = len(self.all_experiments)
        print(f"\n⚠️ About to run {total_experiments} experiments")
        print(f"This will take approximately {total_experiments * 4:.0f} hours")

        response = input("Do you want to continue? (y/N): ").lower().strip()
        return response in ['y', 'yes', 'yes,', '1', 'true']


def create_sample_config():
    """创建示例配置文件"""
    sample_config = {
        "base_config": {
            "model": "SYMUNET_PRETRAIN",
            "dataset": "UCMerced",
            "scale": 4,
            "epochs": 300,
            "batch_size": 4,
            "ext": "img",
            "patch_size": 192,
            "resume": 0
        },
        "hyperparameter_grid": {
            "optimizer": ["ADAMW", "ADAM"],
            "scheduler": ["cosine", "step"],
            "lr": [1e-4, 2e-4],
            "loss": [
                "1*L1",
                "1*L1+0.005*FFT"
            ],
            "symunet_pretrain_width": [32, 48],
            "symunet_pretrain_enc_blk_nums": [
                "2,2,2",
                "4,4,4"
            ],
            "symunet_pretrain_dec_blk_nums": [
                "2,2,2",
                "4,4,4"
            ]
        },
        "experiment_prefix": "quick_exp",
        "max_experiments": 8,
        "use_wandb": True,
        "wandb_project": "SymUNet-Quick",
        "save_every_n_steps": 50,
        "run_name_pattern": "{prefix}_lr{lr}_opt{optimizer}_sch{scheduler}_w{width}"
    }

    with open("batch_config.json", "w", encoding="utf-8") as f:
        json.dump(sample_config, f, indent=2, ensure_ascii=False)

    print("✅ Sample configuration saved to batch_config.json")


def main():
    parser = argparse.ArgumentParser(description="批量训练脚本")
    parser.add_argument("--config", type=str, help="实验配置文件")
    parser.add_argument("--quick", action="store_true", help="快速模式（使用默认配置）")
    parser.add_argument("--create-config", action="store_true", help="创建示例配置文件")
    parser.add_argument("--list", action="store_true", help="列出所有实验但不执行")
    parser.add_argument("--ids", type=str, help="要执行的实验ID列表，格式：1,2,3")
    parser.add_argument("--dry-run", action="store_true", help="试运行（不实际执行训练）")

    args = parser.parse_args()

    # 创建示例配置
    if args.create_config:
        create_sample_config()
        return

    # 初始化实验管理器
    manager = ExperimentManager(args.config)

    # 生成实验
    experiments = manager.generate_experiments()

    if args.list:
        print(f"📋 Generated {len(experiments)} experiments:")
        for exp in experiments:
            print(f"   {exp['id']}: {exp['name']}")
            for key, value in exp['config'].items():
                if key in ['optimizer', 'scheduler', 'lr', 'symunet_pretrain_width']:
                    print(f"      {key}: {value}")
        return

    # 过滤实验ID
    experiment_ids = None
    if args.ids:
        experiment_ids = [int(x.strip()) for x in args.ids.split(",")]

    if args.dry_run:
        print("🔍 Dry run mode - showing experiments that would be run:")
        for exp in experiments:
            if experiment_ids is None or exp['id'] in experiment_ids:
                print(f"\n{exp['id']}: {exp['name']}")
                print(f"   Command: {manager.build_training_command(exp['config'])}")
        return

    # 运行实验
    manager.run_all_experiments(experiment_ids)


if __name__ == "__main__":
    main()
