#!/usr/bin/env python3
"""
快速批量Deploy和PSNR计算脚本

自动检测checkpoint文件，简化命令行输入
"""

import os
import glob
import argparse


def auto_find_checkpoints(experiment_dir, step_list=None):
    """
    自动查找checkpoint文件

    Args:
        experiment_dir: 实验目录
        step_list: 指定step列表，如[17850, 23800, 29750]

    Returns:
        list: 找到的checkpoint文件列表
    """
    if step_list is None:
        # 默认查找这些step
        step_list = [17850, 23800, 29750]

    checkpoint_list = []
    for step in step_list:
        pattern = os.path.join(experiment_dir, f'checkpoint_step_{step}.pt')
        if os.path.exists(pattern):
            checkpoint_list.append(f'checkpoint_step_{step}.pt')
        else:
            print(f"警告: 未找到 checkpoint_step_{step}.pt")

    return checkpoint_list


def auto_find_latest_checkpoints(experiment_dir, count=3):
    """
    自动查找最新的N个checkpoint文件

    Args:
        experiment_dir: 实验目录
        count: 返回最新的N个checkpoint

    Returns:
        list: 最新的checkpoint文件列表
    """
    checkpoint_files = glob.glob(os.path.join(experiment_dir, 'checkpoint_step_*.pt'))

    # 提取step数字并排序
    checkpoints_with_step = []
    for cp_file in checkpoint_files:
        filename = os.path.basename(cp_file)
        try:
            # 从文件名中提取step数字
            step_str = filename.replace('checkpoint_step_', '').replace('.pt', '')
            step_num = int(step_str)
            checkpoints_with_step.append((step_num, filename))
        except ValueError:
            continue

    # 按step数字排序，取最新的N个
    checkpoints_with_step.sort(key=lambda x: x[0], reverse=True)
    latest_checkpoints = [cp[1] for cp in checkpoints_with_step[:count]]

    return latest_checkpoints


def main():
    parser = argparse.ArgumentParser(description='快速批量Deploy和PSNR计算')

    # 必需参数
    parser.add_argument('--pre_train', type=str, required=True,
                        help='预训练模型路径(目录或文件)')
    parser.add_argument('--dir_data', type=str, required=True,
                        help='LR图像目录')
    parser.add_argument('--dir_gt', type=str, required=True,
                        help='Ground Truth图像目录')
    parser.add_argument('--dir_out', type=str, required=True,
                        help='输出目录')

    # 可选参数
    parser.add_argument('--model', type=str, default='SYMUNET_PRETRAIN',
                        help='模型名称')
    parser.add_argument('--dataset', type=str, default='UCMerced',
                        help='数据集名称')
    parser.add_argument('--scale', type=int, default=4,
                        help='超分辨率倍数')
    parser.add_argument('--patch_size', type=int, default=192,
                        help='训练patch大小')
    parser.add_argument('--test_block', action='store_true',
                        help='使用分块测试')
    parser.add_argument('--cpu', action='store_true',
                        help='使用CPU模式')

    # Checkpoint选择方式
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument('--steps', type=int, nargs='+',
                                  default=[17850, 23800, 29750],
                                  help='指定要测试的step数字列表')
    checkpoint_group.add_argument('--latest', type=int,
                                  help='自动选择最新的N个checkpoint')

    args = parser.parse_args()

    print("="*60)
    print("快速批量Deploy和PSNR计算")
    print("="*60)

    # 确定experiment目录和pre_train路径
    if os.path.isdir(args.pre_train):
        experiment_dir = args.pre_train
        # 查找model目录下的pre_train文件
        pre_train_candidates = [
            os.path.join(experiment_dir, 'model', 'model_best.pt'),
            os.path.join(experiment_dir, 'model_best.pt'),
            os.path.join(experiment_dir, 'model', 'model_epoch_1.pt'),
        ]
        pre_train_path = None
        for candidate in pre_train_candidates:
            if os.path.exists(candidate):
                pre_train_path = candidate
                break

        if pre_train_path is None:
            print("错误: 在指定目录中未找到预训练模型文件")
            print("尝试的文件路径:")
            for candidate in pre_train_candidates:
                print(f"  - {candidate}")
            return
    else:
        experiment_dir = os.path.dirname(args.pre_train)
        pre_train_path = args.pre_train

    print(f"实验目录: {experiment_dir}")
    print(f"预训练模型: {pre_train_path}")

    # 自动查找checkpoint
    if args.latest:
        checkpoint_list = auto_find_latest_checkpoints(experiment_dir, args.latest)
        print(f"自动选择最新的 {len(checkpoint_list)} 个checkpoint")
    else:
        checkpoint_list = auto_find_checkpoints(experiment_dir, args.steps)
        print(f"使用指定的 {len(checkpoint_list)} 个checkpoint")

    if len(checkpoint_list) == 0:
        print("错误: 未找到任何checkpoint文件")
        return

    for i, cp in enumerate(checkpoint_list):
        print(f"  {i+1}. {cp}")

    # 生成完整的pre_train_list
    pre_train_list = pre_train_path

    # 生成checkpoint_list (逗号分隔)
    checkpoint_list_str = ','.join(checkpoint_list)

    # 构建最终命令
    cmd = f"""
python batch_deploy_psnr.py \\
    --model {args.model} \\
    --dataset {args.dataset} \\
    --scale {args.scale} \\
    --dir_data {args.dir_data} \\
    --dir_out_base {args.dir_out} \\
    --dir_gt {args.dir_gt} \\
    --pre_train_list "{pre_train_list}" \\
    --checkpoint_list "{checkpoint_list_str}"
"""

    if args.test_block:
        cmd += f" \\\n    --test_block \\\n    --patch_size {args.patch_size}"

    if args.cpu:
        cmd += " \\\n    --cpu"

    print("\n" + "="*60)
    print("执行命令:")
    print("="*60)
    print(cmd)

    print("\n" + "="*60)
    print("是否执行此命令? (y/n)")
    print("="*60)
    response = input("请输入: ").strip().lower()

    if response in ['y', 'yes', '是', '1']:
        print("\n开始执行...")
        os.system(cmd)
    else:
        print("\n已取消执行")


if __name__ == '__main__':
    main()
