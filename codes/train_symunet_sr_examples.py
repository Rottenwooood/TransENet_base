#!/usr/bin/env python
"""
SymUNet 超分辨率版本训练示例
展示如何使用SymUNet-Pretrain和SymUNet-Posttrain进行训练
"""

import os
import sys
import argparse

# 添加路径
sys.path.append('/home/c6h4o2/dev/TransENet_base/codes')

def create_train_command(model_type, scale=4, width=64, batch_size=16, epochs=500):
    """创建训练命令"""

    if model_type.upper() == "PRETRAIN":
        model_name = "SYMUNET_PRETRAIN"
        width_param = f"--symunet_pretrain_width {width}"
        enc_blks = "--symunet_pretrain_enc_blk_nums 2,2,2"
        dec_blks = "--symunet_pretrain_dec_blk_nums 2,2,2"
        heads = "--symunet_pretrain_restormer_heads 1,2,4"
        middle_heads = "--symunet_pretrain_restormer_middle_heads 8"
        ffn_factor = "--symunet_pretrain_ffn_expansion_factor 2.66"
        bias = "--symunet_pretrain_bias False"
        ln_type = "--symunet_pretrain_layer_norm_type WithBias"

    elif model_type.upper() == "POSTTRAIN":
        model_name = "SYMUNET_POSTTRAIN"
        width_param = f"--symunet_posttrain_width {width}"
        enc_blks = "--symunet_posttrain_enc_blk_nums 2,2,2"
        dec_blks = "--symunet_posttrain_dec_blk_nums 2,2,2"
        heads = "--symunet_posttrain_restormer_heads 1,2,4"
        middle_heads = "--symunet_posttrain_restormer_middle_heads 8"
        ffn_factor = "--symunet_posttrain_ffn_expansion_factor 2.66"
        bias = "--symunet_posttrain_bias False"
        ln_type = "--symunet_posttrain_layer_norm_type WithBias"

    else:
        raise ValueError(f"未知模型类型: {model_type}")

    # 基础命令
    cmd = f"""python train.py \\
    --model {model_name} \\
    --scale {scale} \\
    {width_param} \\
    {enc_blks} \\
    {dec_blks} \\
    {heads} \\
    {middle_heads} \\
    {ffn_factor} \\
    {bias} \\
    {ln_type} \\
    --epochs {epochs} \\
    --batch_size {batch_size} \\
    --lr 1e-4 \\
    --loss 1*L1 \\
    --dir_data /path/to/dataset \\
    --data_train /path/to/train \\
    --data_val /path/to/val \\
    --save symunet_{model_type.lower()}_x{scale}_w{width}"""

    return cmd

def print_examples():
    """打印使用示例"""
    print("=" * 80)
    print("SymUNet 超分辨率版本训练示例")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("方案1: SymUNet-Pretrain (预上采样版本)")
    print("=" * 80)

    print("\n【示例1】标准配置 - 4x超分辨率")
    print("-" * 80)
    cmd1 = create_train_command("pretrain", scale=4, width=64, batch_size=8, epochs=500)
    print(cmd1)

    print("\n【示例2】高质量配置 - 4x超分辨率")
    print("-" * 80)
    cmd2 = create_train_command("pretrain", scale=4, width=128, batch_size=4, epochs=1000)
    print(cmd2)

    print("\n【示例3】轻量级配置 - 2x超分辨率")
    print("-" * 80)
    cmd3 = create_train_command("pretrain", scale=2, width=32, batch_size=16, epochs=300)
    print(cmd3)

    print("\n" + "=" * 80)
    print("方案2: SymUNet-Posttrain (后上采样版本)")
    print("=" * 80)

    print("\n【示例4】高效配置 - 4x超分辨率")
    print("-" * 80)
    cmd4 = create_train_command("posttrain", scale=4, width=64, batch_size=16, epochs=500)
    print(cmd4)

    print("\n【示例5】大尺度配置 - 8x超分辨率")
    print("-" * 80)
    cmd5 = create_train_command("posttrain", scale=8, width=64, batch_size=16, epochs=800)
    print(cmd5)

    print("\n【示例6】超高效配置 - 8x超分辨率")
    print("-" * 80)
    cmd6 = create_train_command("posttrain", scale=8, width=32, batch_size=32, epochs=500)
    print(cmd6)

    print("\n" + "=" * 80)
    print("高级配置示例")
    print("=" * 80)

    print("\n【示例7】Pretrain + 大模型 (需要高性能GPU)")
    print("-" * 80)
    cmd7 = """python train.py \\
    --model SYMUNET_PRETRAIN \\
    --scale 4 \\
    --symunet_pretrain_width 256 \\
    --symunet_pretrain_enc_blk_nums 3,3,3 \\
    --symunet_pretrain_dec_blk_nums 3,3,3 \\
    --symunet_pretrain_restormer_heads 2,4,8 \\
    --symunet_pretrain_restormer_middle_heads 16 \\
    --symunet_pretrain_ffn_expansion_factor 2.66 \\
    --symunet_pretrain_bias False \\
    --symunet_pretrain_layer_norm_type WithBias \\
    --epochs 1000 \\
    --batch_size 2 \\
    --lr 5e-5 \\
    --loss 1*L1+1*Perceptual \\
    --dir_data /path/to/dataset \\
    --data_train /path/to/train \\
    --data_val /path/to/val \\
    --save symunet_pretrain_x4_large"""
    print(cmd7)

    print("\n【示例8】Posttrain + 多损失函数")
    print("-" * 80)
    cmd8 = """python train.py \\
    --model SYMUNET_POSTTRAIN \\
    --scale 4 \\
    --symunet_posttrain_width 64 \\
    --symunet_posttrain_enc_blk_nums 2,2,2 \\
    --symunet_posttrain_dec_blk_nums 2,2,2 \\
    --symunet_posttrain_restormer_heads 1,2,4 \\
    --symunet_posttrain_restormer_middle_heads 8 \\
    --symunet_posttrain_ffn_expansion_factor 2.0 \\
    --symunet_posttrain_bias True \\
    --symunet_posttrain_layer_norm_type BiasFree \\
    --epochs 500 \\
    --batch_size 16 \\
    --lr 1e-4 \\
    --lr_decay 200 \\
    --decay_type step \\
    --loss 1*L1+0.1*GAN \\
    --dir_data /path/to/dataset \\
    --data_train /path/to/train \\
    --data_val /path/to/val \\
    --save symunet_posttrain_x4_gan"""
    print(cmd8)

def print_comparison():
    """打印方案对比"""
    print("\n" + "=" * 80)
    print("方案对比与选择建议")
    print("=" * 80)

    comparison_table = """
╔═══════════════════╦═══════════════════════════╦═══════════════════════════╗
║     特性对比       ║   SymUNet-Pretrain       ║   SymUNet-Posttrain      ║
╠═══════════════════╬═══════════════════════════╬═══════════════════════════╣
║ 上采样策略         ║  预上采样 (bicubic)       ║  后上采样 (PixelShuffle) ║
╠═══════════════════╬═══════════════════════════╬═══════════════════════════╣
║ 计算复杂度         ║  高 (HR空间处理)          ║  低 (LR空间处理)          ║
╠═══════════════════╬═══════════════════════════╬═══════════════════════════╣
║ 内存占用          ║  大                        ║  小                        ║
╠═══════════════════╬═══════════════════════════╬═══════════════════════════╣
║ 推理速度          ║  慢                        ║  快                        ║
╠═══════════════════╬═══════════════════════════╬═══════════════════════════╣
║ 图像质量          ║  高 (细节丰富)             ║  中等 (边缘平滑)          ║
╠═══════════════════╬═══════════════════════════╬═══════════════════════════╣
║ 适用尺度          ║  2x, 4x                   ║  4x, 8x, 16x+             ║
╠═══════════════════╬═══════════════════════════╬═══════════════════════════╣
║ 硬件要求          ║  高性能GPU                 ║  中等GPU                   ║
╠═══════════════════╬═══════════════════════════╬═══════════════════════════╣
║ 最佳场景          ║  高质量需求                ║  效率优先需求              ║
╚═══════════════════╩═══════════════════════════╩═══════════════════════════╝
    """
    print(comparison_table)

def print_parameter_guide():
    """打印参数调优指南"""
    print("\n" + "=" * 80)
    print("参数调优指南")
    print("=" * 80)

    print("\n🔧 网络宽度 (width)")
    print("-" * 80)
    print("轻量级: 32  -  适合快速实验和资源受限环境")
    print("标准级: 64  -  平衡性能和效率 (推荐)")
    print("大模型: 128 -  适合高质量需求")
    print("超大模型: 256+ - 需要高性能GPU")

    print("\n🔧 编码器/解码器深度 (enc_blk_nums / dec_blk_nums)")
    print("-" * 80)
    print("浅层: [1,1,1] - 快速训练，容易过拟合")
    print("标准: [2,2,2] - 平衡深度和性能 (推荐)")
    print("深层: [3,3,3] - 高精度，训练时间长")

    print("\n🔧 注意力头数 (restormer_heads)")
    print("-" * 80)
    print("少头数: [1,2,4]     - 轻量级模型")
    print("标准头数: [1,2,4]   - 推荐设置")
    print("多头数: [2,4,8]     - 增强特征提取能力")
    print("中间头数: 8/16      - 增强中间层特征融合")

    print("\n🔧 FFN扩展因子 (ffn_expansion_factor)")
    print("-" * 80)
    print("2.0  - 紧凑模型，减少参数量")
    print("2.66 - 标准设置 (推荐)")
    print("3.0  - 大容量，增强表达能力")

    print("\n🔧 LayerNorm类型")
    print("-" * 80)
    print("WithBias  - 标准LayerNorm，训练稳定 (推荐)")
    print("BiasFree  - 无偏置版本，可能更快")

    print("\n🔧 Batch Size建议")
    print("-" * 80)
    print("Pretrain:   4/8  - 内存占用大")
    print("Posttrain:  16/32 - 内存占用小")

    print("\n🔧 Learning Rate建议")
    print("-" * 80)
    print("Pretrain:   5e-5 ~ 1e-4")
    print("Posttrain:  1e-4 ~ 1e-3")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='SymUNet 超分辨率版本训练示例',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python train_symunet_sr_examples.py --examples     # 显示训练示例
  python train_symunet_sr_examples.py --comparison   # 显示方案对比
  python train_symunet_sr_examples.py --guide        # 显示参数指南
        """
    )

    parser.add_argument('--examples', action='store_true',
                       help='显示训练示例')
    parser.add_argument('--comparison', action='store_true',
                       help='显示方案对比')
    parser.add_argument('--guide', action='store_true',
                       help='显示参数调优指南')
    parser.add_argument('--all', action='store_true',
                       help='显示所有内容')

    args = parser.parse_args()

    # 如果没有参数，显示所有内容
    if not any([args.examples, args.comparison, args.guide, args.all]):
        args.all = True

    if args.all or args.examples:
        print_examples()

    if args.all or args.comparison:
        print_comparison()

    if args.all or args.guide:
        print_parameter_guide()

    print("\n" + "=" * 80)
    print("训练完成后，请使用 test_symunet_sr_versions.py 验证模型")
    print("=" * 80)

if __name__ == "__main__":
    main()