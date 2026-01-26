#!/usr/bin/env python
"""
测试SymUNet的两个超分辨率版本
验证模型能否正常初始化和前向推理
"""

import torch
import torch.nn as nn
import sys
import os

# 添加路径
sys.path.append('/home/c6h4o2/dev/TransENet_base/codes')

from model.symunet_pretrain import SymUNet_Pretrain
from model.symunet_posttrain import SymUNet_Posttrain
from option import args

def test_model(model, model_name, input_size, device):
    """测试模型"""
    print(f"\n{'='*60}")
    print(f"测试 {model_name}")
    print(f"{'='*60}")

    model = model.to(device)
    model.eval()

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 创建输入
    input_lr = torch.rand(1, 3, *input_size).to(device)
    print(f"输入LR尺寸: {input_lr.size()}")

    # 前向推理
    with torch.no_grad():
        output = model(input_lr)

    expected_h = input_size[0] * args.scale[0]
    expected_w = input_size[1] * args.scale[0]
    print(f"输出SR尺寸: {output.size()}")
    print(f"期望输出尺寸: [1, 3, {expected_h}, {expected_w}]")

    # 验证输出尺寸
    if output.size(2) == expected_h and output.size(3) == expected_w:
        print("✅ 输出尺寸验证通过!")
    else:
        print("❌ 输出尺寸验证失败!")
        return False

    # 验证输出范围
    print(f"输出值范围: [{output.min():.4f}, {output.max():.4f}]")

    return True

def main():
    """主函数"""
    print("SymUNet 超分辨率版本测试")
    print("=" * 60)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 设置测试参数
    test_scales = [2, 4, 8]
    test_sizes = [(32, 32), (48, 48), (64, 64)]

    results = []

    for scale in test_scales:
        args.scale = [scale]
        print(f"\n🔍 测试上采样倍数: {scale}x")

        for size in test_sizes:
            print(f"\n📐 测试输入尺寸: {size}")

            # 测试Pretrain版本
            print("\n" + "─" * 40)
            print("SymUNet-Pretrain (预上采样版本)")
            print("─" * 40)

            try:
                args.symunet_pretrain_width = 64
                args.symunet_pretrain_middle_blk_num = 1
                args.symunet_pretrain_enc_blk_nums = [2, 2, 2]
                args.symunet_pretrain_dec_blk_nums = [2, 2, 2]
                args.symunet_pretrain_ffn_expansion_factor = 2.66
                args.symunet_pretrain_bias = False
                args.symunet_pretrain_layer_norm_type = 'WithBias'
                args.symunet_pretrain_restormer_heads = [1, 2, 4]
                args.symunet_pretrain_restormer_middle_heads = 8

                model_pretrain = SymUNet_Pretrain(args).to(device)
                result_pretrain = test_model(model_pretrain, f"SymUNet-Pretrain ({scale}x)", size, device)
                results.append(("Pretrain", scale, size, result_pretrain))
            except Exception as e:
                print(f"❌ SymUNet-Pretrain 测试失败: {e}")
                results.append(("Pretrain", scale, size, False))

            # 测试Posttrain版本
            print("\n" + "─" * 40)
            print("SymUNet-Posttrain (后上采样版本)")
            print("─" * 40)

            try:
                args.symunet_posttrain_width = 64
                args.symunet_posttrain_middle_blk_num = 1
                args.symunet_posttrain_enc_blk_nums = [2, 2, 2]
                args.symunet_posttrain_dec_blk_nums = [2, 2, 2]
                args.symunet_posttrain_ffn_expansion_factor = 2.66
                args.symunet_posttrain_bias = False
                args.symunet_posttrain_layer_norm_type = 'WithBias'
                args.symunet_posttrain_restormer_heads = [1, 2, 4]
                args.symunet_posttrain_restormer_middle_heads = 8

                model_posttrain = SymUNet_Posttrain(args).to(device)
                result_posttrain = test_model(model_posttrain, f"SymUNet-Posttrain ({scale}x)", size, device)
                results.append(("Posttrain", scale, size, result_posttrain))
            except Exception as e:
                print(f"❌ SymUNet-Posttrain 测试失败: {e}")
                results.append(("Posttrain", scale, size, False))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    total_tests = len(results)
    passed_tests = sum(1 for _, _, _, result in results if result)
    failed_tests = total_tests - passed_tests

    print(f"总测试数: {total_tests}")
    print(f"通过: {passed_tests}")
    print(f"失败: {failed_tests}")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%")

    print("\n详细结果:")
    print("-" * 60)
    for version, scale, size, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{version:10s} | Scale: {scale:2d}x | Size: {str(size):20s} | {status}")

    if failed_tests == 0:
        print("\n🎉 所有测试通过!")
        return 0
    else:
        print(f"\n⚠️  {failed_tests} 个测试失败")
        return 1

if __name__ == "__main__":
    exit(main())