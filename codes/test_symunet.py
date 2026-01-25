#!/usr/bin/env python
"""
SymUNet Test Script
测试SymUNet模型的创建、前向传播和参数调节功能
"""

import torch
from model.symunet import make_model
from option import args

def test_model_creation():
    """测试模型创建"""
    print("=== Testing Model Creation ===")
    model = make_model(args)
    print(f"✓ Model created successfully: {type(model)}")
    print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model

def test_forward_pass(model):
    """测试前向传播"""
    print("\n=== Testing Forward Pass ===")
    model.eval()
    test_input = torch.randn(1, 3, 48, 48)
    print(f"Input shape: {test_input.shape}")

    with torch.no_grad():
        output = model(test_input)
        print(f"Output shape: {output.shape}")
        print(f"✓ Forward pass successful!")

    return output

def test_parameter_tuning():
    """测试参数调节"""
    print("\n=== Testing Parameter Tuning ===")

    # 测试不同的宽度参数
    print("1. Testing different widths:")
    for width in [32, 64]:
        args.symunet_width = width
        model = make_model(args)
        params = sum(p.numel() for p in model.parameters())
        print(f"   Width={width}: {params:,} parameters")

    # 测试不同的编码器/解码器深度
    print("\n2. Testing different depths:")
    for depth in [[1,1,1], [2,2,2]]:
        args.symunet_enc_blk_nums = depth
        args.symunet_dec_blk_nums = depth
        model = make_model(args)
        params = sum(p.numel() for p in model.parameters())
        print(f"   Depth={depth}: {params:,} parameters")

    print("✓ Parameter tuning test passed!")

def main():
    """主测试函数"""
    print("SymUNet Model Test")
    print("=" * 50)

    # 创建模型
    model = test_model_creation()

    # 测试前向传播
    test_forward_pass(model)

    # 测试参数调节
    test_parameter_tuning()

    print("\n" + "=" * 50)
    print("All tests passed! ✓")

if __name__ == "__main__":
    main()