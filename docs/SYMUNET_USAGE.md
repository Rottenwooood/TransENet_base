# SymUNet 集成使用指南

## 概述

SymUNet已成功集成到TransENet项目中，采用了单文件形式(`codes/model/symunet.py`)，与现有项目架构保持一致。

## 文件结构

```
codes/
├── model/
│   ├── symunet.py              # SymUNet模型定义
│   ├── __init__.py             # 模型导入管理
│   └── ...                     # 其他模型
├── option.py                   # 参数配置（已添加SymUNet参数）
├── test_symunet.py            # 模型测试脚本
└── train_symunet_example.py   # 训练集成示例
```

## 参数配置

### 基本参数
```bash
--model SYMUNET                    # 使用SymUNet模型
--symunet_width 64                 # 基础通道数（默认: 64）
--symunet_middle_blk_num 1         # 中间块数量（默认: 1）
```

### 网络结构参数
```bash
--symunet_enc_blk_nums 2,2,2      # 编码器各阶段块数
--symunet_dec_blk_nums 2,2,2      # 解码器各阶段块数
--symunet_restormer_heads 1,2,4   # 各阶段注意力头数
--symunet_restormer_middle_heads 8 # 中间层注意力头数
```

### Transformer参数
```bash
--symunet_ffn_expansion_factor 2.66  # FFN扩展因子（默认: 2.66）
--symunet_bias False                  # 是否使用偏置（默认: False）
--symunet_layer_norm_type WithBias    # Layer Norm类型（WithBias/BiasFree）
```

## 使用方法

### 1. 基本使用

```bash
# 基本训练命令
python train.py --model SYMUNET

# 使用自定义参数
python train.py \
    --model SYMUNET \
    --symunet_width 128 \
    --symunet_enc_blk_nums 3,3,3 \
    --symunet_restormer_heads 2,4,8 \
    --symunet_ffn_expansion_factor 2.0
```

### 2. 模型测试

```bash
# 运行模型测试脚本
python test_symunet.py
```

### 3. 训练集成示例

```bash
# 查看训练集成示例
python train_symunet_example.py
```

## 参数调节指南

### 网络宽度调节
```bash
# 轻量级模型（推荐用于资源有限环境）
--symunet_width 32

# 标准模型（平衡性能和效率）
--symunet_width 64

# 大型模型（推荐用于高质量需求）
--symunet_width 128
```

### 深度调节
```bash
# 浅层网络（快速训练）
--symunet_enc_blk_nums 1,1,1 --symunet_dec_blk_nums 1,1,1

# 标准深度（推荐）
--symunet_enc_blk_nums 2,2,2 --symunet_dec_blk_nums 2,2,2

# 深层网络（高精度）
--symunet_enc_blk_nums 3,3,3 --symunet_dec_blk_nums 3,3,3
```

### 注意力头数调节
```bash
# 少量注意力头（轻量级）
--symunet_restormer_heads 1,2,4

# 标准注意力头数（推荐）
--symunet_restormer_heads 1,2,4

# 大量注意力头（高精度）
--symunet_restormer_heads 2,4,8
```

## 模型参数统计

| 配置 | 宽度 | 编码器深度 | 参数量 |
|------|------|------------|--------|
| 轻量级 | 32 | [1,1,1] | ~2.2M |
| 标准 | 64 | [2,2,2] | ~8.8M |
| 大型 | 128 | [3,3,3] | ~34.9M |

## 特性说明

1. **完全集成**: 与现有TransENet训练框架无缝集成
2. **参数可调**: 所有关键参数都可以通过命令行调节
3. **兼容性**: 保持了与原有代码结构的兼容性
4. **测试验证**: 包含完整的测试脚本验证功能

## 注意事项

1. 确保使用conda环境: `conda activate symunet`
2. 大型模型需要足够的GPU内存
3. 根据数据集大小调整批处理大小
4. 建议从标准配置开始，逐步调节参数

## 故障排除

### 常见问题

1. **导入错误**: 确保在正确的conda环境中运行
2. **内存不足**: 减少批处理大小或模型宽度
3. **训练缓慢**: 减少模型深度或使用更小的数据集

### 性能优化

1. 使用混合精度训练（如果支持）
2. 调整学习率调度策略
3. 使用数据并行（多GPU训练）

## 示例配置

### 高质量配置
```bash
python train.py \
    --model SYMUNET \
    --symunet_width 128 \
    --symunet_enc_blk_nums 3,3,3 \
    --symunet_dec_blk_nums 3,3,3 \
    --symunet_restormer_heads 2,4,8 \
    --symunet_restormer_middle_heads 16 \
    --batch_size 8 \
    --lr 1e-4
```

### 快速训练配置
```bash
python train.py \
    --model SYMUNET \
    --symunet_width 32 \
    --symunet_enc_blk_nums 1,1,1 \
    --symunet_dec_blk_nums 1,1,1 \
    --symunet_restormer_heads 1,2,4 \
    --batch_size 32 \
    --lr 1e-3
```

## 总结

SymUNet已成功集成到TransENet项目中，具有以下优势：
- ✅ 单文件形式，便于维护
- ✅ 完整的参数配置系统
- ✅ 与现有训练框架无缝集成
- ✅ 全面的测试验证
- ✅ 详细的使用文档

现在您可以使用`--model SYMUNET`参数开始训练SymUNet模型！