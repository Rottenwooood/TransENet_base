# SymUNet 超分辨率适配版本使用指南

## 概述

为了更好地适配超分辨率任务，我们基于原始SymUNet创建了两个专门的版本：

1. **SymUNet-Pretrain (预上采样版本)**
2. **SymUNet-Posttrain (后上采样版本)**

这两个版本采用了不同的上采样策略，适用于不同的超分辨率场景。

---

## 方案1：SymUNet-Pretrain (预上采样版本)

### 核心思想
在网络最开始使用bicubic插值将LR图像直接放大到目标HR尺寸，然后在整个网络中处理HR尺度的图像。

### 文件
- `codes/model/symunet_pretrain.py`

### 网络结构特点
```
输入LR → bicubic插值 → HR尺寸 → SymUNet特征提取 → 输出HR
```

### 技术细节
1. **预上采样层**：使用`nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False)`
2. **特征提取**：在HR空间进行完整的编码器-解码器特征提取
3. **残差连接**：最终输出加上预上采样的HR图像
4. **网络结构**：与原始SymUNet基本相同

### 优势
- ✅ 简单直观，容易实现
- ✅ 网络可以直接学习HR空间的细节信息
- ✅ 可以复用预训练的分类模型（如有）
- ✅ 适合需要高质量细节恢复的场景

### 劣势
- ❌ 计算复杂度高，需要处理HR尺寸的图像
- ❌ 内存占用大，不适合大尺度超分辨率
- ❌ 对GPU内存要求较高

### 适用场景
- 小尺度超分辨率 (2x, 4x)
- 高质量细节恢复
- 资源充足的环境

---

## 方案2：SymUNet-Posttrain (后上采样版本)

### 核心思想
在LR空间进行特征提取，在decoder末端使用PixelShuffle进行上采样，使用bicubic插值的LR作为残差连接。

### 文件
- `codes/model/symunet_posttrain.py`

### 网络结构特点
```
输入LR → LR空间特征提取 → PixelShuffle上采样 → bicubic残差连接 → 输出HR
```

### 技术细节
1. **LR特征提取**：在原始LR尺寸进行编码器-解码器特征提取
2. **PixelShuffle上采样**：在decoder末端使用`nn.PixelShuffle(scale_factor)`进行学习上采样
3. **残差连接**：最终输出加上bicubic插值的LR图像
4. **通道调整**：最终卷积层输出`img_channel * (scale * scale)`通道，配合PixelShuffle

### 优势
- ✅ 计算效率高，在LR空间进行特征提取
- ✅ 内存占用小，适合大尺寸图像
- ✅ 端到端学习上采样过程
- ✅ 适合实时应用

### 劣势
- ❌ 上采样质量完全依赖网络学习
- ❌ 可能丢失高频细节
- ❌ 训练相对困难

### 适用场景
- 大尺度超分辨率 (8x, 16x)
- 资源受限环境
- 实时推理需求

---

## 参数配置

### 基本使用

#### SymUNet-Pretrain
```bash
python train.py --model SYMUNET_PRETRAIN
```

#### SymUNet-Posttrain
```bash
python train.py --model SYMUNET_POSTTRAIN
```

### 高级参数配置

#### SymUNet-Pretrain参数
```bash
# 网络宽度
--symunet_pretrain_width 64

# 中间块数量
--symunet_pretrain_middle_blk_num 1

# 编码器/解码器深度
--symunet_pretrain_enc_blk_nums 2,2,2
--symunet_pretrain_dec_blk_nums 2,2,2

# 注意力头数
--symunet_pretrain_restormer_heads 1,2,4
--symunet_pretrain_restormer_middle_heads 8

# Transformer参数
--symunet_pretrain_ffn_expansion_factor 2.66
--symunet_pretrain_bias False
--symunet_pretrain_layer_norm_type WithBias
```

#### SymUNet-Posttrain参数
```bash
# 网络宽度
--symunet_posttrain_width 64

# 中间块数量
--symunet_posttrain_middle_blk_num 1

# 编码器/解码器深度
--symunet_posttrain_enc_blk_nums 2,2,2
--symunet_posttrain_dec_blk_nums 2,2,2

# 注意力头数
--symunet_posttrain_restormer_heads 1,2,4
--symunet_posttrain_restormer_middle_heads 8

# Transformer参数
--symunet_posttrain_ffn_expansion_factor 2.66
--symunet_posttrain_bias False
--symunet_posttrain_layer_norm_type WithBias
```

---

## 完整使用示例

### 预上采样版本 - 高质量配置
```bash
python train.py \
    --model SYMUNET_PRETRAIN \
    --symunet_pretrain_width 128 \
    --symunet_pretrain_enc_blk_nums 3,3,3 \
    --symunet_pretrain_dec_blk_nums 3,3,3 \
    --symunet_pretrain_restormer_heads 2,4,8 \
    --symunet_pretrain_restormer_middle_heads 16 \
    --batch_size 8 \
    --lr 1e-4 \
    --scale 4
```

### 后上采样版本 - 高效配置
```bash
python train.py \
    --model SYMUNET_POSTTRAIN \
    --symunet_posttrain_width 64 \
    --symunet_posttrain_enc_blk_nums 2,2,2 \
    --symunet_posttrain_dec_blk_nums 2,2,2 \
    --symunet_posttrain_restormer_heads 1,2,4 \
    --symunet_posttrain_restormer_middle_heads 8 \
    --batch_size 16 \
    --lr 1e-4 \
    --scale 4
```

### 后上采样版本 - 大尺度配置
```bash
python train.py \
    --model SYMUNET_POSTTRAIN \
    --symunet_posttrain_width 32 \
    --symunet_posttrain_enc_blk_nums 1,1,1 \
    --symunet_posttrain_dec_blk_nums 1,1,1 \
    --symunet_posttrain_restormer_heads 1,2,4 \
    --batch_size 32 \
    --lr 1e-3 \
    --scale 8
```

---

## 模型对比

| 特性 | SymUNet-Pretrain | SymUNet-Posttrain |
|------|------------------|-------------------|
| **计算复杂度** | 高 | 低 |
| **内存占用** | 大 | 小 |
| **推理速度** | 慢 | 快 |
| **上采样质量** | 依赖bicubic | 依赖学习 |
| **细节恢复** | 优秀 | 一般 |
| **可扩展性** | 差 | 好 |
| **适用尺度** | 2x, 4x | 4x, 8x, 16x+ |
| **硬件要求** | 高 | 中等 |

---

## 性能对比预期

### 计算复杂度
- **Pretrain**: O(H²×W²) - 在HR空间处理
- **Posttrain**: O(H×W) - 在LR空间处理

### 内存占用
- **Pretrain**: 约4-8倍Posttrain版本
- **Posttrain**: 适合大尺寸图像处理

### 推理质量
- **Pretrain**: 高质量细节恢复，PSNR/SSIM通常更好
- **Posttrain**: 细节可能模糊，但边缘更平滑

---

## 测试验证

### 快速测试
```python
import torch
from codes.model.symunet_pretrain import SymUNet_Pretrain
from codes.model.symunet_posttrain import SymUNet_Posttrain
from codes.option import args

# 测试预上采样版本
model_pretrain = SymUNet_Pretrain(args)
input_lr = torch.rand(1, 3, 48, 48)
output_pretrain = model_pretrain(input_lr)
print(f"Pretrain - 输入: {input_lr.size()} → 输出: {output_pretrain.size()}")

# 测试后上采样版本
model_posttrain = SymUNet_Posttrain(args)
output_posttrain = model_posttrain(input_lr)
print(f"Posttrain - 输入: {input_lr.size()} → 输出: {output_posttrain.size()}")
```

### 训练脚本示例
```bash
# 预上采样版本训练
python train.py \
    --model SYMUNET_PRETRAIN \
    --data_train /path/to/train/dataset \
    --data_val /path/to/val/dataset \
    --epochs 500 \
    --batch_size 8 \
    --lr 1e-4

# 后上采样版本训练
python train.py \
    --model SYMUNET_POSTTRAIN \
    --data_train /path/to/train/dataset \
    --data_val /path/to/val/dataset \
    --epochs 500 \
    --batch_size 16 \
    --lr 1e-4
```

---

## 选择建议

### 选择Pretrain的情况
1. **追求最高质量**：需要最佳的PSNR/SSIM指标
2. **小尺度任务**：2x或4x超分辨率
3. **充足资源**：有足够的GPU内存和计算资源
4. **细节重要**：如人脸、纹理等细节丰富的场景

### 选择Posttrain的情况
1. **效率优先**：需要快速推理或实时处理
2. **大尺度任务**：8x或更高倍数的超分辨率
3. **资源受限**：GPU内存或计算资源有限
4. **批量处理**：需要处理大量高分辨率图像

---

## 注意事项

### 1. 数据准备
- 确保LR-HR图像对数据正确
- 注意图像尺寸必须是网络padder_size的倍数
- 对于Pretrain版本，需要考虑HR图像的内存占用

### 2. 训练策略
- **Pretrain**: 可以使用预训练的图像增强模型
- **Posttrain**: 端到端训练，建议使用学习率调度
- 都可以使用标准的L1/L2损失或感知损失

### 3. 推理优化
- **Pretrain**: 可以使用chop策略处理大图像
- **Posttrain**: 天然支持大尺寸图像推理
- 都可以使用self-ensemble测试时增强

### 4. 超参数调优
- **Pretrain**: 可以使用更大的模型宽度和深度
- **Posttrain**: 建议适度增加中间层注意力头数
- 两个版本都可以调整FFN扩展因子

---

## 总结

| 版本 | 推荐使用场景 | 优势 | 劣势 |
|------|--------------|------|------|
| **SymUNet-Pretrain** | 高质量小尺度SR | 质量最佳，细节丰富 | 资源消耗大 |
| **SymUNet-Posttrain** | 效率优先大尺度SR | 效率高，可扩展性好 | 质量略低 |

两个版本都完整集成了TransENet的训练框架，可以直接使用现有的训练脚本进行训练和测试。选择哪个版本主要取决于您的具体需求和资源约束。