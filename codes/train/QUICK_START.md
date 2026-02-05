# 🚀 Quick Start Guide

## 安装依赖

```bash
# 安装WandB（如果使用 WandB 监控）
pip install wandb

# 安装其他依赖
pip install -r ../requirements.txt
```

## 快速训练（推荐设置）

### 1. 基本增强训练
```bash
python train_enhanced.py \
    --model SYMUNET_PRETRAIN \
    --dataset=UCMerced \
    --scale 4 \
    --epochs 300 \
    --batch_size 4 \
    --optimizer ADAMW \
    --scheduler cosine \
    --lr 2e-4 \
    --cosine_t_max 300 \
    --cosine_eta_min 5e-5 \
    --loss "1*L1+0.005*FFT" \
    --symunet_pretrain_width 48 \
    --symunet_pretrain_enc_blk_nums 4,6,6 \
    --symunet_pretrain_dec_blk_nums 6,6,4 \
    --symunet_pretrain_restormer_heads 1,2,4 \
    --symunet_pretrain_restormer_middle_heads 8 \
    --save_every_n_steps 50 \
    --save symunet_enhanced_x4
```

### 2. 使用WandB监控
```bash
# 首先登录WandB
wandb login

# 然后运行训练
python train_enhanced.py \
    --model SYMUNET_PRETRAIN \
    --dataset=UCMerced \
    --scale 4 \
    --use_wandb \
    --wandb_project "SymUNet-SR" \
    --wandb_name "ucmerced_x4" \
    --epochs 300 \
    --batch_size 4 \
    --optimizer ADAMW \
    --scheduler cosine \
    --lr 2e-4 \
    --loss "1*L1+0.005*FFT" \
    --save_every_n_steps 50 \
    --save symunet_wandb_x4
```

### 3. 恢复训练
```bash
python train_enhanced.py \
    --resume 1 \
    --save symunet_enhanced_x4 \
    --scheduler cosine \
    --optimizer ADAMW
```

## 🆕 新增文件

1. **wandb_utils.py** - WandB监控工具
2. **train_enhanced.py** - 增强训练脚本
3. **train_symunet_enhanced.py** - 完整训练示例
4. **ENHANCED_TRAINING.md** - 详细使用文档

## 📝 修改的文件

1. **option.py** - 添加了WandB、优化器和调度器参数
2. **utils.py** - 添加了AdamW优化器和余弦退火调度器
3. **trainer.py** - 集成了WandB和新的检查点保存功能
4. **requirements.txt** - 添加了wandb依赖

## 🎯 关键特性

- ✅ **AdamW优化器** - 更好的泛化性能
- ✅ **余弦退火调度** - 更平滑的学习率衰减
- ✅ **WandB监控** - 实时实验跟踪
- ✅ **步数检查点** - 每N步保存检查点
- ✅ **兼容性** - 与原有代码完全兼容

## ⚡ 性能提升

相比原始设置：
- **更快收敛** - AdamW + 余弦退火
- **更稳定训练** - 改进的优化器
- **更好监控** - WandB完整跟踪
- **更灵活控制** - 步数检查点保存

## 🚨 注意事项

1. **WandB登录**: 使用前需要 `wandb login`
2. **内存优化**: 建议batch_size=4，width=48
3. **步数设置**: cosine_t_max应<=epochs
4. **检查点管理**: step检查点会自动保存

现在你可以使用这些增强功能来训练你的SymUNet模型了！🎉