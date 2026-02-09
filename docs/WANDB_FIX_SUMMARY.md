# 🎯 WandB修复总结

## 问题识别

从用户提供的日志中发现了以下问题：

1. **WandB Step记录混乱**：
   ```
   [33mWARNING[0m Tried to log to step 1 that is less than the current step 110.
   Steps must be monotonically increasing, so this data will be ignored.
   ```

2. **模型保存路径错误**：
   ```
   ⚠️ Failed to save model artifact: Path is not a file: '../experiment/.../model/model_epoch_1.pt'
   ```

3. **训练逻辑被干扰**：每个batch都在记录WandB，影响训练效率

## 修复措施

### 1. 简化WandB记录逻辑
**修复前**：
- 每个batch都记录WandB指标
- 复杂的step计数和step-based记录
- 多个地方调用WandB日志

**修复后**：
- 只在test时记录epoch级别的loss和PSNR
- 移除step-based记录，改为epoch-based
- 保持与原有loss曲线绘制相同的时机

### 2. 移除有问题的功能
- ❌ 移除WandB model artifact自动保存（路径错误）
- ❌ 移除step-based WandB记录（step计数混乱）
- ❌ 移除训练循环中的WandB调用（影响性能）

### 3. 保留重要功能
- ✅ 保留`save_every_n_steps`功能（step-based checkpoint保存）
- ✅ 保留WandB初始化（安全检查）
- ✅ 保留epoch级别的validation记录

## 修复后的记录时机

### 训练记录（与原有逻辑一致）
```bash
🏃 Starting training...
[Epoch 1]	Learning rate: 2.00e-4
[80/945]	[L1: 0.0957]	0.0s
[160/945]	+0.0[L1: 0.0725]	0.0+0.0s
...
[880/945]	[L1: 0.0471]	0.0+0.0s
```

### Validation记录（WandB）
```bash
Evaluation:
[UCMerced x4]	psnr: 26.100 (Best: 26.100 @epoch 1)
Total time: 1.79s
```
**WandB记录**：epoch=1, psnr=26.100

### Checkpoint保存
- **Epoch-based**：每个epoch结束时保存（原有逻辑）
- **Step-based**：`save_every_n_steps`参数控制（新增功能）

## WandB记录的内容

### Validation指标
- `val/epoch`：训练轮数
- `val/psnr`：当前PSNR值
- `val/best_psnr`：最佳PSNR值

### 记录时机
- 只在`test()`方法结束时记录
- 只记录第一个scale（idx_scale == 0）
- 记录频率：每个epoch一次

## 修复代码对比

### 修复前（有问题）
```python
# 每个batch都记录WandB
for batch in loader_train:
    # 训练...
    wandb_logger.log_training_step(step=self.global_step, ...)  # 问题！

# 每个scale都记录WandB
for idx_scale, scale in enumerate(self.scale):
    # 记录PSNR
    wandb_logger.log_validation(...)  # 重复记录！

# 每次best model都保存artifact
if is_best:
    wandb_logger.save_model_artifact(model_path)  # 路径错误！
```

### 修复后（正确）
```python
# 训练循环中不记录WandB
for batch in loader_train:
    # 训练...
    # 没有WandB记录

# 只记录第一个scale的validation
if idx_scale == 0:  # 只记录一次
    wandb_logger.log_validation(...)

# 不保存WandB model artifact
# 移除保存逻辑，避免路径错误
```

## 预期效果

### 修复前的问题
- ❌ WandB step计数混乱
- ❌ 每个batch都记录，影响性能
- ❌ 模型保存路径错误
- ❌ 重复记录validation指标

### 修复后的效果
- ✅ 清晰的epoch级别记录
- ✅ 不影响训练性能
- ✅ 无路径错误
- ✅ 与原有loss曲线时机一致

## 核心原则

1. **最小化干扰**：WandB不应该影响核心训练逻辑
2. **时机一致**：WandB记录时机与loss曲线绘制一致
3. **错误容忍**：WandB初始化失败不应该中断训练
4. **简洁明了**：只记录关键指标，避免过度记录

---

**修复完成！现在WandB记录逻辑简洁、高效、可靠。** ✅
