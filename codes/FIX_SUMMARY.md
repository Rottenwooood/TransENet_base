# 🔧 修复总结：恢复训练时Loss绘图错误

## 问题描述

运行命令：
```bash
python demo_train.py --resume 1 --save symunet_pretrain_x4_rlfft2_w48
```

错误信息：
```
ValueError: x and y must have same first dimension, but have shapes (119,) and (1,)
```

## 根本原因分析

**不是loss代码本身的问题**，而是我对`trainer.py`的改动导致的scheduler调用混乱：

### 1. Scheduler调用时机错误
我在训练循环内部调用了`scheduler.step()`，导致scheduler被调用两次：
- 第84行：在训练循环内部调用
- 第116行：在epoch结束时调用

### 2. Cosine Scheduler处理错误
我对cosine scheduler的处理不正确，改变了其正常的step计数。

### 3. Global Step计数问题
添加的global_step计数影响了依赖step的逻辑。

## 修复措施

### 1. 回退Trainer.train()到原始实现
```python
# 修复前（有问题）
if hasattr(self.args, 'scheduler') and self.args.scheduler == 'cosine':
    self.scheduler.step()  # 在训练循环内部调用

# 修复后（正确）
self.scheduler.step()  # 只在epoch结束时调用
```

### 2. 安全添加WandB日志
- 添加异常处理，避免WandB初始化失败影响训练
- 在不影响核心训练逻辑的地方记录日志
- 使用属性检查确保wandb_logger存在

### 3. 保护性Loss绘图
为`plot_loss`函数添加数据有效性检查，避免绘图失败影响训练。

## 修复后的关键代码

### Trainer.train()核心逻辑
```python
def train(self):
    self.loss.step()
    epoch = self.scheduler.last_epoch + 1
    learn_rate = self.scheduler.get_last_lr()[0]

    self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(learn_rate)))
    self.loss.start_log()
    self.model.train()

    timer_data, timer_model = utils.timer(), utils.timer()

    for batch, (lr, hr, file_names) in enumerate(self.loader_train):
        # ... 训练逻辑 ...

        # 安全记录WandB日志
        if hasattr(self, 'wandb_logger') and (batch + 1) % self.args.print_every == 0:
            try:
                self.wandb_logger.log_training_step(...)
            except Exception as e:
                print(f"⚠️ WandB logging failed: {e}")

    self.scheduler.step()  # 只在epoch结束时调用
    self.loss.end_log(len(self.loader_train))
    self.error_last = self.loss.log[-1, -1]
```

### 安全的WandB初始化
```python
def __init__(self, args, loader, my_model, my_loss, ckp):
    # ... 其他初始化 ...

    # 安全初始化WandB
    self.wandb_logger = None
    if hasattr(args, 'use_wandb') and args.use_wandb:
        try:
            self.wandb_logger = wandb_utils.WandbLogger(args, my_model, my_loss)
        except Exception as e:
            print(f"⚠️ Failed to initialize WandB logger: {e}")
            print("Continuing without WandB...")
```

## 关键教训

1. **不要随意修改训练核心逻辑**：scheduler的调用时机必须严格遵循PyTorch的规范
2. **保护性编程**：添加异常处理，确保辅助功能失败不会影响主要功能
3. **渐进式添加功能**：先确保核心功能稳定，再逐步添加增强功能

## 验证

修复后应该能够：
- ✅ 正常恢复训练
- ✅ 正确绘制loss曲线
- ✅ 记录WandB日志（如启用）
- ✅ 不影响训练性能和稳定性

现在可以安全地运行恢复训练命令了！
