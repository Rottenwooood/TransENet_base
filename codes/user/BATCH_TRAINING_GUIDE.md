# 🚀 批量训练管理系统 - 完整使用指南

这个批量训练管理系统帮助你高效地管理超参数实验，进行串行训练，并分析结果。

## 📁 文件结构

```
codes/
├── batch_train.py              # 通用批量训练脚本
├── quick_batch.py             # 快速批量训练脚本
├── analyze_experiments.py      # 实验结果分析工具
├── experiments_config.json     # 实验配置文件模板
├── wandb_utils.py             # WandB监控工具
├── train_enhanced.py          # 增强训练脚本
└── experiment/                # 实验结果目录
    ├── experiment_results.csv # 实验结果数据
    └── analysis_output/        # 分析图表输出
```

## 🎯 三种使用方式

### 方式一：快速批量训练（推荐新手）

**特点**: 预设几种常用实验配置，一键开始

```bash
# 1. 列出所有预设配置
python quick_batch.py --list

# 2. 运行学习率对比实验
python quick_batch.py --preset lr_comparison

# 3. 运行所有预设实验（谨慎使用！）
python quick_batch.py --all

# 4. 试运行（不实际执行）
python quick_batch.py --preset optimizer_comparison --dry-run
```

**预设配置**:
- `lr_comparison` - 学习率对比实验 (1e-4, 2e-4, 5e-4)
- `optimizer_comparison` - 优化器对比实验 (ADAMW vs ADAM)
- `width_comparison` - 模型宽度对比实验 (32, 48, 64)
- `loss_comparison` - 损失函数对比实验 (L1, L1+FFT)

### 方式二：自定义批量训练（推荐高级用户）

**特点**: 完全自定义超参数网格，适合深度调优

```bash
# 1. 创建自定义配置文件
python batch_train.py --create-config

# 2. 编辑 experiments_config.json 文件
vim experiments_config.json

# 3. 查看将运行的实验
python batch_train.py --config experiments_config.json --list

# 4. 试运行
python batch_train.py --config experiments_config.json --dry-run

# 5. 执行批量训练
python batch_train.py --config experiments_config.json
```

### 方式三：混合使用（推荐专业用户）

**特点**: 结合两种方式的优点

```bash
# 1. 先用快速模式探索几个关键参数
python quick_batch.py --preset lr_comparison --no-wandb

# 2. 分析结果，找出最佳学习率
python analyze_experiments.py --top 3

# 3. 基于结果，创建自定义实验
python batch_train.py --create-config

# 4. 精细调优其他参数
python batch_train.py --config my_experiments.json
```

## 📊 实验配置详解

### experiments_config.json 配置说明

```json
{
  "base_config": {
    "model": "SYMUNET_PRETRAIN",     // 模型架构
    "dataset": "UCMerced",           // 数据集
    "scale": 4,                      // 超分辨率倍数
    "epochs": 300,                   // 训练轮数
    "batch_size": 4,                 // 批次大小
    "ext": "img",                   // 数据格式
    "patch_size": 192               // 图像块大小
  },
  "hyperparameter_grid": {
    "optimizer": ["ADAMW", "ADAM"],        // 要对比的优化器
    "scheduler": ["cosine", "step"],      // 学习率调度器
    "lr": [1e-4, 2e-4, 5e-4],           // 学习率
    "loss": [                              // 损失函数
      "1*L1",
      "1*L1+0.005*FFT"
    ],
    "symunet_pretrain_width": [32, 48, 64],  // 模型宽度
    "symunet_pretrain_enc_blk_nums": [        // 编码器深度
      "2,2,2",
      "4,4,4"
    ]
  },
  "experiment_prefix": "my_exp",           // 实验前缀
  "max_experiments": 20,                   // 最大实验数量
  "use_wandb": true,                       // 是否使用WandB
  "wandb_project": "MyProject",            // WandB项目名
  "save_every_n_steps": 50,                // 每N步保存检查点
  "run_name_pattern": "{prefix}_lr{lr}_opt{optimizer}"  // 实验命名模式
}
```

## 🔧 超参数网格建议

### 对于遥感图像SR的推荐配置

```json
{
  "hyperparameter_grid": {
    "optimizer": ["ADAMW"],                    // 遥感图像推荐ADAMW
    "scheduler": ["cosine"],                  // 余弦退火更稳定
    "lr": [1e-4, 2e-4],                     // 学习率范围
    "loss": [
      "1*L1+0.005*FFT",                     // 空间+频率损失
      "1*L1+0.01*FFT"
    ],
    "symunet_pretrain_width": [32, 48, 64], // 内存限制下的宽度
    "symunet_pretrain_enc_blk_nums": [
      "2,2,2",                               // 基础配置
      "4,6,6"                                // 深度配置
    ],
    "symunet_pretrain_dec_blk_nums": [
      "2,2,2",
      "6,6,4"
    ]
  }
}
```

### 实验数量控制

| 超参数数量 | 总组合数 | 推荐实验数 | 说明 |
|-----------|----------|------------|------|
| 2-3个     | 8-27     | 8-20       | 快速探索 |
| 4-5个     | 32-243   | 20-50      | 深度调优 |
| 6+个      | 64+      | 50+        | 全面搜索 |

## 📈 结果分析

### 基本分析

```bash
# 显示实验摘要
python analyze_experiments.py

# 显示最佳实验
python analyze_experiments.py --top 10

# 过滤特定实验
python analyze_experiments.py --filter optimizer=ADAMW
```

### 高级分析

```bash
# 生成可视化图表
python analyze_experiments.py --visualize

# 导出HTML报告
python analyze_experiments.py --export my_report.html

# 指定结果文件
python analyze_experiments.py --results-file /path/to/results.csv
```

### 分析输出

```
analysis_output/
├── training_time_distribution.png     # 训练时间分布
├── hyperparameter_analysis.png       # 超参数影响分析
└── correlation_matrix.png           # 参数相关性热图
```

## 🎛️ 高级功能

### 1. 增量实验

```bash
# 先运行一部分实验
python batch_train.py --config my_config.json --ids 1,3,5,7

# 再运行剩余实验
python batch_train.py --config my_config.json --ids 2,4,6,8
```

### 2. 断点续传

```bash
# 训练中断后，恢复训练
python train_enhanced.py --resume 1 --save previous_experiment_name

# 继续批量训练（会跳过已完成的实验）
python batch_train.py --config my_config.json
```

### 3. 实验比较

```bash
# 比较两个实验的性能
python analyze_experiments.py --filter "config_optimizer=ADAMW" > adamw_results.txt
python analyze_experiments.py --filter "config_optimizer=ADAM" > adam_results.txt
```

## 📊 WandB集成

### 批量实验的WandB管理

```bash
# 1. 登录WandB
wandb login

# 2. 批量训练会自动创建实验
python quick_batch.py --preset lr_comparison

# 3. 在WandB dashboard中查看所有实验
# https://wandb.ai/your_project
```

### WandB最佳实践

1. **项目命名**: 使用有意义的项目名，如 `SymUNet-SR-Batch-2024`
2. **实验命名**: 使用清晰的实验名，包含关键超参数
3. **标签管理**: 在WandB界面中为实验添加标签
4. **图表对比**: 利用WandB的对比功能分析不同实验

## ⚠️ 注意事项

### 内存管理

```bash
# 内存不足时，减少批次大小和模型宽度
--batch_size 2
--symunet_pretrain_width 32
```

### 时间估算

| 实验数量 | 每个实验时间 | 总时间估算 |
|----------|-------------|------------|
| 3个      | 4小时       | 12小时     |
| 5个      | 4小时       | 20小时     |
| 10个     | 4小时       | 40小时     |
| 20个     | 4小时       | 80小时     |

### 磁盘空间

- 每个实验: ~2GB (模型权重 + 日志)
- 20个实验: ~40GB
- 建议定期清理: `rm -rf ../experiment/old_experiments`

## 🛠️ 故障排除

### 常见错误

1. **ImportError**: 缺少依赖
   ```bash
   pip install pandas seaborn matplotlib
   ```

2. **CUDA OOM**: 显存不足
   ```bash
   # 减少batch_size或model_width
   --batch_size 2
   --symunet_pretrain_width 32
   ```

3. **WandB登录失败**
   ```bash
   wandb login --relogin
   ```

4. **数据集路径错误**
   ```bash
   # 检查数据路径
   ls /your/data/path
   ```

### 调试模式

```bash
# 启用详细输出
export DEBUG=1
python batch_train.py --config my_config.json

# 检查配置文件
python -m json.tool experiments_config.json
```

## 📝 实际使用案例

### 案例1：学习率调优

```bash
# 1. 创建学习率调优实验
python quick_batch.py --preset lr_comparison

# 2. 分析结果
python analyze_experiments.py --top 3

# 3. 基于最佳学习率继续调优
python batch_train.py --create-config
# 编辑配置文件，使用最佳学习率，测试其他参数
```

### 案例2：全面超参数搜索

```bash
# 1. 第一阶段：粗搜索
python batch_train.py --config broad_search.json

# 2. 分析结果，找出最佳区域
python analyze_experiments.py --visualize

# 3. 第二阶段：精细搜索
python batch_train.py --config fine_search.json
```

### 案例3：持续改进

```bash
# 1. 运行基础实验
python quick_batch.py --preset optimizer_comparison

# 2. 分析并选择最佳配置
python analyze_experiments.py --export baseline_report.html

# 3. 基于最佳配置微调
python train_enhanced.py --model SYMUNET_PRETRAIN \
    --optimizer ADAMW --scheduler cosine --lr 2e-4 \
    --save final_optimized_model
```

## 🎯 最佳实践

1. **从简单开始**: 先用快速预设探索关键参数
2. **逐步深入**: 基于结果进行精细调优
3. **记录分析**: 每次实验后都进行结果分析
4. **版本控制**: 保存配置文件，方便重现实验
5. **资源管理**: 合理控制实验数量，避免资源浪费

## 📞 获取帮助

- 查看帮助: `python script_name.py --help`
- 创建配置: `python batch_train.py --create-config`
- 列出预设: `python quick_batch.py --list`
- 试运行: 添加 `--dry-run` 参数

---

**祝实验顺利！** 🚀