# 批量Deploy和PSNR计算工具包

这个工具包提供了批量deploy模型并计算PSNR的完整解决方案，支持多模型、多checkpoint的批量测试。

## 工具包文件

### 1. 核心脚本

#### `batch_deploy_psnr.py`
主要的批量deploy和PSNR计算脚本。

**功能:**
- 批量加载多个预训练模型
- 批量测试多个checkpoint
- 自动计算PSNR和SSIM
- 生成结果表格
- 保存所有生成图像

**特点:**
- 支持分块测试(节省GPU内存)
- 支持CPU模式
- 支持自定义图像格式
- 自动保存结果到文件

### 2. 辅助脚本

#### `quick_batch_deploy.py`
快速批量deploy脚本，自动检测checkpoint文件。

**特点:**
- 自动查找checkpoint文件
- 支持指定step列表或自动选择最新的N个
- 交互式命令行界面
- 自动生成完整的执行命令

#### `example_batch_deploy.py`
示例脚本，展示各种使用场景。

**包含:**
- 单模型多checkpoint示例
- 多模型多checkpoint示例
- 分块测试示例
- CPU模式示例
- 输出格式说明

### 3. 文档

#### `BATCH_DEPLOY_PSNR_GUIDE.md`
详细的使用指南。

**内容:**
- 完整参数说明
- 使用示例
- 目录结构说明
- 常见问题解答

#### `BATCH_DEPLOY_README.md`
本文档，工具包概览。

## 快速开始

### 方法1: 使用quick_batch_deploy.py (推荐)

```bash
python quick_batch_deploy.py \
    --pre_train /experiment/symunet_pretrain \
    --dir_data /data/UCMerced/test/LR_x4 \
    --dir_gt /data/UCMerced/test/HR_x4 \
    --dir_out /experiment/psnr_results
```

### 方法2: 直接使用batch_deploy_psnr.py

```bash
python batch_deploy_psnr.py \
    --model SYMUNET_PRETRAIN \
    --dataset UCMerced \
    --scale 4 \
    --dir_data /data/UCMerced/test/LR_x4 \
    --dir_out_base /experiment/psnr_results \
    --dir_gt /data/UCMerced/test/HR_x4 \
    --pre_train_list /experiment/symunet_pretrain/model/model_best.pt \
    --checkpoint_list "checkpoint_step_17850.pt,checkpoint_step_23800.pt,checkpoint_step_29750.pt"
```

## 输出结果

### 控制台输出

```
============================================================
最终结果表格 (PSNR)
============================================================
Pre-train Model                    17850         23800         29750
--------------------------------------------------------------------------------
model1                            26.1234       26.3456       26.5678
model2                            25.9876       26.1234       26.2345
================================================================================
```

### 文件输出

```
/experiment/psnr_results/
├── psnr_results.txt                    # 结果表格
├── pre_train_0/                        # 第一个预训练模型
│   ├── checkpoint_step_17850/
│   │   ├── image01.tif
│   │   ├── image02.tif
│   │   └── ...
│   ├── checkpoint_step_23800/
│   └── checkpoint_step_29750/
└── pre_train_1/                        # 第二个预训练模型
    └── ...
```

## 适用场景

### 1. 模型比较
比较不同预训练模型的性能：
```bash
python batch_deploy_psnr.py \
    --pre_train_list "/path/to/model1.pt,/path/to/model2.pt,/path/to/model3.pt" \
    --checkpoint_list "checkpoint_step_17850.pt"
```

### 2. 训练过程分析
分析模型在训练过程中的性能变化：
```bash
python batch_deploy_psnr.py \
    --pre_train_list /path/to/model.pt \
    --checkpoint_list "checkpoint_step_1000.pt,checkpoint_step_5000.pt,checkpoint_step_10000.pt"
```

### 3. 大规模批量测试
测试多个模型在多个checkpoint上的性能：
```bash
python batch_deploy_psnr.py \
    --pre_train_list "/path/to/model1.pt,/path/to/model2.pt,/path/to/model3.pt" \
    --checkpoint_list "cp1.pt,cp2.pt,cp3.pt,cp4.pt,cp5.pt"
```

## 性能优化

### GPU内存不足
使用分块测试：
```bash
python batch_deploy_psnr.py \
    --test_block \
    --patch_size 192
```

### 速度优化
1. 使用GPU而非CPU
2. 使用较大的patch_size (如果内存允许)
3. 关闭分块测试 (如果内存充足)

### 批量测试
```bash
# 创建批处理脚本
for model in model1.pt model2.pt model3.pt; do
    python batch_deploy_psnr.py \
        --pre_train_list /path/to/$model \
        --checkpoint_list "cp1.pt,cp2.pt,cp3.pt" \
        --dir_out /results/$model
done
```

## 注意事项

1. **图像格式**: 默认使用 `.tif` 格式，如需修改请编辑脚本中的 `img_ext` 变量
2. **命名规则**: 生成图像与GT图像的命名必须一致
3. **checkpoint格式**: checkpoint文件应包含 `model_state_dict` 或直接是模型权重
4. **GPU内存**: 大图像或大batch size可能导致OOM，可使用分块测试
5. **文件路径**: 确保所有路径正确且有读写权限

## 故障排除

### 问题1: 未找到checkpoint文件
**原因**: checkpoint路径或命名不正确
**解决**: 使用 `quick_batch_deploy.py` 自动检测，或检查experiment目录

### 问题2: CUDA内存不足
**原因**: 图像太大或batch size太大
**解决**: 使用 `--test_block` 参数

### 问题3: PSNR计算结果为0
**原因**: GT图像路径不正确或命名不匹配
**解决**: 检查 `--dir_gt` 路径和图像命名

### 问题4: 生成的图像全是黑色
**原因**: 模型加载失败或输入数据问题
**解决**: 检查checkpoint文件是否损坏，验证输入图像

## 扩展功能

### 添加更多指标
在 `calculate_psnr` 函数后添加其他指标计算函数：
```python
def calculate_other_metric(img1, img2):
    # 实现其他指标计算
    pass
```

### 支持其他图像格式
修改 `img_ext` 变量：
```python
img_ext = '.png'  # 支持PNG格式
```

### 自定义后处理
在deploy函数中添加自定义后处理步骤：
```python
# 在保存前添加后处理
final_sr = custom_post_process(final_sr)
```

## 联系和支持

如有问题或建议，请查看：
1. `BATCH_DEPLOY_PSNR_GUIDE.md` - 详细使用指南
2. `example_batch_deploy.py` - 更多示例

---

**版本**: 1.0.0
**更新日期**: 2026-02-06
