# 批量Deploy和PSNR计算脚本使用指南

## 概述

`batch_deploy_psnr.py` 是一个批量deploy模型并计算PSNR的工具。它可以：
- 测试多个预训练模型
- 测试多个checkpoint
- 自动计算每个组合的PSNR
- 生成结果表格

## 使用方法

### 基本命令

```bash
python batch_deploy_psnr.py \
    --model MODEL_NAME \
    --dataset DATASET_NAME \
    --scale 4 \
    --dir_data /path/to/LR/images \
    --dir_out_base /path/to/output \
    --dir_gt /path/to/GT/images \
    --pre_train_list "path/to/pre_train1.pt,path/to/pre_train2.pt" \
    --checkpoint_list "checkpoint_step_17850.pt,checkpoint_step_23800.pt,checkpoint_step_29750.pt"
```

### 必需参数

- `--dir_data`: LR图像目录路径
- `--dir_out_base`: 输出目录基础路径
- `--dir_gt`: Ground Truth图像目录路径
- `--pre_train_list`: 预训练模型路径列表 (逗号分隔)
- `--checkpoint_list`: checkpoint文件名列表 (逗号分隔)

### 可选参数

- `--model`: 模型名称 (默认: SYMUNET_PRETRAIN)
- `--dataset`: 数据集名称 (默认: UCMerced)
- `--scale`: 超分辨率倍数 (默认: [4])
- `--patch_size`: 训练patch大小 (默认: 192)
- `--test_block`: 使用分块测试
- `--cubic_input`: 使用cubic输入
- `--back_projection_iters`: 后投影迭代次数 (默认: 0)
- `--rgb_range`: RGB值范围 (默认: 255)
- `--cpu`: 使用CPU模式

## 使用示例

### 示例1: 单个预训练模型，3个checkpoint

```bash
python batch_deploy_psnr.py \
    --model SYMUNET_PRETRAIN \
    --dataset UCMerced \
    --scale 4 \
    --dir_data /data/UCMerced/test/LR_x4 \
    --dir_out_base /experiment/psnr_results \
    --dir_gt /data/UCMerced/test/HR_x4 \
    --pre_train_list /experiment/TransENETx4/model/model_best.pt \
    --checkpoint_list "checkpoint_step_17850.pt,checkpoint_step_23800.pt,checkpoint_step_29750.pt"
```

### 示例2: 多个预训练模型，3个checkpoint

```bash
python batch_deploy_psnr.py \
    --model SYMUNET_PRETRAIN \
    --dataset UCMerced \
    --scale 4 \
    --dir_data /data/UCMerced/test/LR_x4 \
    --dir_out_base /experiment/psnr_results \
    --dir_gt /data/UCMerced/test/HR_x4 \
    --pre_train_list "/experiment/model1.pt,/experiment/model2.pt,/experiment/model3.pt" \
    --checkpoint_list "checkpoint_step_17850.pt,checkpoint_step_23800.pt,checkpoint_step_29750.pt"
```

### 示例3: 使用分块测试

```bash
python batch_deploy_psnr.py \
    --model SYMUNET_PRETRAIN \
    --dataset UCMerced \
    --scale 4 \
    --dir_data /data/UCMerced/test/LR_x4 \
    --dir_out_base /experiment/psnr_results \
    --dir_gt /data/UCMerced/test/HR_x4 \
    --pre_train_list /experiment/model.pt \
    --checkpoint_list "checkpoint_step_17850.pt,checkpoint_step_23800.pt,checkpoint_step_29750.pt" \
    --test_block \
    --patch_size 192
```

## 输出格式

### 控制台输出

```
============================================================
批量Deploy和PSNR计算
============================================================
Pre-train模型数量: 2
  1. /experiment/model1.pt
  2. /experiment/model2.pt

Checkpoint数量: 3
  1. checkpoint_step_17850.pt
  2. checkpoint_step_23800.pt
  3. checkpoint_step_29750.pt

############################################################
处理Pre-train模型 1/2
############################################################

============================================================
处理: checkpoint_step_17850.pt
============================================================
加载checkpoint: /experiment/checkpoint_step_17850.pt
找到 21 张测试图像

部署完成，耗时: 15.23s

计算PSNR...

平均结果: PSNR: 26.1234 dB, SSIM: 0.8765

...

============================================================
最终结果表格 (PSNR)
============================================================
Pre-train Model                    17850         23800         29750
--------------------------------------------------------------------------------
model1                            26.1234       26.3456       26.5678
model2                            25.9876       26.1234       26.2345
================================================================================

结果已保存到: /experiment/psnr_results/psnr_results.txt
```

### 文件输出

结果也会保存到 `psnr_results.txt` 文件中，包含完整的测试配置和结果表格。

## 目录结构

```
/experiment/psnr_results/
├── psnr_results.txt                    # 结果文件
├── pre_train_0/                        # 第一个预训练模型
│   ├── checkpoint_step_17850/          # checkpoint 17850
│   │   ├── image1.tif                  # 生成的SR图像
│   │   ├── image2.tif
│   │   └── ...
│   ├── checkpoint_step_23800/          # checkpoint 23800
│   └── checkpoint_step_29750/          # checkpoint 29750
└── pre_train_1/                        # 第二个预训练模型
    ├── checkpoint_step_17850/
    ├── checkpoint_step_23800/
    └── checkpoint_step_29750/
```

## 注意事项

1. **图像格式**: 测试图像和GT图像应为 `.tif` 格式
2. **命名规则**: 生成图像与GT图像的命名应一致（扩展名除外）
3. **GPU内存**: 如果遇到CUDA内存不足，可以：
   - 使用 `--test_block` 进行分块测试
   - 减少 `--patch_size`
   - 使用 `--cpu` 模式
4. **checkpoint文件**: 需要包含 `model_state_dict` 或直接是模型权重
5. **Ground Truth目录**: 只包含HR图像，不包含子目录

## 常见问题

### Q: 如何使用自定义数据集？
A: 确保 `--dir_data` 和 `--dir_gt` 指向正确的目录，图像命名一致即可。

### Q: 如何修改图像格式？
A: 当前脚本固定使用 `.tif` 格式，如需修改，请编辑 `img_ext` 变量。

### Q: 如何添加SSIM到结果中？
A: 脚本已计算SSIM但未在表格中显示，可以修改代码将SSIM也输出到表格中。

### Q: checkpoint文件路径不对怎么办？
A: `--checkpoint_list` 需要的是相对于实验目录的文件名，完整路径由 `--pre_train_list` 的目录推断。
