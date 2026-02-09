#!/usr/bin/env python3
"""
批量Deploy和PSNR计算示例脚本

这个脚本展示了如何使用batch_deploy_psnr.py进行批量测试
"""

import os
import sys


def example_1():
    """示例1: 测试单个预训练模型的3个checkpoint"""
    print("="*60)
    print("示例1: 单个预训练模型，3个checkpoint")
    print("="*60)

    cmd = """
python batch_deploy_psnr.py \
    --model SYMUNET_PRETRAIN \
    --dataset UCMerced \
    --scale 4 \
    --dir_data /data/UCMerced/test/LR_x4 \
    --dir_out_base /experiment/psnr_results \
    --dir_gt /data/UCMerced/test/HR_x4 \
    --pre_train_list /experiment/symunet_pretrain/model/model_best.pt \
    --checkpoint_list "checkpoint_step_17850.pt,checkpoint_step_23800.pt,checkpoint_step_29750.pt"
"""
    print(cmd)
    print("\n说明:")
    print("- 1个预训练模型")
    print("- 3个checkpoint")
    print("- 总共测试3次")
    print()


def example_2():
    """示例2: 测试3个预训练模型的3个checkpoint"""
    print("="*60)
    print("示例2: 3个预训练模型，3个checkpoint")
    print("="*60)

    cmd = """
python batch_deploy_psnr.py \
    --model SYMUNET_PRETRAIN \
    --dataset UCMerced \
    --scale 4 \
    --dir_data /data/UCMerced/test/LR_x4 \
    --dir_out_base /experiment/psnr_results \
    --dir_gt /data/UCMerced/test/HR_x4 \
    --pre_train_list "/experiment/model1.pt,/experiment/model2.pt,/experiment/model3.pt" \
    --checkpoint_list "checkpoint_step_17850.pt,checkpoint_step_23800.pt,checkpoint_step_29750.pt"
"""
    print(cmd)
    print("\n说明:")
    print("- 3个预训练模型")
    print("- 3个checkpoint")
    print("- 总共测试9次")
    print("- 输出表格: 3行(每个模型一行) x 3列(每个checkpoint一列)")
    print()


def example_3():
    """示例3: 使用分块测试"""
    print("="*60)
    print("示例3: 使用分块测试(节省GPU内存)")
    print("="*60)

    cmd = """
python batch_deploy_psnr.py \
    --model SYMUNET_PRETRAIN \
    --dataset UCMerced \
    --scale 4 \
    --dir_data /data/UCMerced/test/LR_x4 \
    --dir_out_base /experiment/psnr_results \
    --dir_gt /data/UCMerced/test/HR_x4 \
    --pre_train_list /experiment/symunet_pretrain/model/model_best.pt \
    --checkpoint_list "checkpoint_step_17850.pt,checkpoint_step_23800.pt,checkpoint_step_29750.pt" \
    --test_block \
    --patch_size 192
"""
    print(cmd)
    print("\n说明:")
    print("- 使用分块测试可以节省GPU内存")
    print("- 适用于大图像或小GPU内存的情况")
    print()


def example_4():
    """示例4: 使用CPU模式"""
    print("="*60)
    print("示例4: 使用CPU模式(没有GPU时)")
    print("="*60)

    cmd = """
python batch_deploy_psnr.py \
    --model SYMUNET_PRETRAIN \
    --dataset UCMerced \
    --scale 4 \
    --dir_data /data/UCMerced/test/LR_x4 \
    --dir_out_base /experiment/psnr_results \
    --dir_gt /data/UCMerced/test/HR_x4 \
    --pre_train_list /experiment/symunet_pretrain/model/model_best.pt \
    --checkpoint_list "checkpoint_step_17850.pt,checkpoint_step_23800.pt,checkpoint_step_29750.pt" \
    --cpu
"""
    print(cmd)
    print("\n说明:")
    print("- 使用CPU模式运行")
    print("- 速度较慢，但不需要GPU")
    print()


def show_directory_structure():
    """显示输出目录结构"""
    print("="*60)
    print("输出目录结构示例")
    print("="*60)

    structure = """
/experiment/psnr_results/
├── psnr_results.txt                    # 结果表格
├── pre_train_0/                        # 第一个预训练模型
│   ├── checkpoint_step_17850/          # checkpoint 17850
│   │   ├── image01.tif                 # 生成的SR图像
│   │   ├── image02.tif
│   │   └── ...
│   ├── checkpoint_step_23800/          # checkpoint 23800
│   │   ├── image01.tif
│   │   ├── image02.tif
│   │   └── ...
│   └── checkpoint_step_29750/          # checkpoint 29750
│       ├── image01.tif
│       ├── image02.tif
│       └── ...
└── pre_train_1/                        # 第二个预训练模型
    ├── checkpoint_step_17850/
    ├── checkpoint_step_23800/
    └── checkpoint_step_29750/
"""
    print(structure)


def show_result_format():
    """显示结果格式"""
    print("="*60)
    print("结果表格格式")
    print("="*60)

    result_table = """
Pre-train Model                    17850         23800         29750
--------------------------------------------------------------------------------
model1                            26.1234       26.3456       26.5678
model2                            25.9876       26.1234       26.2345
model3                            26.0123       26.2567       26.3789
================================================================================

说明:
- 行: 不同的预训练模型
- 列: 不同的checkpoint
- 数值: PSNR值 (dB)
"""
    print(result_table)


def main():
    print("\n" + "="*60)
    print("批量Deploy和PSNR计算 - 示例脚本")
    print("="*60 + "\n")

    example_1()
    example_2()
    example_3()
    example_4()

    show_directory_structure()
    print()
    show_result_format()

    print("\n" + "="*60)
    print("使用步骤")
    print("="*60)
    print("""
1. 确保有预训练的模型文件(.pt格式)
2. 确保有checkpoint文件(如checkpoint_step_17850.pt等)
3. 确保有测试数据的LR和HR图像目录
4. 根据需求选择合适的示例命令
5. 运行命令开始批量测试
6. 查看输出的结果表格和psnr_results.txt文件

更多信息请参考: BATCH_DEPLOY_PSNR_GUIDE.md
""")


if __name__ == '__main__':
    main()
