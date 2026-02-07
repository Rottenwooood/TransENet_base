#!/usr/bin/env python3
"""
批量deploy模型并计算PSNR的脚本

流程：
1. 对每个pre_train模型和checkpoint组合进行deploy
2. 计算每个组合的PSNR
3. 输出格式：三个checkpoint的结果放一行，不同pre_train放不同行
"""

import os
import sys
import glob
import numpy as np
import cv2
import math
import time
import torch
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils
import model
import data.common as common


def calculate_psnr(img1, img2):
    """计算PSNR"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_rgb_psnr(img1, img2):
    """计算RGB通道PSNR"""
    n_channels = np.ndim(img1)
    sum_psnr = 0
    for i in range(n_channels):
        this_psnr = calculate_psnr(img1[:,:,i], img2[:,:,i])
        sum_psnr += this_psnr
    return sum_psnr/n_channels


def calculate_ssim(img1, img2):
    """计算SSIM"""
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def deploy_and_evaluate(args, checkpoint_dir, checkpoint_name, output_dir, gt_dir):
    """
    对单个checkpoint进行deploy并计算PSNR

    Args:
        args: 命令行参数
        checkpoint_dir: checkpoint目录
        checkpoint_name: checkpoint文件名
        output_dir: 输出目录
        gt_dir: Ground truth目录

    Returns:
        avg_psnr: 平均PSNR值
    """
    print(f"\n{'='*60}")
    print(f"处理: {checkpoint_name}")
    print(f"{'='*60}")

    # 设置device
    device = torch.device('cpu' if args.cpu else 'cuda')
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    args.pre_train = checkpoint_path
    # 加载模型
    checkpoint = utils.checkpoint(args)
    sr_model = model.Model(args, checkpoint)
    sr_model.eval()

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取测试图像列表
    img_ext = '.tif'
    img_lists = glob.glob(os.path.join(args.dir_data, '*'+img_ext))

    if len(img_lists) == 0:
        print("错误: 测试目录中没有图像!")
        return 0.0

    print(f"找到 {len(img_lists)} 张测试图像")

    # 部署模型处理图像
    start_time = time.time()
    with torch.no_grad():
        for i, img_path in enumerate(img_lists):
            print(f"[{i+1}/{len(img_lists)}] 处理: {os.path.basename(img_path)}")

            # 读取LR图像
            lr_np = cv2.imread(img_path, cv2.IMREAD_COLOR)
            lr_np = cv2.cvtColor(lr_np, cv2.COLOR_BGR2RGB)

            # 如果需要cubic input，先上采样
            if args.cubic_input:
                lr_np = cv2.resize(lr_np, (lr_np.shape[0] * args.scale[0], lr_np.shape[1] * args.scale[0]),
                                interpolation=cv2.INTER_CUBIC)

            # 转换为tensor
            lr = common.np2Tensor([lr_np], args.rgb_range)[0].unsqueeze(0)

            # 分块测试
            if args.test_block:
                b, c, h, w = lr.shape
                factor = args.scale[0]
                tp = args.patch_size
                if not args.cubic_input:
                    ip = tp // factor
                else:
                    ip = tp

                assert h >= ip and w >= ip, 'LR input must be larger than the training inputs'
                if not args.cubic_input:
                    sr = torch.zeros((b, c, h * factor, w * factor))
                else:
                    sr = torch.zeros((b, c, h, w))

                for iy in range(0, h, ip):
                    if iy + ip > h:
                        iy = h - ip
                    ty = factor * iy

                    for ix in range(0, w, ip):
                        if ix + ip > w:
                            ix = w - ip
                        tx = factor * ix

                        lr_p = lr[:, :, iy:iy + ip, ix:ix + ip]
                        lr_p = lr_p.to(device)
                        sr_p = sr_model(lr_p)
                        sr[:, :, ty:ty + tp, tx:tx + tp] = sr_p
            else:
                lr = lr.to(device)
                sr = sr_model(lr)

            # 转换为numpy
            sr_np = np.array(sr.cpu().detach())
            sr_np = sr_np[0, :].transpose([1, 2, 0])
            lr_np = lr_np * args.rgb_range / 255.

            # Again back projection for the final fused result
            for bp_iter in range(args.back_projection_iters):
                sr_np = utils.back_projection(sr_np, lr_np, down_kernel='cubic',
                                           up_kernel='cubic', sf=args.scale[0], range=args.rgb_range)

            # 量化
            if args.rgb_range == 1:
                final_sr = np.clip(sr_np * 255, 0, args.rgb_range * 255)
            else:
                final_sr = np.clip(sr_np, 0, args.rgb_range)

            final_sr = final_sr.astype(np.uint8)
            final_sr = cv2.cvtColor(final_sr, cv2.COLOR_RGB2BGR)

            # 保存结果
            output_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(output_path, final_sr)

    deploy_time = time.time() - start_time
    print(f"部署完成，耗时: {deploy_time:.2f}s")

    # 计算PSNR
    print(f"\n计算PSNR...")
    PSNR_all = []
    SSIM_all = []

    # 边界裁剪
    crop_border = args.scale[0]

    # 获取GT图像列表
    gt_lists = sorted(glob.glob(os.path.join(gt_dir, '*')))

    for i, gt_path in enumerate(gt_lists):
        base_name = os.path.splitext(os.path.basename(gt_path))[0]
        gen_path = os.path.join(output_dir, base_name + img_ext)

        if not os.path.exists(gen_path):
            print(f"警告: 生成图像不存在: {gen_path}")
            continue

        # 读取图像
        im_GT = cv2.imread(gt_path) / 255.
        im_Gen = cv2.imread(gen_path) / 255.

        # 裁剪边界
        if crop_border == 0:
            cropped_GT = im_GT
            cropped_Gen = im_Gen
        else:
            if im_GT.ndim == 3:
                cropped_GT = im_GT[crop_border:-crop_border, crop_border:-crop_border, :]
                cropped_Gen = im_Gen[crop_border:-crop_border, crop_border:-crop_border, :]
            elif im_GT.ndim == 2:
                cropped_GT = im_GT[crop_border:-crop_border, crop_border:-crop_border]
                cropped_Gen = im_Gen[crop_border:-crop_border, crop_border:-crop_border]

        # 计算PSNR和SSIM
        PSNR = calculate_rgb_psnr(cropped_GT * 255, cropped_Gen * 255)
        SSIM = calculate_ssim(cropped_GT * 255, cropped_Gen * 255)

        PSNR_all.append(PSNR)
        SSIM_all.append(SSIM)

        print(f"  {base_name:30s} - PSNR: {PSNR:.4f} dB, SSIM: {SSIM:.4f}")

    avg_psnr = sum(PSNR_all) / len(PSNR_all)
    avg_ssim = sum(SSIM_all) / len(SSIM_all)

    print(f"\n平均结果: PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.4f}")

    return avg_psnr


def main():
    # 解析参数（从option.py继承）
    from option import args

    print("="*60)
    print("批量Deploy和PSNR计算")
    print("="*60)

    # 解析pre_train列表
    pre_train_list = [p.strip() for p in args.pre_train_list.split(',')]
    print(f"Pre-train模型数量: {len(pre_train_list)}")
    for i, p in enumerate(pre_train_list):
        print(f"  {i+1}. {p}")

    # 解析checkpoint列表
    checkpoint_list = [c.strip() for c in args.checkpoint_list.split(',')]
    print(f"Checkpoint数量: {len(checkpoint_list)}")
    for i, c in enumerate(checkpoint_list):
        print(f"  {i+1}. {c}")

    # 创建结果存储
    results = []

    # 对每个pre_train进行测试
    for pre_train_idx, pre_train_path in enumerate(pre_train_list):
        print(f"\n{'#'*60}")
        print(f"处理Pre-train模型 {pre_train_idx+1}/{len(pre_train_list)}")
        print(f"路径: {pre_train_path}")
        print(f"{'#'*60}")

        # 获取experiment目录
        experiment_dir = os.path.dirname(pre_train_path)
        model_name = os.path.basename(pre_train_path).replace('.pt', '')

        pre_train_results = []

        # 对每个checkpoint进行测试
        for checkpoint_idx, checkpoint_name in enumerate(checkpoint_list):
            print(f"\n处理Checkpoint {checkpoint_idx+1}/{len(checkpoint_list)}: {checkpoint_name}")

            # 设置输出目录
            output_dir = os.path.join(
                args.dir_out_base,
                f"pre_train_{pre_train_idx}",
                checkpoint_name.replace('.pt', '')
            )

            # 执行deploy和评估
            avg_psnr = deploy_and_evaluate(
                args,
                experiment_dir,
                checkpoint_name,
                output_dir,
                args.dir_gt
            )

            pre_train_results.append(avg_psnr)

        results.append(pre_train_results)

    # 输出最终结果表格
    print("\n" + "="*80)
    print("最终结果表格 (PSNR)")
    print("="*80)

    # 表头
    header = "Pre-train Model".ljust(40)
    for checkpoint_name in checkpoint_list:
        checkpoint_short = checkpoint_name.replace('checkpoint_step_', '').replace('.pt', '')
        header += f"{checkpoint_short}".center(15)
    print(header)
    print("-" * len(header))

    # 表格内容
    for pre_train_idx, pre_train_path in enumerate(pre_train_list):
        model_name = os.path.basename(pre_train_path).replace('.pt', '')
        row = model_name.ljust(40)
        for psnr in results[pre_train_idx]:
            row += f"{psnr:.4f}".center(15)
        print(row)

    print("="*80)

    # 保存结果到文件
    result_file = os.path.join(args.dir_out_base, 'psnr_results.txt')
    with open(result_file, 'w') as f:
        f.write("批量Deploy和PSNR计算结果\n")
        f.write("="*80 + "\n\n")

        f.write(f"Pre-train模型: {len(pre_train_list)}\n")
        for i, p in enumerate(pre_train_list):
            f.write(f"  {i+1}. {p}\n")

        f.write(f"\nCheckpoint数量: {len(checkpoint_list)}\n")
        for i, c in enumerate(checkpoint_list):
            f.write(f"  {i+1}. {c}\n")

        f.write("\n" + "-"*80 + "\n")

        # 表头
        header = "Pre-train Model".ljust(40)
        for checkpoint_name in checkpoint_list:
            checkpoint_short = checkpoint_name.replace('checkpoint_step_', '').replace('.pt', '')
            header += f"{checkpoint_short}".center(15)
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        # 表格内容
        for pre_train_idx, pre_train_path in enumerate(pre_train_list):
            model_name = os.path.basename(pre_train_path).replace('.pt', '')
            row = model_name.ljust(40)
            for psnr in results[pre_train_idx]:
                row += f"{psnr:.4f}".center(15)
            f.write(row + "\n")

        f.write("="*80 + "\n")

    print(f"\n结果已保存到: {result_file}")


if __name__ == '__main__':
    main()
