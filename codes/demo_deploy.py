from option import args
import model
import utils
import data.common as common

import torch
import numpy as np
import os
import glob
import cv2
import time
import psutil
from thop import profile
from torch.utils.tensorboard import SummaryWriter

# Additional imports for FLOPs calculation
PTFLOPS_AVAILABLE = False
FVCORE_AVAILABLE = False

try:
    import ptflops
    PTFLOPS_AVAILABLE = True
except (ImportError, AttributeError) as e:
    PTFLOPS_AVAILABLE = False

try:
    from fvcore.nn import flop_count
    FVCORE_AVAILABLE = True
except (ImportError, AttributeError) as e:
    FVCORE_AVAILABLE = False

device = torch.device('cpu' if args.cpu else 'cuda')


def test_model_performance(args, sr_model, sample_input=None):
    """测试模型性能：参数量、FLOPs、内存占用、处理时间"""

    print("\n" + "="*60)
    print("模型性能测试报告")
    print("="*60)

    # 1. 测试参数量
    total_params = sum(p.numel() for p in sr_model.parameters())
    trainable_params = sum(p.numel() for p in sr_model.parameters() if p.requires_grad)

    print(f"📊 模型参数量:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")  # 假设float32

    # 2. 测试FLOPs
    print(f"\n🔢 FLOPs计算:")
    if sample_input is None:
        # 创建一个示例输入
        if not args.cubic_input:
            sample_input = torch.randn(1, 3, 64, 64).to(device)
        else:
            sample_input = torch.randn(1, 3, 256, 256).to(device)

    flops = 0
    params = 0

    # Method 1: thop (original)
    try:
        flops_thop, params_thop = profile(sr_model, inputs=(sample_input,), verbose=False)
        flops += flops_thop
        params = params_thop
        print(f"  [thop] FLOPs: {flops_thop:,}")
        print(f"  [thop] FLOPs (G): {flops_thop / 1e9:.2f} G")
        print(f"  [thop] 参数量 (M): {params_thop / 1e6:.2f} M")
    except Exception as e:
        print(f"  [thop] FLOPs计算失败: {e}")

    # Method 2: ptflops
    flops_ptflops = None
    params_ptflops = None
    if PTFLOPS_AVAILABLE:
        try:
            # Get input shape
            input_shape = (3, sample_input.shape[2], sample_input.shape[3])

            # Use ptflops get_model_complexity_info with positional arguments
            macs, params_ptflops = ptflops.get_model_complexity_info(
                sr_model,
                input_shape,
                as_strings=False,  # Return numbers instead of strings
                input_constructor=lambda inputs: torch.randn(1, *input_shape).to(device),
                verbose=False
            )

            flops_ptflops = macs

            print(f"  [ptflops] FLOPs: {macs:,}")
            print(f"  [ptflops] FLOPs (G): {macs / 1e9:.2f} G")
            print(f"  [ptflops] 参数量 (M): {params_ptflops / 1e6:.2f} M")
        except Exception as e:
            print(f"  [ptflops] FLOPs计算失败: {e}")
            print(f"    错误详情: {str(e)}")
            flops_ptflops = None
            params_ptflops = None
    else:
        print(f"  [ptflops] 库未安装，跳过计算")

    # Method 3: fvcore
    total_flops = None
    if FVCORE_AVAILABLE:
        try:
            # fvcore's flop_count returns (total_flops, detailed_stats)
            result = flop_count(sr_model, (sample_input,))
            if isinstance(result, tuple):
                total_flops, flop_dict = result
            else:
                # If it's a dict directly
                flop_dict = result
                total_flops = sum(flop_dict.values())

            print(f"  [fvcore] FLOPs: {total_flops:,}")
            print(f"  [fvcore] FLOPs (G): {total_flops / 1e9:.2f} G")
            print(f"  [fvcore] 详细统计:")
            if isinstance(flop_dict, dict):
                for op, count in flop_dict.items():
                    print(f"    {op}: {count:,}")
            else:
                print(f"    {flop_dict}")
        except Exception as e:
            print(f"  [fvcore] FLOPs计算失败: {e}")
            total_flops = None
    else:
        print(f"  [fvcore] 库未安装，跳过计算")

    # Comparison summary
    if PTFLOPS_AVAILABLE or FVCORE_AVAILABLE:
        print(f"\n  📊 各工具计算结果对比:")
        print(f"    thop: {flops / 1e9:.2f} G FLOPs")
        if PTFLOPS_AVAILABLE and flops_ptflops is not None:
            print(f"    ptflops: {flops_ptflops / 1e9:.2f} G FLOPs")
        if FVCORE_AVAILABLE and total_flops is not None:
            print(f"    fvcore: {total_flops / 1e9:.2f} G FLOPs")

    # 3. 测试内存占用
    print(f"\n💾 内存占用:")

    # GPU内存占用
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # 记录初始GPU内存
        if args.test_block:
            initial_memory = torch.cuda.memory_allocated()

            # 执行前向传播
            with torch.no_grad():
                _ = sr_model(sample_input)

            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = peak_memory - initial_memory

            print(f"  GPU内存占用: {memory_used / 1024 / 1024:.2f} MB")
        else:
            initial_memory = torch.cuda.memory_allocated()
            with torch.no_grad():
                _ = sr_model(sample_input)
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = peak_memory - initial_memory
            print(f"  GPU内存占用: {memory_used / 1024 / 1024:.2f} MB")
    else:
        print(f"  CPU模式 - GPU内存: 0 MB")

    # CPU内存占用
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / 1024 / 1024
    print(f"  CPU内存占用: {cpu_memory:.2f} MB")

    # 4. 测试处理时间
    print(f"\n⏱️  处理时间测试:")

    # 预热
    with torch.no_grad():
        for _ in range(5):
            _ = sr_model(sample_input)

    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.synchronize()

    # 正式测试
    times = []
    for i in range(10):  # 测试10次取平均
        if torch.cuda.is_available() and not args.cpu:
            torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                _ = sr_model(sample_input)

            torch.cuda.synchronize()
            end_time = time.time()
        else:
            start_time = time.time()
            with torch.no_grad():
                _ = sr_model(sample_input)
            end_time = time.time()

        elapsed = end_time - start_time
        times.append(elapsed)

        if i == 0:  # 第一次可能比较慢，跳过
            continue

    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"  平均处理时间: {avg_time * 1000:.2f} ms")
    print(f"  标准差: {std_time * 1000:.2f} ms")
    print(f"  最快时间: {min_time * 1000:.2f} ms")
    print(f"  最慢时间: {max_time * 1000:.2f} ms")

    # 5. 吞吐量计算
    input_size = sample_input.shape
    if not args.cubic_input:
        output_size = (input_size[2] * args.scale[0], input_size[3] * args.scale[0])
    else:
        output_size = input_size[2:]

    pixels_processed = output_size[0] * output_size[1]
    throughput = pixels_processed / (avg_time * 1000)  # 像素/毫秒

    print(f"\n📈 吞吐量:")
    print(f"  输入尺寸: {input_size[2]}x{input_size[3]}")
    print(f"  输出尺寸: {output_size[0]}x{output_size[1]}")
    print(f"  吞吐量: {throughput:.0f} 像素/毫秒")
    print(f"  FPS (估算): {1000/avg_time:.1f}")

    print("="*60 + "\n")

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops': flops if 'flops' in locals() else 0,
        'memory_used': memory_used if 'memory_used' in locals() else 0,
        'avg_time': avg_time,
        'fps': 1000/avg_time if avg_time > 0 else 0
    }


def deploy(args, sr_model):

    img_ext = '.tif'
    img_lists = glob.glob(os.path.join(args.dir_data, '*'+img_ext))

    if len(img_lists) == 0:
        print("Error: there are no images in given folder!")

    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out)

    with torch.no_grad():
        for i in range(len(img_lists)):
            print("[%d/%d] %s" % (i+1, len(img_lists), img_lists[i]))
            lr_np = cv2.imread(img_lists[i], cv2.IMREAD_COLOR)
            lr_np = cv2.cvtColor(lr_np, cv2.COLOR_BGR2RGB)

            if args.cubic_input:
                lr_np = cv2.resize(lr_np, (lr_np.shape[0] * args.scale[0], lr_np.shape[1] * args.scale[0]),
                                interpolation=cv2.INTER_CUBIC)

            lr = common.np2Tensor([lr_np], args.rgb_range)[0].unsqueeze(0)

            if args.test_block:
                # test block-by-block

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

                        # forward-pass
                        lr_p = lr[:, :, iy:iy + ip, ix:ix + ip]
                        lr_p = lr_p.to(device)
                        sr_p = sr_model(lr_p)
                        sr[:, :, ty:ty + tp, tx:tx + tp] = sr_p

            else:

                lr = lr.to(device)
                sr = sr_model(lr)

            sr_np = np.array(sr.cpu().detach())
            sr_np = sr_np[0, :].transpose([1, 2, 0])
            lr_np = lr_np * args.rgb_range / 255.

            # Again back projection for the final fused result
            for bp_iter in range(args.back_projection_iters):
                sr_np = utils.back_projection(sr_np, lr_np, down_kernel='cubic',
                                           up_kernel='cubic', sf=args.scale[0], range=args.rgb_range)
            if args.rgb_range == 1:
                final_sr = np.clip(sr_np * 255, 0, args.rgb_range * 255)
            else:
                final_sr = np.clip(sr_np, 0, args.rgb_range)

            final_sr = final_sr.astype(np.uint8)
            final_sr = cv2.cvtColor(final_sr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.dir_out, os.path.split(img_lists[i])[-1]), final_sr)
    #    # 计算PSNR
    # folder_GT = '/root/autodl-tmp/TransENet_base/datasets/UCMerced-train/UCMerced-dataset/test/HR_x4'
    # folder_Gen = '/root/autodl-tmp/TransENet_base/experiment/results/UCMerced_UCMercedtest/x4'
    # img_ext = '.tif'
    # crop_border = 4  # same with scale
    # suffix = ''  # suffix for Gen images
    # test_Y = False  # True: test Y channel only; False: test RGB channels

    # PSNR_all = []
    # SSIM_all = []
    # print(f"\n计算PSNR...")

    # # 获取GT图像列表
    # gt_lists = sorted(glob.glob(os.path.join(folder_GT, '/*')))
    # for i, gt_path in enumerate(gt_lists):
    #     base_name = os.path.splitext(os.path.basename(gt_path))[0]
    #     gen_path = os.path.join(folder_Gen, base_name + img_ext)

    #     if not os.path.exists(gen_path):
    #         print(f"警告: 生成图像不存在: {gen_path}")
    #         continue

    #     # 读取图像
    #     im_GT = cv2.imread(gt_path) / 255.
    #     im_Gen = cv2.imread(gen_path) / 255.

    #     # 裁剪边界
    #     if crop_border == 0:
    #         cropped_GT = im_GT
    #         cropped_Gen = im_Gen
    #     else:
    #         if im_GT.ndim == 3:
    #             cropped_GT = im_GT[crop_border:-crop_border, crop_border:-crop_border, :]
    #             cropped_Gen = im_Gen[crop_border:-crop_border, crop_border:-crop_border, :]
    #         elif im_GT.ndim == 2:
    #             cropped_GT = im_GT[crop_border:-crop_border, crop_border:-crop_border]
    #             cropped_Gen = im_Gen[crop_border:-crop_border, crop_border:-crop_border]

    #     # 计算PSNR和SSIM
    #     PSNR = calculate_rgb_psnr(cropped_GT * 255, cropped_Gen * 255)
    #     SSIM = calculate_ssim(cropped_GT * 255, cropped_Gen * 255)

    #     PSNR_all.append(PSNR)
    #     SSIM_all.append(SSIM)

    #     print(f"  {base_name:30s} - PSNR: {PSNR:.4f} dB, SSIM: {SSIM:.4f}")

    # avg_psnr = sum(PSNR_all) / len(PSNR_all)
    # avg_ssim = sum(SSIM_all) / len(SSIM_all)

    # print(f"\n平均结果: PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.4f}")



if __name__ == '__main__':

    # args parameter setting
    # You can configure these paths via command line arguments or set them here
    # args.pre_train = '/root/autodl-tmp/TransENet_base/experiment/test_dir/pair01_NAF_symunet_blk46_w32/model/model_best.pt'
    
    # args.dir_data = '/root/autodl-tmp/TransENet_base/datasets/AID-train/AID-dataset/test/LR_x4'
    args.dir_data = '/root/autodl-tmp/TransENet_base/datasets/UCMerced-dataset/test/LR_x4'
    # args.dir_out = '../experiment/results/466L1W32/x4'

    print("Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Scale: {args.scale}")
    print(f"  Data dir: {args.dir_data}")
    print(f"  Output dir: {args.dir_out}")
    print(f"  Pre-trained model: {args.pre_train}")
    print("-" * 50)

    checkpoint = utils.checkpoint(args)
    sr_model = model.Model(args, checkpoint)
    sr_model.eval()

    # 模型性能测试
    test_model_performance(args, sr_model)

    # # analyse the params of the load model
    # pytorch_total_params = sum(p.numel() for p in sr_model.parameters())
    # print(pytorch_total_params)
    # pytorch_total_params2 = sum(p.numel() for p in sr_model.parameters() if p.requires_grad)
    # print(pytorch_total_params2)
    #
    # for name, p in sr_model.named_parameters():
    #     print(name)
    #     print(p.numel())
    #     print('========')

    # deploy(args, sr_model)