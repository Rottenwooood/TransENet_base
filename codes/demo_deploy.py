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

    try:
        flops, params = profile(sr_model, inputs=(sample_input,), verbose=False)
        print(f"  FLOPs: {flops:,}")
        print(f"  FLOPs (G): {flops / 1e9:.2f} G")
        print(f"  参数量 (M): {params / 1e6:.2f} M")
    except Exception as e:
        print(f"  FLOPs计算失败: {e}")

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



if __name__ == '__main__':

    # args parameter setting
    # You can configure these paths via command line arguments or set them here
    args.pre_train = '/root/autodl-tmp/TransENet/experiment/TRANSENETx4_UCMerced/model/model_best.pt'
    # args.dir_data = '/root/autodl-tmp/TransENet/datasets/AID-train/AID-dataset/test/LR_x4'
    args.dir_data = '/root/autodl-tmp/TransENet/datasets/UCMerced-train/UCMerced-dataset/test/LR_x4'
    args.dir_out = '../experiment/results/UCMerced_UCMercedtest/x4'

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

    deploy(args, sr_model)