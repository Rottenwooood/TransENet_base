import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from model import common
from typing import List, Optional

#from utils.registry import ARCH_REGISTRY

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MIN_NUM_PATCHES = 12


def make_model(args, parent=False):
    return SymUNet_Pretrain_S1_Trans_NoMAB1_Conv_Mask(args)


# ============== Channel Attention ==============
class ChannelAttention(nn.Module):
    def __init__(self, dim, squeeze_factor=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // squeeze_factor, 1),
            nn.SiLU(),
            nn.Conv2d(dim // squeeze_factor, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


# ============== LAB (Local Aggregation Block) ==============
class LAB(nn.Module):
    """Local Aggregation Block from MAT"""
    def __init__(self, dim, local_dwconv=3, expanded_ratio=1., squeeze_factor=4):
        super().__init__()
        hidden_dim = int(dim * expanded_ratio)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1), nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, local_dwconv, padding=local_dwconv // 2, groups=hidden_dim), nn.GELU(),
            ChannelAttention(dim=hidden_dim, squeeze_factor=squeeze_factor), nn.Conv2d(hidden_dim, dim, 1))

    def forward(self, x):
        u = x.clone()
        x = self.net(x)
        return u + x


# ============== MSDWConv (Multi-Scale Depthwise Conv) ==============
class MSDWConv(nn.Module):
    """Multi-Scale Depthwise Convolution from MAT"""
    def __init__(self, dim, dw_sizes=(1, 3, 5, 7)):
        super().__init__()
        self.dim = dim
        self.dw_sizes = dw_sizes
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(dw_sizes)):
            if i == 0:
                channels = dim - dim // len(dw_sizes) * (len(dw_sizes) - 1)
            else:
                channels = dim // len(dw_sizes)
            conv = nn.Conv2d(channels, channels, kernel_size=dw_sizes[i], padding=dw_sizes[i] // 2, groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x


# ============== MSConvStar (Multi-Scale Conv Star) ==============
class MSConvStar(nn.Module):
    """Multi-Scale Conv Star from MAT"""
    def __init__(self, dim, mlp_ratio=2., dw_sizes=[1, 3, 5, 7]):
        super().__init__()
        self.dim = dim
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.dwconv = MSDWConv(dim=hidden_dim, dw_sizes=dw_sizes)
        self.fc2 = nn.Conv2d(hidden_dim // 2, dim, 1)
        self.num_head = len(dw_sizes)
        self.act = nn.GELU()

        assert hidden_dim // self.num_head % 2 == 0

    def forward(self, x):
        # x is (B, C, H, W) format
        x = self.fc1(x)
        x = x + self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = self.act(x1) * x2
        x = self.fc2(x)
        return x


# ============== SimpleGate ==============
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# ============== LayerNorm2d ==============
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class DecomposedKernel(nn.Module):
    """
    大核分解模块 (VapSR 风格 ds -> ds-dilated)
    """
    def __init__(self, dim, k1, d1, k2, d2):
        super().__init__()
        # 第一阶段卷积
        p1 = (k1 - 1) * d1 // 2
        self.conv1 = nn.Conv2d(dim, dim, k1, padding=p1, dilation=d1, groups=dim)
        # 第二阶段卷积
        p2 = (k2 - 1) * d2 // 2
        self.conv2 = nn.Conv2d(dim, dim, k2, padding=p2, dilation=d2, groups=dim)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class LargeKernelMAB_Mask(nn.Module):
    """
    基于大核分解卷积生成Mask的多尺度注意力机制
    Branch 1 (dil=5 head): X1 -> DecomposedKernel(3,1,5,5) -> Mask1 -> X1 * Mask1 = Out1
    Branch 2 (dil=3 head): X2 -> DecomposedKernel(5,1,7,3) -> Mask2 -> X2 * Mask2 = Out2
    Concat(Out1, Out2) -> 1x1 mixing
    """
    def __init__(self, dim, kernel_sizes=[7, 11], dilations=[1, 1]):
        super().__init__()
        half = dim // 2

        # ---- Mask生成分支1 (dil=5头): 3x3(dense) + 5x5(dil=5) ----
        self.norm1 = LayerNorm2d(dim)
        # DecomposedKernel(dim, k1=3, d1=1, k2=5, d2=5)
        self.branch1_decomp = DecomposedKernel(half, k1=3, d1=1, k2=5, d2=5)
        # Sigmoid生成Mask
        self.branch1_conv = nn.Sequential(
            nn.Conv2d(half, half, 1), nn.Sigmoid()
        )

        # ---- Mask生成分支2 (dil=3头): 5x5(dense) + 7x7(dil=3) ----
        self.norm2 = LayerNorm2d(dim)
        # DecomposedKernel(dim, k1=5, d1=1, k2=7, d2=3)
        self.branch2_decomp = DecomposedKernel(half, k1=5, d1=1, k2=7, d2=3)
        # Sigmoid生成Mask
        self.branch2_conv = nn.Sequential(
            nn.Conv2d(half, half, 1), nn.Sigmoid()
        )

        # ---- 融合层 ----
        self.mix = nn.Conv2d(dim, dim, 1)

        # ---- FFN 部分 ----
        self.norm3 = LayerNorm2d(dim)
        self.ffn = MSConvStar(dim=dim, mlp_ratio=2., dw_sizes=[1, 3, 5, 7])

    def forward(self, x):
        # Part 1: Mask-based 多尺度特征提取
        shortcut = x
        x = self.norm1(x)
        x1, x2 = x.chunk(2, dim=1)

        # Branch 1: DecomposedKernel(3,1,5,5) -> Mask1 -> X1 * Mask1
        x1_feat = self.branch1_decomp(x1)
        mask1 = self.branch1_conv(x1_feat)  # 生成Mask
        out1 = x1 * mask1  # 应用Mask

        # Branch 2: DecomposedKernel(5,1,7,3) -> Mask2 -> X2 * Mask2
        x2_feat = self.branch2_decomp(x2)
        mask2 = self.branch2_conv(x2_feat)  # 生成Mask
        out2 = x2 * mask2  # 应用Mask

        # Concat + 1x1混合
        x = torch.cat([out1, out2], dim=1)
        x = self.mix(x)
        x = shortcut + x

        # Part 2: FFN（保持不变）
        x = x + self.ffn(self.norm3(x))
        return x


# ============== S1_Trans Block (NoMAB1 + Conv替换MA) ==============
class S1_TransBlock_NoMAB1_Conv_Mask(nn.Module):
    """
    S1 Series Block (去掉MAB1，用Mask版大核卷积替换MAB2):
    x = LAB(x) -> LargeKernelMAB_Mask(7,11,dil=[1,1]) -> Conv -> +shortcut
    """
    def __init__(self, c, drop_out_rate=0.):
        super().__init__()
        # LAB for local aggregation
        self.lab = LAB(dim=c, local_dwconv=3, expanded_ratio=1., squeeze_factor=4)

        # 用 LargeKernelMAB_Mask 替换 MAB1
        self.lk_mab = LargeKernelMAB_Mask(dim=c, kernel_sizes=[7, 11], dilations=[5, 3])

        # Conv + residual (like RMAG) with zero initialization
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        shortcut = x  # 保存输入用于残差连接

        # LAB -> LargeKernelMAB_Mask -> Conv (去掉MAB1，用Mask版大核卷积替换MAB2)
        x = self.lab(x)
        x = self.lk_mab(x)
        x = self.conv(x)

        return shortcut + x  # 整体残差连接


# ============== FeedForward & Attention for TransformerBlock ==============
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x_out = x + self.attn(self.norm1(x))
        x_out = x_out + self.ffn(self.norm2(x_out))
        return x_out


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return rearrange(self.body(rearrange(x, 'b c h w -> b (h w) c')), 'b (h w) c -> b c h w', h=h, w=w)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


# ========== Downsample/Upsample ==========
class DownsampleDW(nn.Module):
    def __init__(self, n_feat):
        super(DownsampleDW, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3,
                      stride=1, padding=1, groups=n_feat, bias=False),
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat // 2, kernel_size=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class UpsampleDW(nn.Module):
    def __init__(self, n_feat):
        super(UpsampleDW, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3,
                      stride=1, padding=1, groups=n_feat, bias=False),
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat * 2, kernel_size=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


#@ARCH_REGISTRY.register("CSymUNet_Pretrain_S1_Trans_NoMAB1_Conv_Mask")
class SymUNet_Pretrain_S1_Trans_NoMAB1_Conv_Mask(nn.Module):
    """
    S1_Trans_NoMAB1_Conv_Mask: 去掉MAB1，用Mask版大核卷积替换MAB2
    - 使用 LAB (Local Aggregation Block)
    - 使用 LargeKernelMAB_Mask (基于Mask的多尺度注意力)
    - 使用 MSConvStar
    """
    def __init__(self, args, conv=common.default_conv):
        super(SymUNet_Pretrain_S1_Trans_NoMAB1_Conv_Mask, self).__init__()

        self.args = args
        self.scale = args.scale[0]

        img_channel = args.n_colors
        width = getattr(args, 'symunet_pretrain_width', 32)
        middle_blk_num = getattr(args, 'symunet_pretrain_middle_blk_num', 1)
        enc_blk_nums = getattr(args, 'symunet_pretrain_enc_blk_nums', [4,6])
        dec_blk_nums = getattr(args, 'symunet_pretrain_dec_blk_nums', [6,4])

        ffn_expansion_factor = getattr(args, 'symunet_pretrain_ffn_expansion_factor', 2.66)
        bias = getattr(args, 'symunet_pretrain_bias', False)
        LayerNorm_type = getattr(args, 'symunet_pretrain_layer_norm_type', 'WithBias')

        restormer_heads = getattr(args, 'symunet_pretrain_restormer_heads', [1, 2, 4])
        restormer_middle_heads = getattr(args, 'symunet_pretrain_restormer_middle_heads', 8)

        drop_out_rate = getattr(args, 'symunet_pretrain_dropout', 0.)

        self.pre_upsample = nn.Upsample(scale_factor=self.scale, mode='bicubic', align_corners=False)

        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for i, num in enumerate(enc_blk_nums):
            self.encoders.append(nn.Sequential(*[
                S1_TransBlock_NoMAB1_Conv_Mask(c=chan, drop_out_rate=drop_out_rate) for _ in range(num)
            ]))
            self.downs.append(DownsampleDW(chan))
            chan *= 2

        self.middle_blks = nn.Sequential(*[
            TransformerBlock(
                dim=chan,
                num_heads=restormer_middle_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(middle_blk_num)
        ])

        for i, num in enumerate(dec_blk_nums):
            self.ups.append(UpsampleDW(chan))
            chan //= 2

            self.decoders.append(nn.Sequential(*[
                S1_TransBlock_NoMAB1_Conv_Mask(c=chan, drop_out_rate=drop_out_rate) for _ in range(num)
            ]))

        self.padder_size = (2 ** len(self.encoders)) * 4

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp_upsampled = self.pre_upsample(inp)
        inp_padded = self.check_image_size(inp_upsampled)
        x = self.intro(inp_padded)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder_blocks, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder_blocks(x)

        x = self.ending(x)
        x = x + inp_padded

        H_target = H * self.scale
        W_target = W * self.scale
        final_image_output = x[:, :, :H_target, :W_target]

        return final_image_output

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


if __name__ == "__main__":
    from option import args
    model = SymUNet_Pretrain_S1_Trans_NoMAB1_Conv_Mask(args)
    model.eval()
    input_lr = torch.rand(1, 3, 48, 48)
    sr = model(input_lr)
    print(f"输入LR尺寸: {input_lr.size()}")
    print(f"输出SR尺寸: {sr.size()}")