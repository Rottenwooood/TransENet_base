import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import numbers
from einops import rearrange
from model import common
from typing import List, Optional

#from utils.registry import ARCH_REGISTRY

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MIN_NUM_PATCHES = 12


def make_model(args, parent=False):
    return SymUNet_Pretrain_MLLA_Dynamic(args)


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


# ============== Dynamic RoPE (Rotary Position Embedding) ==============
class DynamicRoPE(nn.Module):
    r"""Rotary positional embedding with spatial-shape cache."""
    def __init__(self, base=10000):
        super().__init__()
        self.base = base
        self.cache = {}

    def _get_rotations(self, h, w, feature_dim, device):
        if feature_dim % 4 != 0:
            raise ValueError(f"RoPE expects feature_dim divisible by 4, got {feature_dim}")

        key = (h, w, feature_dim, device)
        if key not in self.cache:
            k_max = feature_dim // 4
            theta_ks = 1 / (self.base ** (torch.arange(k_max, device=device, dtype=torch.float32) / k_max))
            yy = torch.arange(h, device=device, dtype=torch.float32)
            xx = torch.arange(w, device=device, dtype=torch.float32)
            yy, xx = torch.meshgrid(yy, xx, indexing='ij')
            angles = torch.cat([yy.unsqueeze(-1) * theta_ks, xx.unsqueeze(-1) * theta_ks], dim=-1)
            rotations = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
            self.cache[key] = rotations
        return self.cache[key]

    def forward(self, x, h, w):
        rotations = self._get_rotations(h, w, x.shape[-1], x.device)
        original_dtype = x.dtype
        x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(rotations).unsqueeze(0) * x
        return torch.view_as_real(pe_x).flatten(-2).to(original_dtype)


# ============== MLLA Linear Attention ==============
class MLLA(nn.Module):
    r"""MLLA with dynamic RoPE and output projection."""
    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = DynamicRoPE()
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        if n != h * w:
            raise ValueError(f"MLLA expected n == h * w, got n={n}, h={h}, w={w}")
        if c % self.num_heads != 0:
            raise ValueError(f"MLLA expected dim divisible by num_heads, got dim={c}, heads={self.num_heads}")

        head_dim = c // self.num_heads
        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, h, w, c), h, w).reshape(b, n, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, h, w, c), h, w).reshape(b, n, self.num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, self.num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        local_v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(local_v).permute(0, 2, 3, 1).reshape(b, n, c)
        return self.out_proj(x)


# ============== MLLA2D Wrapper ==============
class MLLA2D(nn.Module):
    """MLLA 2D wrapper with dynamic latent resolution instead of a fixed 14x14 grid."""
    def __init__(self, dim, num_heads=2, max_tokens=256, min_tokens=64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.mlla = MLLA(dim=dim, num_heads=num_heads, qkv_bias=True)

    def _get_latent_size(self, h, w):
        if self.max_tokens <= 0 or h * w <= self.max_tokens:
            return h, w

        pool_stride = max(1, math.ceil(math.sqrt((h * w) / self.max_tokens)))
        target_h = max(1, math.ceil(h / pool_stride))
        target_w = max(1, math.ceil(w / pool_stride))

        min_side = int(math.sqrt(self.min_tokens)) if self.min_tokens > 0 else 1
        target_h = min(h, max(min_side if h >= min_side else 1, target_h))
        target_w = min(w, max(min_side if w >= min_side else 1, target_w))

        while target_h * target_w > self.max_tokens:
            if target_h >= target_w and target_h > 1:
                target_h -= 1
            elif target_w > 1:
                target_w -= 1
            else:
                break

        return target_h, target_w

    def forward(self, x):
        b, c, h, w = x.shape
        latent_h, latent_w = self._get_latent_size(h, w)

        if latent_h != h or latent_w != w:
            pooled = F.adaptive_avg_pool2d(x, (latent_h, latent_w))
        else:
            pooled = x

        x_flat = pooled.view(b, c, latent_h * latent_w).permute(0, 2, 1).contiguous()
        x_flat = self.mlla(x_flat, latent_h, latent_w)
        x = x_flat.permute(0, 2, 1).contiguous().view(b, c, latent_h, latent_w)

        if latent_h != h or latent_w != w:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x


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
        x = self.fc1(x)
        x = x + self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = self.act(x1) * x2
        x = self.fc2(x)
        return x


# ============== StripModule ==============
class StripModule(nn.Module):
    """条形卷积模块 - StripNet StripBlock"""
    def __init__(self, dim, k1=1, k2=47):
        super().__init__()
        self.dim = dim
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial1 = nn.Conv2d(dim, dim, kernel_size=(k1, k2),
                                        stride=1, padding=(k1 // 2, k2 // 2), groups=dim)
        self.conv_spatial2 = nn.Conv2d(dim, dim, kernel_size=(k2, k1),
                                        stride=1, padding=(k2 // 2, k1 // 2), groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial1(attn)
        attn = self.conv_spatial2(attn)
        attn = self.conv1(attn)
        return x * attn


# ============== StripAttention ==============
class StripAttention(nn.Module):
    """条形注意力模块"""
    def __init__(self, dim, k1=1, k2=47):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = StripModule(dim, k1=k1, k2=k2)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x


# ============== StripMiddleBlock ==============
class StripMiddleBlock(nn.Module):
    """Middle Block using StripAttention"""
    def __init__(self, dim, ffn_expansion_factor=2., bias=False, k1=1, k2=47):
        super().__init__()
        self.dim = dim
        self.norm1 = LayerNorm2d(channels=dim)
        self.strip_attn = StripAttention(dim=dim, k1=k1, k2=k2)
        self.norm2 = LayerNorm2d(channels=dim)
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        layer_scale_init = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.layer_scale_1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * \
            self.strip_attn(self.norm1(x))
        shortcut = x
        y = self.norm2(x)
        y = self.project_in(y)
        y1, y2 = self.dwconv(y).chunk(2, dim=1)
        y = F.gelu(y1) * y2
        y = self.project_out(y)
        x = shortcut + self.layer_scale_2.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * y
        return x


# ============== MAB with MLLA ==============
class MAB(nn.Module):
    """Multi-head Attention Block using dynamic MLLA."""
    def __init__(self, dim, num_head=2):
        super().__init__()
        self.dim = dim
        self.num_head = num_head

        self.norm1 = LayerNorm2d(channels=dim)
        self.attn = MLLA2D(dim=dim, num_heads=num_head, max_tokens=256, min_tokens=64)
        print(f"[MAB] 使用动态 MLLA2D (num_head={num_head}, max_tokens=256)")

        self.norm2 = LayerNorm2d(channels=dim)
        self.ffn = MSConvStar(dim=dim, mlp_ratio=2., dw_sizes=[1, 3, 5, 7])

    def forward(self, x):
        shortcut_attn = x
        x_norm1 = self.norm1(x)
        attn_out = self.attn(x_norm1)
        x = shortcut_attn + attn_out

        shortcut_ffn = x
        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2)
        x = shortcut_ffn + ffn_out
        return x


# ============== S1_Trans Block (NoMAB1 - 仅保留MAB2) ==============
class S1_TransBlock_NoMAB1(nn.Module):
    """S1 Series Block (去掉MAB1): x = LAB(x) -> MAB -> Conv -> +shortcut"""
    def __init__(self, c, drop_out_rate=0.):
        super().__init__()
        self.lab = LAB(dim=c, local_dwconv=3, expanded_ratio=1., squeeze_factor=4)
        self.mab2 = MAB(dim=c, num_head=2)
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        shortcut = x
        x = self.lab(x)
        x = self.mab2(x)
        x = self.conv(x)
        return shortcut + x


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


class SymUNet_Pretrain_MLLA_Dynamic(nn.Module):
    """symunet_pretrain_mlla_dynamic: MAB使用动态 latent 分辨率的 MLLA"""
    def __init__(self, args, conv=common.default_conv):
        super(SymUNet_Pretrain_MLLA_Dynamic, self).__init__()

        self.args = args
        self.scale = args.scale[0]

        img_channel = args.n_colors
        width = getattr(args, 'symunet_pretrain_width', 32)
        middle_blk_num = getattr(args, 'symunet_pretrain_middle_blk_num', 1)
        enc_blk_nums = getattr(args, 'symunet_pretrain_enc_blk_nums', [4,6])
        dec_blk_nums = getattr(args, 'symunet_pretrain_dec_blk_nums', [6,4])

        ffn_expansion_factor = getattr(args, 'symunet_pretrain_ffn_expansion_factor', 2.66)
        bias = getattr(args, 'symunet_pretrain_bias', False)

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
                S1_TransBlock_NoMAB1(c=chan, drop_out_rate=drop_out_rate) for _ in range(num)
            ]))
            self.downs.append(DownsampleDW(chan))
            chan *= 2

        strip_k1 = getattr(args, 'symunet_pretrain_strip_k1', 1)
        strip_k2 = getattr(args, 'symunet_pretrain_strip_k2', 47)

        self.middle_blks = nn.Sequential(*[
            StripMiddleBlock(
                dim=chan,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                k1=strip_k1,
                k2=strip_k2
            ) for _ in range(middle_blk_num)
        ])

        for i, num in enumerate(dec_blk_nums):
            self.ups.append(UpsampleDW(chan))
            chan //= 2
            self.decoders.append(nn.Sequential(*[
                S1_TransBlock_NoMAB1(c=chan, drop_out_rate=drop_out_rate) for _ in range(num)
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
    model = SymUNet_Pretrain_MLLA_Dynamic(args)
    model.eval()
    input_lr = torch.rand(1, 3, 48, 48)
    sr = model(input_lr)
    print(f"输入LR尺寸: {input_lr.size()}")
    print(f"输出SR尺寸: {sr.size()}")
