"""
MAT (Multi-scale Adaptive Transformer) Model for Image Super-Resolution

Based on: MAT/basicsr/archs/mat_arch.py
Training config: MAT/options/train/MAT/train_MAT_x4_finetune.yml
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import trunc_normal_
from typing import List, Optional

try:
    from natten.functional import na2d_av, na2d_qk
    NATTERN_AVAILABLE = True
except ImportError:
    NATTERN_AVAILABLE = False
    print("Warning: NATTEN is not installed. NeighborhoodAttention2D will not work.")

from model import common
#from utils.registry import ARCH_REGISTRY


def make_model(args, parent=False):
    return MAT(args)


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


# ============== Local Attention Block ==============

class LAB(nn.Module):
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


# ============== Neighborhood Attention 2D ==============

class NeighborhoodAttention2D(nn.Module):
    """
    Neighborhood Attention 2D Module
    Requires NATTEN library: pip install natten
    """

    def __init__(
        self,
        dim: int,
        num_head: int,
        kernel_sizes: List[int] = [7, 9, 11],
        dilations: List[int] = [1, 1, 1],
        is_causal: List[bool] = [False, False],
        rel_pos_bias: bool = False,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations)
        if any(is_causal) and rel_pos_bias:
            raise NotImplementedError("Causal neighborhood attention is undefined with positional biases."
                                      "Please consider disabling positional biases, or open an issue.")

        self.k = len(kernel_sizes)
        self.channels = []
        for i in range(self.k):
            if i == 0:
                channels = dim * 3 - dim * 3 // len(kernel_sizes) * (len(kernel_sizes) - 1)
            else:
                channels = dim * 3 // len(kernel_sizes)
            assert (channels % (3 * num_head // self.k) == 0)
            self.channels.append(channels)

        self.num_head = num_head
        self.head_dim = dim // self.num_head
        self.scale = qk_scale or self.head_dim**-0.5
        self.kernel_sizes = tuple((i, i) for i in kernel_sizes)
        self.dilations = tuple((i, i) for i in dilations)
        self.is_causal = is_causal

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if rel_pos_bias:
            self.rpb = nn.ParameterList()
            for i in range(len(kernel_sizes)):
                temp = nn.Parameter(torch.zeros(
                    num_head // self.k,
                    (2 * kernel_sizes[i] - 1),
                    (2 * kernel_sizes[i] - 1),
                ))
                trunc_normal_(temp, mean=0.0, std=0.02, a=-2.0, b=2.0)
                self.rpb.append(temp)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(self.attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        if not NATTERN_AVAILABLE:
            raise RuntimeError("NATTEN is required for NeighborhoodAttention2D. Install with: pip install natten")

        if x.dim() != 4:
            raise ValueError(f"NeighborhoodAttention2D expected a rank-4 input tensor; got {x.dim()=}.")

        x = self.qkv(x)
        x = torch.split(x, split_size_or_sections=self.channels, dim=3)
        attns = []
        for i, x_i in enumerate(x):
            B, H, W, C = x_i.shape
            qkv = (x_i.reshape(B, H, W, 3, self.num_head // self.k, self.head_dim).permute(3, 0, 4, 1, 2, 5))
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q * self.scale
            attn = na2d_qk(
                q,
                k,
                kernel_size=self.kernel_sizes[i],
                dilation=self.dilations[i],
                is_causal=self.is_causal,
                rpb=self.rpb[i] if self.rpb is not None else None,
            )
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            y = na2d_av(
                attn,
                v,
                kernel_size=self.kernel_sizes[i],
                dilation=self.dilations[i],
                is_causal=self.is_causal,
            )
            y = y.permute(0, 2, 3, 1, 4).reshape(B, H, W, C // 3)
            attns.append(y)
        x = torch.cat(attns, dim=3)
        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (f"head_dim={self.head_dim}, num_head={self.num_head}, " + f"kernel_sizes={self.kernel_sizes}, " +
                f"dilations={self.dilations}, " + f"is_causal={self.is_causal}, " + f"has_bias={self.rpb is not None}")


# ============== Multi-Scale Depthwise Convolution ==============

class MSDWConv(nn.Module):
    def __init__(self, dim, dw_sizes=(1, 3, 5, 7)):
        super().__init__()
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


# ============== Multi-Scale Conv Star ==============

class MSConvStar(nn.Module):
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
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.fc1(x)
        x = x + self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = self.act(x1) * x2
        x = self.fc2(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


# ============== Multi-head Attention Block ==============

class MAB(nn.Module):
    def __init__(self,
                 dim,
                 num_head,
                 kernel_sizes=[7, 9, 11],
                 dilations=[1, 1, 1],
                 rel_pos_bias=True,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 mlp_ratio=2.0,
                 dw_sizes=[1, 3, 5, 7]) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = NeighborhoodAttention2D(
            dim=dim,
            num_head=num_head,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MSConvStar(dim, mlp_ratio=mlp_ratio, dw_sizes=dw_sizes)
        self.dilations = dilations

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============== Residual Multi-head Attention Group ==============

class RMAG(nn.Module):
    def __init__(self,
                 dim,
                 local_dwconv=3,
                 expanded_ratio=1.,
                 squeeze_factor=4,
                 depth=4,
                 num_head=6,
                 kernel_sizes=[7, 9, 11],
                 dilations=[[1, 1, 1], [4, 4, 4]],
                 rel_pos_bias=True,
                 qkv_bias=True,
                 qk_scale=None,
                 mlp_ratio=2.0,
                 dw_sizes=[1, 3, 5, 7]):
        super().__init__()
        self.lab = LAB(dim=dim, local_dwconv=local_dwconv, expanded_ratio=expanded_ratio, squeeze_factor=squeeze_factor)
        self.mabs = nn.ModuleList()
        for i_mab in range(depth):
            mab = MAB(
                dim=dim,
                num_head=num_head,
                kernel_sizes=kernel_sizes,
                dilations=dilations[i_mab % 2],
                rel_pos_bias=rel_pos_bias,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                mlp_ratio=mlp_ratio,
                dw_sizes=dw_sizes)
            self.mabs.append(mab)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        shortcut = x
        x = self.lab(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        for mab in self.mabs:
            x = mab(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        x = x + shortcut
        return x


# ============== Upsample Modules ==============

def UpsampleOneStep(in_channels, out_channels, upscale_factor=4):
    """Upsample features according to `upscale_factor`."""
    conv = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=4):
        super().__init__()
        num_feat = 64
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(in_channels, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(upscale_factor, num_feat)
        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))
        return x


# ============== Main MAT Model ==============

#@ARCH_REGISTRY.register("MAT")
class MAT(nn.Module):
    """
    Multi-scale Adaptive Transformer (MAT) for Image Super-Resolution

    Args:
        args: Configuration arguments
        conv: Convolution layer (default: common.default_conv)

    Default parameters (MAT_light from train_MAT_light_x4_finetune.yml):
        num_in_ch: 3
        num_out_ch: 3
        num_feat: 60
        num_block: 4
        expanded_ratio: 1.0
        squeeze_factor: 4
        depth: 4
        num_head: 6
        kernel_sizes: [7, 9, 11]
        dilations: [[1, 1, 1], [9, 7, 5]]
        rel_pos_bias: true
        dw_sizes: [1, 3, 5, 7]
        upscale: 4
        upsampler: 'pixelshuffledirect'

    Full model (MAT) parameters from train_MAT_x4_finetune.yml:
        num_feat: 156, num_block: 6, expanded_ratio: 2.0, squeeze_factor: 2,
        depth: 6, kernel_sizes: [13, 15, 17], dilations: [[1, 1, 1], [4, 4, 3]],
        upsampler: 'pixelshuffle'
    """

    def __init__(self, args, conv=common.default_conv):
        super(MAT, self).__init__()

        # Extract parameters from args or use defaults from training config (MAT_light)
        self.num_in_ch = getattr(args, 'n_colors', 3)  # n_colors from args, default 3
        self.num_out_ch = getattr(args, 'n_colors', 3)
        self.num_feat = getattr(args, 'n_feats', 60)  # Default: 60 (MAT_light)
        self.num_block = getattr(args, 'mat_num_block', 4)  # Default: 4 (MAT_light)
        self.local_dwconv = getattr(args, 'mat_local_dwconv', 3)
        self.expanded_ratio = getattr(args, 'mat_expanded_ratio', 1.0)  # Default: 1.0 (MAT_light)
        self.squeeze_factor = getattr(args, 'mat_squeeze_factor', 4)  # Default: 4 (MAT_light)
        self.depth = getattr(args, 'mat_depth', 4)  # Default: 4 (MAT_light)
        self.num_head = getattr(args, 'mat_num_head', 6)
        self.kernel_sizes = getattr(args, 'mat_kernel_sizes', [7, 9, 11])  # Default: [7, 9, 11] (MAT_light)
        self.dilations = getattr(args, 'mat_dilations', [[1, 1, 1], [9, 7, 5]])  # Default: MAT_light
        self.rel_pos_bias = getattr(args, 'mat_rel_pos_bias', True)
        self.qkv_bias = getattr(args, 'mat_qkv_bias', True)
        self.qk_scale = getattr(args, 'mat_qk_scale', False)
        self.mlp_ratio = getattr(args, 'mat_mlp_ratio', 2.0)
        self.dw_sizes = getattr(args, 'mat_dw_sizes', [1, 3, 5, 7])
        self.upscale = getattr(args, 'scale', [4])[0] if hasattr(args, 'scale') else getattr(args, 'mat_upscale', 4)
        self.upsampler_type = getattr(args, 'mat_upsampler', 'pixelshuffledirect')  # Default: pixelshuffledirect (MAT_light)
        self.img_range = getattr(args, 'rgb_range', 1.0)

        # RGB mean for normalization (UCMerced dataset)
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.register_buffer('mean', torch.Tensor(rgb_mean).view(1, 3, 1, 1))

        # Feature extraction
        self.fea_conv = nn.Conv2d(self.num_in_ch, self.num_feat, 3, 1, 1)

        # Body: Multiple RMAG blocks
        self.body = nn.ModuleList()
        for _ in range(self.num_block):
            self.body.append(
                RMAG(
                    dim=self.num_feat,
                    local_dwconv=self.local_dwconv,
                    expanded_ratio=self.expanded_ratio,
                    squeeze_factor=self.squeeze_factor,
                    depth=self.depth,
                    num_head=self.num_head,
                    kernel_sizes=self.kernel_sizes,
                    dilations=self.dilations,
                    rel_pos_bias=self.rel_pos_bias,
                    qkv_bias=self.qkv_bias,
                    qk_scale=self.qk_scale,
                    mlp_ratio=self.mlp_ratio,
                    dw_sizes=self.dw_sizes
                )
            )

        self.conv_afterbody = nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1)

        # Upsampler
        if self.upsampler_type == 'pixelshuffledirect':
            self.upsampler = UpsampleOneStep(self.num_feat, self.num_out_ch, upscale_factor=self.upscale)
        elif self.upsampler_type == 'pixelshuffle':
            self.upsampler = PixelShuffleBlock(self.num_feat, self.num_out_ch, upscale_factor=self.upscale)
        else:
            raise NotImplementedError(f"Upsampler {self.upsampler_type} not supported.")

    def check_image_size(self, x, min_size=64):
        """Ensure input size is at least min_size for neighborhood attention"""
        _, _, H, W = x.size()
        if H >= min_size and W >= min_size:
            return x
        mod_pad_h = max(min_size - H, 0)
        mod_pad_w = max(min_size - W, 0)
        padding = (0, mod_pad_w, 0, mod_pad_h)
        x = F.pad(x, padding, 'reflect')
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        # Normalize input
        mean = self.mean
        if x.is_cuda and not mean.is_cuda:
            mean = mean.cuda()
        x = (x - mean) * self.img_range

        # Feature extraction
        x = self.fea_conv(x)

        # Body: Process through RMAG blocks
        for block in self.body:
            x = block(x)

        x = self.conv_afterbody(x) + x

        # Upsample
        x = self.upsampler(x)

        # Denormalize and crop to original size
        x = x / self.img_range + mean
        return x[:, :, :H * self.upscale, :W * self.upscale]

    def load_state_dict(self, state_dict, strict=False):
        """Load state dict with backward compatibility"""
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    print(f'Warning: While copying {name}, dimensions mismatch.')
            elif strict:
                print(f'Warning: Key {name} not found in model state_dict.')

    def get_model_info(self):
        """Print model information"""
        print(f"MAT Model Info:")
        print(f"  - Number of features: {self.num_feat}")
        print(f"  - Number of blocks: {self.num_block}")
        print(f"  - Depth per block: {self.depth}")
        print(f"  - Number of heads: {self.num_head}")
        print(f"  - Kernel sizes: {self.kernel_sizes}")
        print(f"  - Dilations: {self.dilations}")
        print(f"  - Upscale: {self.upscale}x")
        print(f"  - Upsampler: {self.upsampler_type}")
