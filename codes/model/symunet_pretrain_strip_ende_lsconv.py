import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from natten.functional import na2d_av, na2d_qk
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
    return SymUNet_Pretrain_Strip_ende_lsconv(args)

import math
import torch
from torch import nn
from torch.cuda.amp import custom_fwd, custom_bwd  # 补充缺失的 AMP 导入
import triton
import triton.language as tl

# ==========================================
# Triton SKA (Selective Kernel Attention) 核心算子
# ==========================================
def _grid(numel: int, bs: int) -> tuple:
    return (triton.cdiv(numel, bs),)

@triton.jit
def _idx(i, n: int, c: int, h: int, w: int):
    ni = i // (c * h * w)
    ci = (i // (h * w)) % c
    hi = (i // w) % h
    wi = i % w
    m = i < (n * c * h * w)
    return ni, ci, hi, wi, m
    
@triton.jit
def ska_fwd(x_ptr, w_ptr, o_ptr, n, ic, h, w, ks, pad, wc, BS: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BS
    offs = start + tl.arange(0, BS)

    ni, ci, hi, wi, m = _idx(offs, n, ic, h, w)
    val = tl.zeros((BS,), dtype=tl.float32)

    for kh in range(ks):
        hin = hi - pad + kh
        hb = (hin >= 0) & (hin < h)
        for kw in range(ks):
            win = wi - pad + kw
            b = hb & (win >= 0) & (win < w)

            x_off = ((ni * ic + ci) * h + hin) * w + win
            w_off = ((ni * wc + ci % wc) * ks * ks + (kh * ks + kw)) * h * w + hi * w + wi

            x_val = tl.load(x_ptr + x_off, mask=m & b, other=0.0).to(tl.float32)
            w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(tl.float32)
            val += tl.where(b & m, x_val * w_val, 0.0)

    tl.store(o_ptr + offs, val.to(o_ptr.dtype.element_ty), mask=m)

@triton.jit
def ska_bwd_x(go_ptr, w_ptr, gi_ptr, n, ic, h, w, ks, pad, wc, BS: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BS
    offs = start + tl.arange(0, BS)

    ni, ci, hi, wi, m = _idx(offs, n, ic, h, w)
    val = tl.zeros((BS,), dtype=tl.float32)

    for kh in range(ks):
        ho = hi + pad - kh
        hb = (ho >= 0) & (ho < h)
        for kw in range(ks):
            wo = wi + pad - kw
            b = hb & (wo >= 0) & (wo < w)

            go_off = ((ni * ic + ci) * h + ho) * w + wo
            w_off = ((ni * wc + ci % wc) * ks * ks + (kh * ks + kw)) * h * w + ho * w + wo

            go_val = tl.load(go_ptr + go_off, mask=m & b, other=0.0).to(tl.float32)
            w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(tl.float32)
            val += tl.where(b & m, go_val * w_val, 0.0)

    tl.store(gi_ptr + offs, val.to(gi_ptr.dtype.element_ty), mask=m)

@triton.jit
def ska_bwd_w(go_ptr, x_ptr, gw_ptr, n, wc, h, w, ic, ks, pad, BS: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BS
    offs = start + tl.arange(0, BS)

    ni, ci, hi, wi, m = _idx(offs, n, wc, h, w)

    for kh in range(ks):
        hin = hi - pad + kh
        hb = (hin >= 0) & (hin < h)
        for kw in range(ks):
            win = wi - pad + kw
            b = hb & (win >= 0) & (win < w)
            w_off = ((ni * wc + ci) * ks * ks + (kh * ks + kw)) * h * w + hi * w + wi

            val = tl.zeros((BS,), dtype=tl.float32)
            steps = (ic - ci + wc - 1) // wc
            for s in range(tl.max(steps, axis=0)):
                cc = ci + s * wc
                cm = (cc < ic) & m & b

                x_off = ((ni * ic + cc) * h + hin) * w + win
                go_off = ((ni * ic + cc) * h + hi) * w + wi

                x_val = tl.load(x_ptr + x_off, mask=cm, other=0.0).to(tl.float32)
                go_val = tl.load(go_ptr + go_off, mask=cm, other=0.0).to(tl.float32)
                val += tl.where(cm, x_val * go_val, 0.0)

            tl.store(gw_ptr + w_off, val.to(gw_ptr.dtype.element_ty), mask=m)

class SkaFn(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # 修复：把 math_lib.sqrt 改为标准的 math.sqrt
        ks = int(math.sqrt(w.shape[2]))
        pad = (ks - 1) // 2
        ctx.ks, ctx.pad = ks, pad
        n, ic, h, width = x.shape
        wc = w.shape[1]
        o = torch.empty(n, ic, h, width, device=x.device, dtype=x.dtype)
        numel = o.numel()

        x = x.contiguous()
        w = w.contiguous()

        grid = lambda meta: _grid(numel, meta["BS"])
        ska_fwd[grid](x, w, o, n, ic, h, width, ks, pad, wc, BS=1024)

        ctx.save_for_backward(x, w)
        return o

    @staticmethod
    @custom_bwd
    def backward(ctx, go: torch.Tensor) -> tuple:
        ks, pad = ctx.ks, ctx.pad
        x, w = ctx.saved_tensors
        n, ic, h, width = x.shape
        wc = w.shape[1]

        go = go.contiguous()
        gx = gw = None

        if ctx.needs_input_grad[0]:
            gx = torch.empty_like(x)
            numel = gx.numel()
            ska_bwd_x[lambda meta: _grid(numel, meta["BS"])](go, w, gx, n, ic, h, width, ks, pad, wc, BS=1024)

        if ctx.needs_input_grad[1]:
            gw = torch.empty_like(w)
            numel = gw.numel() // w.shape[2]
            ska_bwd_w[lambda meta: _grid(numel, meta["BS"])](go, x, gw, n, wc, h, width, ic, ks, pad, BS=1024)

        return gx, gw, None, None

class SKA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return SkaFn.apply(x, w)

# ==========================================
# LKP & LSConv
# ==========================================
class LKP(nn.Module):
    def __init__(self, dim, lks=7, sks=3, groups=8):
        super().__init__()
        # 确保 groups 合理 (防止 dim=32 时，dim//groups = 4 等报错)
        # 一般 SR 模型 dim=32, 48, 64，都是 8 的倍数，没问题
        assert dim % groups == 0, f"dim {dim} must be divisible by groups {groups}"
        
        self.cv1 = nn.Conv2d(dim, dim // 2, 1)
        self.act = nn.ReLU()
        self.cv2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=lks, padding=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = nn.Conv2d(dim // 2, dim // 2, 1)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)

        self.sks = sks
        self.groups = groups
        self.dim = dim

    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w

class LSConv(nn.Module):
    """
    Large Selective Conv
    修改：去掉了内部的 +x 残差，使其成为纯粹的 Attention/Mixing 模块
    """
    def __init__(self, dim, groups=8):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=groups)
        self.ska = SKA()

    def forward(self, x):
        # 原来是 return self.ska(x, self.lkp(x)) + x
        # 现在直接返回混合特征，让外层的 MAB 去加残差
        return self.ska(x, self.lkp(x))

    
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


class NeighborhoodAttention2D(nn.Module):
    """
    Neighborhood Attention 2D Module
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
        # self.extra_repr()
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


# # ============== MAB (Multi-head Attention Block) from MAT ==============
# class StandardSelfAttention(nn.Module):
#     """Standard self-attention as fallback"""
#     def __init__(self, dim, num_head):
#         super().__init__()
#         self.num_head = num_head
#         self.head_dim = dim // num_head
#         self.scale = self.head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=True)
#         self.proj = nn.Linear(dim, dim)

#     def forward(self, x):
#         B, H, W, C = x.shape
#         qkv = self.qkv(x).reshape(B, H, W, 3, self.num_head, self.head_dim).permute(3, 0, 4, 1, 2, 5)
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)

#         out = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
#         out = self.proj(out)
#         return out


# ============== StripModule (参考 StripNet) ==============
class StripModule(nn.Module):
    """
    条形卷积模块 - StripNet StripBlock
    """
    def __init__(self, dim, k1=1, k2=47):
        super().__init__()
        self.dim = dim

        # 1. 局部特征提取 (Local Context) - 5x5 DWConv
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        # 2. 双向条形卷积
        self.conv_spatial1 = nn.Conv2d(dim, dim, kernel_size=(k1, k2),
                                        stride=1, padding=(k1 // 2, k2 // 2), groups=dim)
        self.conv_spatial2 = nn.Conv2d(dim, dim, kernel_size=(k2, k1),
                                        stride=1, padding=(k2 // 2, k1 // 2), groups=dim)

        # 3. 注意力投影
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial1(attn)
        attn = self.conv_spatial2(attn)
        attn = self.conv1(attn)
        return x * attn


# ============== StripAttention ==============
class StripAttention(nn.Module):
    """
    条形注意力模块 - 参考 StripNet Attention
    proj_1 -> GELU -> StripModule -> proj_2 + shortcut
    """
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
    """
    Middle Block using StripAttention
    与官方 StripNet Block 保持一致：内部残差 + layer_scale
    """
    def __init__(self, dim, ffn_expansion_factor=2., bias=False, k1=1, k2=47):
        super().__init__()
        self.dim = dim

        # Strip Attention
        self.norm1 = LayerNorm2d(channels=dim)
        self.strip_attn = StripAttention(dim=dim, k1=k1, k2=k2)

        # FFN
        self.norm2 = LayerNorm2d(channels=dim)
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # ⭐ layer_scale（官方 StripNet 使用 1e-2）
        layer_scale_init = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        # Strip Attention path
        x = x + self.layer_scale_1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * \
            self.strip_attn(self.norm1(x))

        # FFN path
        shortcut = x
        y = self.norm2(x)
        y = self.project_in(y)
        y1, y2 = self.dwconv(y).chunk(2, dim=1)
        y = F.gelu(y1) * y2
        y = self.project_out(y)
        x = shortcut + self.layer_scale_2.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * y

        return x



# ==========================================
# 完美的 MAB 替换
# ==========================================
class MAB(nn.Module):
    """
    Multi-head Attention Block from MAT
    使用 LSConv 完美替换 NeighborhoodAttention2D
    """
    def __init__(self, dim, num_head=None, kernel_sizes=None, dilations=None):
        super().__init__()
        self.dim = dim

        self.norm1 = LayerNorm2d(channels=dim)

        # ==========================================
        # 核心替换：使用 LSConv
        # LSConv 直接接受 BCHW 格式，不需要任何变形！
        # ==========================================
        self.attn = LSConv(dim=dim)
        print(f"[MAB] 使用 LSConv 作为 Attention (Triton Accelerated)")

        self.norm2 = LayerNorm2d(channels=dim)
        self.ffn = MSConvStar(dim=dim, mlp_ratio=2., dw_sizes=[1, 3, 5, 7])

    def forward(self, x):
        # 输入 x shape: (B, C, H, W)

        # ==========================================
        # Part 1: Self-Attention (被 LSConv 替代)
        # ==========================================
        shortcut_attn = x  

        x_norm1 = self.norm1(x)

        # LSConv 运算，输出依然是 (B, C, H, W)
        attn_out = self.attn(x_norm1)

        # 标准残差连接
        x = shortcut_attn + attn_out

        # ==========================================
        # Part 2: FFN / MSConvStar
        # ==========================================
        shortcut_ffn = x  

        x_norm2 = self.norm2(x)
        ffn_out_bchw = self.ffn(x_norm2)
        x = shortcut_ffn + ffn_out_bchw

        return x


# ============== S1_Trans Block (NoMAB1 - 仅保留MAB2) ==============
class S1_TransBlock_NoMAB1(nn.Module):
    """
    S1 Series Block (去掉MAB1):
    x = LAB(x) -> MAB(dilations=[5,3]) -> Conv -> +shortcut
    """
    def __init__(self, c, drop_out_rate=0.):
        super().__init__()
        # LAB for local aggregation
        self.lab = LAB(dim=c, local_dwconv=3, expanded_ratio=1., squeeze_factor=4)

        # MAB2: num_head=2, kernel_sizes=[7, 11], dilations=[5, 3]
        self.mab2 = MAB(dim=c, num_head=2, kernel_sizes=[7, 11], dilations=[5, 3])

        # Conv + residual (like RMAG) with zero initialization
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        shortcut = x  # 保存输入用于残差连接

        # LAB -> MAB2 -> Conv (去掉MAB1)
        x = self.lab(x)
        x = self.mab2(x)
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


#@ARCH_REGISTRY.register("symunet_pretrain_Strip_ende_lsconv")
class SymUNet_Pretrain_Strip_ende_lsconv(nn.Module):
    """
    symunet_pretrain_Strip_ende_lsconv: Middle Blk使用StripAttention (k=47)
    - 使用 LAB (Local Aggregation Block)
    - 仅使用 MAB2 (dilations=[5,3])
    - 使用 MSConvStar
    - Middle Blk使用StripAttention (k1=1, k2=47)
    """
    def __init__(self, args, conv=common.default_conv):
        super(SymUNet_Pretrain_Strip_ende_lsconv, self).__init__()

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
                S1_TransBlock_NoMAB1(c=chan, drop_out_rate=drop_out_rate) for _ in range(num)
            ]))
            self.downs.append(DownsampleDW(chan))
            chan *= 2

        # Strip kernel sizes
        strip_k1 = getattr(args, 'symunet_pretrain_Strip_ende_lsconv_k1', 1)
        strip_k2 = getattr(args, 'symunet_pretrain_Strip_ende_lsconv_k2', 47)

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
    model = SymUNet_Pretrain_S1_Trans(args)
    model.eval()
    input_lr = torch.rand(1, 3, 48, 48)
    sr = model(input_lr)
    print(f"输入LR尺寸: {input_lr.size()}")
    print(f"输出SR尺寸: {sr.size()}")
