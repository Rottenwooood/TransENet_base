import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from natten.functional import na2d_av, na2d_qk
from torch.nn.init import trunc_normal_
from model import common
from math import prod
from typing import List, Optional

#from utils.registry import ARCH_REGISTRY

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MIN_NUM_PATCHES = 12


def _lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


def make_model(args, parent=False):
    return SymUNet_Pretrain_Strip_CAM_Parallel_Add_Window(args)


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


class CPB_MLP(nn.Sequential):
    def __init__(self, in_channels, out_channels, channels=512):
        modules = [
            nn.Linear(in_channels, channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels, out_channels, bias=False),
        ]
        super().__init__(*modules)


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(
        b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size[0], window_size[1], c)


def window_reverse(windows, window_size, img_size):
    h, w = img_size
    b = int(windows.shape[0] / (h * w / window_size[0] / window_size[1]))
    x = windows.view(
        b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


def calculate_mask(input_resolution, window_size, shift_size):
    img_mask = torch.zeros((1, *input_resolution, 1))
    h_slices = (
        slice(0, -window_size[0]),
        slice(-window_size[0], -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size[1]),
        slice(-window_size[1], -shift_size),
        slice(-shift_size, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, prod(window_size))
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


def get_relative_coords_table(window_size):
    coord_h = torch.arange(-(window_size[0] - 1), window_size[0], dtype=torch.float32)
    coord_w = torch.arange(-(window_size[1] - 1), window_size[1], dtype=torch.float32)
    table = torch.stack(torch.meshgrid([coord_h, coord_w], indexing="ij")).permute(1, 2, 0)
    table = table.contiguous().unsqueeze(0)
    table[:, :, :, 0] /= max(window_size[0] - 1, 1)
    table[:, :, :, 1] /= max(window_size[1] - 1, 1)
    table *= 8
    table = torch.sign(table) * torch.log2(torch.abs(table) + 1.0) / math.log2(8)
    return table


def get_relative_position_index(window_size):
    coord_h = torch.arange(window_size[0])
    coord_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid([coord_h, coord_w], indexing="ij"))
    coords = torch.flatten(coords, 1)
    coords = coords[:, :, None] - coords[:, None, :]
    coords = coords.permute(1, 2, 0).contiguous()
    coords[:, :, 0] += window_size[0] - 1
    coords[:, :, 1] += window_size[1] - 1
    coords[:, :, 0] *= 2 * window_size[1] - 1
    return coords.sum(-1)


class AffineTransform(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        logit_scale = torch.log(10 * torch.ones((num_heads, 1, 1)))
        self.logit_scale = nn.Parameter(logit_scale, requires_grad=True)
        self.cpb_mlp = CPB_MLP(2, num_heads)

    def forward(self, attn, relative_coords_table, relative_position_index, mask):
        b_, h, n1, n2 = attn.shape
        attn = attn * torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()

        bias_table = self.cpb_mlp(relative_coords_table).view(-1, h)
        bias = bias_table[relative_position_index.view(-1)]
        bias = bias.view(n1, n2, -1).permute(2, 0, 1).contiguous()
        bias = 16 * torch.sigmoid(bias)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            n_w = mask.shape[0]
            mask = mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(b_ // n_w, n_w, h, n1, n2) + mask
            attn = attn.view(-1, h, n1, n2)

        return attn


class WindowSelfAttention2D(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4, qkv_bias=True, window_shift=False):
        super().__init__()
        self.window_size = (window_size, window_size)
        self.num_heads = self._resolve_num_heads(dim, num_heads)
        self.shift_size = self.window_size[0] // 2 if window_shift else 0
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_transform = AffineTransform(self.num_heads)
        self.attn_drop = nn.Dropout(0.0)
        self.softmax = nn.Softmax(dim=-1)

        self.register_buffer("relative_coords_table", get_relative_coords_table(self.window_size))
        self.register_buffer("relative_position_index", get_relative_position_index(self.window_size))

    @staticmethod
    def _resolve_num_heads(dim, num_heads):
        heads = max(1, min(dim, num_heads))
        while dim % heads != 0 and heads > 1:
            heads -= 1
        return heads

    def forward(self, x):
        b, c, h, w = x.shape
        if h % self.window_size[0] != 0 or w % self.window_size[1] != 0:
            raise ValueError(
                f"WindowSelfAttention2D requires feature size divisible by window_size, got {(h, w)} and {self.window_size}."
            )

        x = x.permute(0, 2, 3, 1).contiguous()
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = window_partition(x, self.window_size)
        windows = windows.view(-1, prod(self.window_size), c)

        b_, n, _ = windows.shape
        qkv = self.qkv(windows).reshape(b_, n, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        mask = None
        if self.shift_size > 0:
            mask = calculate_mask((h, w), self.window_size, self.shift_size).to(x.device)
        attn = self.attn_transform(
            attn,
            self.relative_coords_table.to(x.device),
            self.relative_position_index.to(x.device),
            mask,
        )
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        out = self.proj(out)
        out = out.view(-1, *self.window_size, c)
        out = window_reverse(out, self.window_size, (h, w))

        if self.shift_size > 0:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        return out.permute(0, 3, 1, 2).contiguous()


def split_channels(dim, num_splits):
    channels = []
    for idx in range(num_splits):
        if idx == 0:
            split_dim = dim - dim // num_splits * (num_splits - 1)
        else:
            split_dim = dim // num_splits
        channels.append(split_dim)
    return channels


# ============== Parallel Strip/Window + CAM Attention ==============
class StripCAMParallelAddWindowAttention(nn.Module):
    """
    split 后让 strip 与 window self-attention 两个空间分支并联并 concat，
    再与 full-width CAM 分支并联相加。
    """
    def __init__(self, dim, k1=1, k2=47, squeeze_factor=4, window_size=8, num_heads=4, window_shift=False):
        super().__init__()
        self.branch_dims = split_channels(dim, 2)
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()

        self.strip_branch = StripModule(self.branch_dims[0], k1=k1, k2=k2)
        self.window_branch = WindowSelfAttention2D(
            dim=self.branch_dims[1], window_size=window_size, num_heads=num_heads, window_shift=window_shift
        )
        self.cam_branch = ChannelAttention(dim=dim, squeeze_factor=squeeze_factor)

        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x = self.proj_1(x)
        x = self.activation(x)

        x_strip, x_window = torch.split(x, self.branch_dims, dim=1)
        x_strip = self.strip_branch(x_strip)
        x_window = self.window_branch(x_window)
        x_spatial = torch.cat([x_strip, x_window], dim=1)
        x_cam = self.cam_branch(x)

        x = x_spatial + x_cam
        x = self.proj_2(x)
        return x


# # ============== Channel Attention ==============
# class ChannelAttention(nn.Module):
#     def __init__(self, dim, squeeze_factor=16):
#         super().__init__()
#         self.attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(dim, dim // squeeze_factor, 1),
#             nn.SiLU(),
#             nn.Conv2d(dim // squeeze_factor, dim, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         return x * self.attention(x)

class StripCAMParallelAddWindowMiddleBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2., bias=False, k1=1, k2=47, window_size=8, num_heads=4, window_shift=False):
        super().__init__()
        self.dim = dim

        # Part 1: Split spatial branches (strip + window) and full-width CAM branch
        self.norm1 = LayerNorm2d(channels=dim)
        self.parallel_attn = StripCAMParallelAddWindowAttention(
            dim=dim, k1=k1, k2=k2, window_size=window_size, num_heads=num_heads, window_shift=window_shift
        )
        
        # Part 2: FFN
        self.norm2 = LayerNorm2d(channels=dim)
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # 只需要 2 个 LayerScale
        layer_scale_init = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        # 1. Parallel attention path
        x = x + self.layer_scale_1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * \
            self.parallel_attn(self.norm1(x))

        # 2. FFN path
        shortcut = x
        y = self.norm2(x)
        y = self.project_in(y)
        y1, y2 = self.dwconv(y).chunk(2, dim=1)
        y = F.gelu(y1) * y2
        y = self.project_out(y)
        x = shortcut + self.layer_scale_2.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * y

        return x

class MAB(nn.Module):
    """
    Multi-head Attention Block from MAT
    使用 LayerNorm2d 减少格式转换 (4次 -> 2次)
    """
    def __init__(self, dim, num_head=2, kernel_sizes=[7, 11], dilations=[1, 1]):
        super().__init__()
        self.dim = dim
        self.num_head = num_head
        self.dilations = dilations

        # 使用 LayerNorm2d 直接处理 BCHW 格式
        self.norm1 = LayerNorm2d(channels=dim)

        self.attn = NeighborhoodAttention2D(
            dim=dim,
            num_head=num_head,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            rel_pos_bias=True,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0
        )
        print(f"[MAB] 使用 NeighborhoodAttention2D (num_head={num_head}, kernel_sizes={kernel_sizes}, dilations={dilations})")

        self.norm2 = LayerNorm2d(channels=dim)
        self.ffn = MSConvStar(dim=dim, mlp_ratio=2., dw_sizes=[1, 3, 5, 7])

    def forward(self, x):
        # 输入 x shape: (B, C, H, W)

        # ==========================================
        # Part 1: Self-Attention
        # ==========================================
        shortcut_attn = x  # 保存 BCHW 格式的残差

        # 1. LayerNorm2d 直接处理 BCHW，无需转换
        x_norm1 = self.norm1(x)

        # 2. 转换到 BHWC 给 Attention 计算
        x_bhwc = x_norm1.permute(0, 2, 3, 1).contiguous()
        attn_out_bhwc = self.attn(x_bhwc)

        # 3. 转回 BCHW 并加上残差
        x = shortcut_attn + attn_out_bhwc.permute(0, 3, 1, 2).contiguous()

        # ==========================================
        # Part 2: FFN / MSConvStar
        # ==========================================
        shortcut_ffn = x  # 保存 BCHW 格式的残差

        # 1. LayerNorm2d 直接处理 BCHW，无需转换
        x_norm2 = self.norm2(x)

        # 2. MSConvStar 直接处理 BCHW 格式
        ffn_out_bchw = self.ffn(x_norm2)

        # 3. 直接在 BCHW 维度上加残差并输出
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


#@ARCH_REGISTRY.register("symunet_pretrain_strip_cam_parallel_add_window")
class SymUNet_Pretrain_Strip_CAM_Parallel_Add_Window(nn.Module):
    """
    symunet_pretrain_strip_cam_parallel_add_window:
    Middle Blk 使用:
    split(strip + window self-attention) -> concat -> + full-width CAM -> proj
    - 使用 LAB (Local Aggregation Block)
    - 仅使用 MAB2 (dilations=[5,3])
    - 使用 MSConvStar
    - Middle Blk 使用共享前投影 + split 空间并联 + CAM 并联 + add 融合 + 输出投影
    """
    def __init__(self, args, conv=common.default_conv):
        super(SymUNet_Pretrain_Strip_CAM_Parallel_Add_Window, self).__init__()

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
        strip_k1 = getattr(args, 'symunet_pretrain_strip_k1', 1)
        strip_k2 = getattr(args, 'symunet_pretrain_strip_k2', 47)
        window_size = getattr(args, 'symunet_pretrain_window_size', 8)
        window_heads = getattr(args, 'symunet_pretrain_window_heads', 4)
        window_shift = getattr(args, 'symunet_pretrain_window_shift', False)

        self.middle_blks = nn.Sequential(*[
            StripCAMParallelAddWindowMiddleBlock(
                dim=chan,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                k1=strip_k1,
                k2=strip_k2,
                window_size=window_size,
                num_heads=window_heads,
                window_shift=window_shift,
            ) for _ in range(middle_blk_num)
        ])

        for i, num in enumerate(dec_blk_nums):
            self.ups.append(UpsampleDW(chan))
            chan //= 2

            self.decoders.append(nn.Sequential(*[
                S1_TransBlock_NoMAB1(c=chan, drop_out_rate=drop_out_rate) for _ in range(num)
            ]))

        down_factor = 2 ** len(self.encoders)
        self.padder_size = down_factor * _lcm(4, window_size)

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
    model = SymUNet_Pretrain_Strip_CAM_Parallel_Add_Window(args)
    model.eval()
    input_lr = torch.rand(1, 3, 48, 48)
    sr = model(input_lr)
    print(f"输入LR尺寸: {input_lr.size()}")
    print(f"输出SR尺寸: {sr.size()}")
