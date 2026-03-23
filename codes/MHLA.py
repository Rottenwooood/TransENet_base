import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
import math

class DynamicBlockDistanceWeight(nn.Module):
    """动态生成块间距离权重，完美适应 192x192 和 96x96 的不同块数量"""
    def __init__(self, transform="exp", exp_sigma=3.0):
        super().__init__()
        self.transform = transform
        self.exp_sigma = exp_sigma

    def forward(self, num_blocks_h, num_blocks_w, device, dtype):
        # 实时生成距离网格
        y, x = torch.meshgrid(torch.arange(num_blocks_h), torch.arange(num_blocks_w), indexing='ij')
        centers = torch.stack([x.flatten(), y.flatten()], dim=1).to(device=device, dtype=dtype)
        dist_matrix = torch.norm(centers.unsqueeze(1) - centers.unsqueeze(0), p=2, dim=-1)

        # 超分推荐使用 exp 衰减
        if self.transform == "exp":
            mat = torch.exp(-dist_matrix / self.exp_sigma)
            return mat / (mat.sum(dim=0, keepdim=True) + 1e-6)
        elif self.transform == "cos":
            max_dist = dist_matrix.max() + 1e-6
            mat = torch.cos(dist_matrix / max_dist * math.pi / 4)
            return mat / (mat.sum(dim=0, keepdim=True) + 1e-6)
        else:
            max_dist = dist_matrix.max() + 1e-6
            mat = 1.0 - (dist_matrix / max_dist)
            return mat / (mat.sum(dim=0, keepdim=True) + 1e-6)

class MHLA_Normed_Torch_Dynamic(nn.Module):
    def __init__(self, dim, heads=2, dropout=0., transform="exp", window_size=64):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        inner_dim = self.head_dim * heads
        
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.window_size = window_size
        self.window_len = int(window_size ** 0.5)
        
        self.piece_attn = DynamicBlockDistanceWeight(transform=transform)
        self.eps = 1e-6
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def _mlp_lepe(self, x, pieces_h, pieces_w):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # 动态计算局部增强位置编码 (LePE)
        v_lepe = rearrange(v, 'b (ph pw) (wh ww) d -> b d (ph wh) (pw ww)', 
                           ph=pieces_h, pw=pieces_w, wh=self.window_len, ww=self.window_len)
        lepe = self.lepe(v_lepe)
        lepe = rearrange(lepe, 'b d (ph wh) (pw ww) -> b (ph pw) (wh ww) d', 
                         ph=pieces_h, pw=pieces_w, wh=self.window_len, ww=self.window_len)
        return q, k, v, lepe

    def forward(self, x, pieces_h, pieces_w):
        x = self.norm(x)
        B, N, W, C = x.shape
        H_head, D = self.num_heads, self.head_dim

        q, k, v, lepe = self._mlp_lepe(x, pieces_h, pieces_w)
        
        k = torch.relu(k) + self.eps
        q = torch.relu(q) + self.eps

        q, k, v = map(lambda t: rearrange(t, "b n w (h d) -> (b h) n w d", h=H_head, d=D), (q, k, v))
        k = k.transpose(-2, -1)

        kv = torch.matmul(k, v) 
        k_sum = k.sum(dim=-1, keepdim=True)
        
        # 获取动态权重矩阵
        dist_weight = self.piece_attn(pieces_h, pieces_w, x.device, x.dtype)
        
        # 块间信息融合
        kv_mixed = torch.einsum('m n, b n i j -> b m i j', dist_weight, kv)
        normalizer = torch.einsum('m n, b n i j -> b m i j', dist_weight, torch.matmul(q, k_sum)) + self.eps

        out = torch.matmul(q, kv_mixed) / normalizer
        out = rearrange(out, "(b h) n w d -> b n w (h d)", b=B, h=self.num_heads)
        
        return self.to_out(out + lepe)
