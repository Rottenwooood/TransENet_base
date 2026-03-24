import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
import math

class DynamicBlockDistanceWeight(nn.Module):
    """
    动态生成块间距离权重，支持任意分辨率带来的不同 Block 数量。
    """
    def __init__(self, transform="cos", local_thres=1.5, exp_sigma=3):
        super().__init__()
        self.transform = transform
        self.local_thres = local_thres
        self.exp_sigma = exp_sigma

    def forward(self, num_blocks_h, num_blocks_w, device, dtype):
        total_blocks = num_blocks_h * num_blocks_w
        
        # 1. 生成网格坐标
        y, x = torch.meshgrid(torch.arange(num_blocks_h), torch.arange(num_blocks_w), indexing='ij')
        centers = torch.stack([x.flatten(), y.flatten()], dim=1).to(device=device, dtype=dtype) # [total_blocks, 2]
        
        # 2. 计算欧氏距离矩阵
        # [total_blocks, 1, 2] -[1, total_blocks, 2] -> [total_blocks, total_blocks]
        dist_matrix = torch.norm(centers.unsqueeze(1) - centers.unsqueeze(0), p=2, dim=-1)

        # 3. 应用转换函数
        if self.transform == "linear":
            max_dist = dist_matrix.max() + 1e-6
            mat = 1.0 - (dist_matrix / max_dist)
            return mat / mat.sum(dim=0, keepdim=True)
            
        elif self.transform == "cos":
            max_dist = dist_matrix.max() + 1e-6
            normalized_dist = dist_matrix / max_dist * math.pi / 4
            mat = torch.cos(normalized_dist)
            return mat / mat.sum(dim=0, keepdim=True)
            
        elif self.transform == "exp":
            mat = torch.exp(-dist_matrix / self.exp_sigma)
            return mat / mat.sum(dim=0, keepdim=True)
            
        elif self.transform == "local":
            mat = (dist_matrix <= self.local_thres).float()
            return mat / (mat.sum(dim=0, keepdim=True) + 1e-6)
            
        return dist_matrix # Fallback

class MHLA_Normed_Torch_Dynamic(nn.Module):
    def __init__(self, dim, heads=4, dim_head=None, dropout=0.1, qk_norm=False, transform="cos", window_size=49):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim_head if dim_head else dim // heads
        inner_dim = self.head_dim * heads
        
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.q_norm = nn.RMSNorm(dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(dim) if qk_norm else nn.Identity()
        
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.window_size = window_size
        self.window_len = int(window_size ** 0.5)
        
        self.piece_attn = DynamicBlockDistanceWeight(transform=transform)
        self.eps = 1e-6
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def _mlp_lepe(self, x, pieces_h, pieces_w):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # 动态重排进行 LePE 计算
        v_lepe = rearrange(v, 'b (ph pw) (wh ww) d -> b d (ph wh) (pw ww)', 
                           ph=pieces_h, pw=pieces_w, wh=self.window_len, ww=self.window_len)
        lepe = self.lepe(v_lepe)
        lepe = rearrange(lepe, 'b d (ph wh) (pw ww) -> b (ph pw) (wh ww) d', 
                         ph=pieces_h, pw=pieces_w, wh=self.window_len, ww=self.window_len)
        return q, k, v, lepe

    def forward(self, x, pieces_h, pieces_w):
        # x shape:[B, num_pieces, window_size, C]
        x = self.norm(x)
        B, N, W, C = x.shape
        H_head, D = self.num_heads, self.head_dim

        q, k, v, lepe = self._mlp_lepe(x, pieces_h, pieces_w)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        k = torch.relu(k) + self.eps
        q = torch.relu(q) + self.eps

        q, k, v = map(lambda t: rearrange(t, "b n w (h d) -> (b h) n w d", h=H_head, d=D), (q, k, v))
        k = k.transpose(-2, -1)

        #[B*H, num_pieces, D, D]
        kv = torch.matmul(k, v) 
        k_sum = k.sum(dim=-1, keepdim=True) #[B*H, num_pieces, D, 1]
        
        # 获取动态距离权重 [num_pieces, num_pieces]
        dist_weight = self.piece_attn(pieces_h, pieces_w, x.device, x.dtype)
        
        # 替代原本的 1x1 Conv，进行动态块间信息聚合
        # dist_weight: [M, M], kv:[B*H, M, D, D] -> [B*H, M, D, D]
        kv_mixed = torch.einsum('m n, b n i j -> b m i j', dist_weight, kv)
        normalizer = torch.einsum('m n, b n i j -> b m i j', dist_weight, torch.matmul(q, k_sum)) + self.eps

        out = torch.matmul(q, kv_mixed) / normalizer
        out = rearrange(out, "(b h) n w d -> b n w (h d)", b=B, h=self.num_heads)
        out = out + lepe

        return self.to_out(out)