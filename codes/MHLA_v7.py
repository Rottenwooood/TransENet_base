import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
import math

class DynamicBlockDistanceWeight(nn.Module):
    """
    动态生成块间距离权重，支持任意分辨率带来的不同 Block 数量。
    """
    def __init__(self, num_heads):
        super().__init__()
        # 输入：两个块在二维网格上的相对坐标 (dy, dx)
        # 输出：为每个 Attention Head 生成不同的特征混合权重
        self.mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_heads)
        )
        
        # 缓存固定的坐标网格，避免每步重算（注意：只缓存坐标，不缓存权重！）
        self.cache_coords = {}

        # 智能初始化：让它初始阶段类似高斯衰减，之后任由网络自己魔改
        nn.init.constant_(self.mlp[-1].bias, 0.0)
        nn.init.normal_(self.mlp[-1].weight, std=0.02)

    def forward(self, num_blocks_h, num_blocks_w, device, dtype):
        key = (num_blocks_h, num_blocks_w, device)
        
        # 1. 只在分辨率改变时，重新生成相对坐标矩阵
        if key not in self.cache_coords:
            y = torch.arange(num_blocks_h, device=device, dtype=dtype)
            x = torch.arange(num_blocks_w, device=device, dtype=dtype)
            y_coords, x_coords = torch.meshgrid(y, x, indexing='ij')
            coords = torch.stack([y_coords.flatten(), x_coords.flatten()], dim=1)
            
            # 计算两两之间的绝对相对坐标 (dy, dx)
            rel_coords = coords.unsqueeze(1) - coords.unsqueeze(0)
            
            # ========================================================
            # 核心修复：SwinV2 的 Log-Spaced Coordinates
            # 不要除以当前的宽高！用 log 函数把距离映射到一个平滑的空间
            # 这样无论是训练的 24x24 还是测试的 32x32，相邻块的输入永远是一样的！
            # ========================================================
            sign = torch.sign(rel_coords)
            # log(1 + |x|) 既保留了相对距离的物理一致性，又防止了远距离数值过大
            log_coords = sign * torch.log1p(rel_coords.abs())
            
            # 除以一个常数 (比如 log(1 + 64)) 将输入大致缩放到 [-1, 1] 以利于 MLP 训练
            # 这个常数必须是写死的，绝对不能随图像大小变化！
            rel_coords = log_coords / math.log(1 + 64.0)
            
            self.cache_coords[key] = rel_coords
            
        rel_coords = self.cache_coords[key] # [M, M, 2]
        
        # ========================================================
        # 2. 核心：用 MLP 根据相对位置【动态且带梯度地】算出权重！
        # weights 形状:[M, M, num_heads]
        # ========================================================
        weights = self.mlp(rel_coords) 
        
        # 转换到 [num_heads, M, M]
        weights = weights.permute(2, 0, 1)
        
        # 原论文要求：必须非负且归一化 (Softmax 保证每行和为1)
        # 对最后一个维度 (被聚合的源块) 做 Softmax
        weights = torch.softmax(weights, dim=-1)
        
        return weights

class MHLA_Normed_Torch_Dynamic(nn.Module):
    def __init__(self, dim, heads=4, dim_head=None, dropout=0.1, qk_norm=True, transform="cos", window_size=49):
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
        
        self.piece_attn = DynamicBlockDistanceWeight(num_heads=heads)
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