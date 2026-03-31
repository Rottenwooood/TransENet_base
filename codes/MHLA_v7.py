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
        self.num_heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_heads)
        )
        
        self.cache_indices = {}
        self.cache_rel_coords = {}

        nn.init.constant_(self.mlp[-1].bias, 0.0)
        nn.init.normal_(self.mlp[-1].weight, std=0.02)

    def forward(self, num_blocks_h, num_blocks_w, device, dtype):
        key = (num_blocks_h, num_blocks_w, device)
        
        # 1. 只在分辨率改变时，预计算索引矩阵和唯一坐标
        if key not in self.cache_indices:
            # === A. 计算 M x M 的索引矩阵 ===
            y = torch.arange(num_blocks_h, device=device, dtype=torch.long)
            x = torch.arange(num_blocks_w, device=device, dtype=torch.long)
            y_coords, x_coords = torch.meshgrid(y, x, indexing='ij')
            coords = torch.stack([y_coords.flatten(), x_coords.flatten()], dim=1) #[M, 2]
            
            rel_coords = coords.unsqueeze(1) - coords.unsqueeze(0) # [M, M, 2]
            
            # 将相对坐标平移到正数区间，方便做索引
            rel_coords[:, :, 0] += num_blocks_h - 1
            rel_coords[:, :, 1] += num_blocks_w - 1
            
            # 扁平化为 1D 索引
            rel_position_index = rel_coords[:, :, 0] * (2 * num_blocks_w - 1) + rel_coords[:, :, 1]
            self.cache_indices[key] = rel_position_index # [M, M]
            
            # === B. 计算 (2H-1) x (2W-1) 的唯一相对坐标 ===
            uy = torch.arange(-(num_blocks_h - 1), num_blocks_h, device=device, dtype=dtype)
            ux = torch.arange(-(num_blocks_w - 1), num_blocks_w, device=device, dtype=dtype)
            uy_coords, ux_coords = torch.meshgrid(uy, ux, indexing='ij')
            unique_coords = torch.stack([uy_coords.flatten(), ux_coords.flatten()], dim=1) # [K, 2]
            
            # SwinV2 的连续对数坐标映射
            sign = torch.sign(unique_coords)
            log_coords = sign * torch.log1p(unique_coords.abs())
            unique_coords = log_coords / math.log(1 + 64.0)
            
            self.cache_rel_coords[key] = unique_coords

        # 读取缓存
        rel_position_index = self.cache_indices[key] # [M, M]
        unique_coords = self.cache_rel_coords[key]   # [K, 2], K 极小

        # ========================================================
        # 2. 性能核弹优化：
        # 仅仅对 K 个（比如 2209 个）唯一坐标跑 MLP，而不是 M*M 个（33万个）！
        # 避免了巨量的显存分配，前向与反向传播速度起飞。
        # ========================================================
        bias_table = self.mlp(unique_coords) # [K, num_heads]
        
        # 3. 查表：使用预存的索引矩阵，瞬间还原出 M x M 矩阵
        weights = bias_table[rel_position_index.view(-1)].view(
            num_blocks_h * num_blocks_w, num_blocks_h * num_blocks_w, self.num_heads
        ) # [M, M, num_heads]
        
        weights = weights.permute(2, 0, 1) # [num_heads, M, M]
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
        B, N, W, C = x.shape # 这里的 N 就是块的数量 (M)
        H_head, D = self.num_heads, self.head_dim

        q, k, v, lepe = self._mlp_lepe(x, pieces_h, pieces_w)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        k = torch.relu(k) + self.eps
        q = torch.relu(q) + self.eps

        # ==========================================
        # 修复 1：保留独立的 Head 维度，不要压扁
        # b n w (h d) -> b h n w d
        # ==========================================
        q, k, v = map(lambda t: rearrange(t, "b n w (h d) -> b h n w d", h=H_head, d=D), (q, k, v))
        k = k.transpose(-2, -1) # [B, H_head, N, D, W]

        # [B, H_head, N, D, D]
        kv = torch.matmul(k, v) 
        k_sum = k.sum(dim=-1, keepdim=True) #[B, H_head, N, D, 1]
        
        # 获取动态距离权重[H_head, N, N]
        dist_weight = self.piece_attn(pieces_h, pieces_w, x.device, x.dtype)
        
        # ==========================================
        # 修复 2：极速矩阵乘法，彻底抛弃 einsum，完美匹配多头维度
        # ==========================================
        # 扩展权重以匹配 Batch 维度: [1, H_head, N, N]
        dist_w_broadcast = dist_weight.unsqueeze(0)
        
        # 展平 DxD 维度:[B, H_head, N, D*D]
        kv_flat = kv.reshape(B, H_head, N, D * D)
        
        # [1, H_head, N, N] @[B, H_head, N, D*D] -> [B, H_head, N, D*D]
        kv_mixed_flat = torch.matmul(dist_w_broadcast, kv_flat)
        
        # 还原形状: [B, H_head, N, D, D]
        kv_mixed = kv_mixed_flat.reshape(B, H_head, N, D, D)

        # Normalizer 同理处理
        q_k_sum = torch.matmul(q, k_sum) #[B, H_head, N, W, 1]
        q_k_sum_flat = q_k_sum.reshape(B, H_head, N, W)
        norm_flat = torch.matmul(dist_w_broadcast, q_k_sum_flat)
        normalizer = norm_flat.reshape(B, H_head, N, W, 1) + self.eps
        # ==========================================

        # 输出计算: [B, H_head, N, W, D]
        out = torch.matmul(q, kv_mixed) / normalizer
        
        # 重新压扁回 [B, N, W, C]
        out = rearrange(out, "b h n w d -> b n w (h d)")
        out = out + lepe

        return self.to_out(out)