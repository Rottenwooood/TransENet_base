import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
import math

class DynamicLearnableBlockMixing(nn.Module):
    """
    专为超分(SR)等动态分辨率任务设计的 MHLA Multi-Head Mixing 模块。
    完美复刻官方初始化，同时保持矩阵动态可学习。
    """
    def __init__(self, transform="cos", local_thres=1.5, exp_sigma=3, hidden_dim=16):
        super().__init__()
        self.transform = transform
        self.local_thres = local_thres
        self.exp_sigma = exp_sigma

        # 使用轻量级 MLP 根据相对距离动态生成“学习残差”
        # 这样无论推理时 M 变得多大，网络都能推断出对应的混合权重
        self.learned_residual = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 【关键】将最后一层初始化为 0。
        # 这样训练初始阶段，MLP输出全为0，整个权重矩阵完全等于官方基于物理距离的初始化！
        nn.init.zeros_(self.learned_residual[-1].weight)
        nn.init.zeros_(self.learned_residual[-1].bias)

    def forward(self, num_blocks_h, num_blocks_w, device, dtype):
        total_blocks = num_blocks_h * num_blocks_w
        
        # 1. 动态生成块网格坐标与距离矩阵 (完全对应官方 _compute_block_distances)
        y, x = torch.meshgrid(torch.arange(num_blocks_h), torch.arange(num_blocks_w), indexing='ij')
        centers = torch.stack([x.flatten(), y.flatten()], dim=1).to(device=device, dtype=torch.float32)
        # centers 加上 0.5 对齐块中心（与官方一致）
        centers = centers + 0.5 
        
        #[total_blocks, total_blocks]
        dist_matrix = torch.norm(centers.unsqueeze(1) - centers.unsqueeze(0), p=2, dim=-1)

        # 2. 应用官方的距离变换逻辑 (_apply_transform)
        max_dist = dist_matrix.max() + 1e-6
        if self.transform == "linear":
            base_mat = 1.0 - (dist_matrix / max_dist)
        elif self.transform == "cos":
            normalized_dist = dist_matrix / max_dist * math.pi / 4
            base_mat = torch.cos(normalized_dist)
        elif self.transform == "exp":
            base_mat = torch.exp(-dist_matrix / self.exp_sigma)
        elif self.transform == "local":
            base_mat = (dist_matrix <= self.local_thres).float()
        else:
            base_mat = dist_matrix # fallback

        # 官方的初步归一化
        base_mat = base_mat / (base_mat.sum(dim=0, keepdim=True) + 1e-6)
        base_mat = base_mat.to(dtype)

        # 3. 加上可学习的动态调整残差
        # 将距离归一化后输入 MLP，学习块与块之间的语义自适应偏置
        norm_dist_feat = (dist_matrix / max_dist).unsqueeze(-1).to(dtype)
        # res:[M, M]
        res = self.learned_residual(norm_dist_feat).squeeze(-1)
        
        # base_mat 提供了局部感知的强先验，res 提供了可学习的灵活性
        weight_matrix = base_mat + res
        
        # 4. 遵循论文的严格要求：限制在 (0,1) 之间，并且按行重新归一化
        weight_matrix = torch.clamp(weight_matrix, min=0.0, max=1.0)
        weight_matrix = weight_matrix / (weight_matrix.sum(dim=0, keepdim=True) + 1e-6)
        
        # 输出的形状为 [M, M]，表示 第 i 个块 对 第 j 个块 的注意力/混合权重
        return weight_matrix

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
        
        self.piece_attn = DynamicLearnableBlockMixing(transform=transform)
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
        x = self.norm(x)
        B, N, W, C = x.shape
        H_head, D = self.num_heads, self.head_dim

        q, k, v, lepe = self._mlp_lepe(x, pieces_h, pieces_w)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # 线性注意力的正值激活
        k = torch.relu(k) + self.eps
        q = torch.relu(q) + self.eps

        q, k, v = map(lambda t: rearrange(t, "b n w (h d) -> (b h) n w d", h=H_head, d=D), (q, k, v))
        k = k.transpose(-2, -1)

        # 核心：计算每个局部块的 KV Summary
        # kv:[B*H, num_pieces, D, D]
        kv = torch.matmul(k, v) 
        k_sum = k.sum(dim=-1, keepdim=True) #[B*H, num_pieces, D, 1]
        
        # 🌟 获取动态且可学习的距离权重 [num_pieces, num_pieces]
        # dist_weight[i, j] 表示查询块 i 对 键值块 j 的注意力权重
        dist_weight = self.piece_attn(pieces_h, pieces_w, x.device, x.dtype)
        
        # 🌟 执行块间信息聚合（等价于官方的 1x1 Conv）
        # 'i j, b j x y -> b i x y' 完全对应了 Conv2d 的跨通道混合！
        kv_mixed = torch.einsum('i j, b j x y -> b i x y', dist_weight, kv)
        normalizer_mixed = torch.einsum('i j, b j x y -> b i x y', dist_weight, torch.matmul(q, k_sum)) + self.eps

        # 计算最终输出
        out = torch.matmul(q, kv_mixed) / normalizer_mixed
        out = rearrange(out, "(b h) n w d -> b n w (h d)", b=B, h=self.num_heads)
        
        # 加上局部位置编码(CPE)
        out = out + lepe

        return self.to_out(out)