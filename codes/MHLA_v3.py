import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
import math

class MultiScaleDistanceWeight(nn.Module):
    """
    多尺度动态距离矩阵：为不同的 Attention Head 生成不同衰减率的感受野
    """
    def __init__(self, sigmas=[1.5, 7.0]):
        super().__init__()
        # sigma 越小，越关注局部；sigma 越大，感受野越全局
        self.sigmas = sigmas

    def forward(self, num_blocks_h, num_blocks_w, device, dtype):
        y, x = torch.meshgrid(torch.arange(num_blocks_h), torch.arange(num_blocks_w), indexing='ij')
        centers = torch.stack([x.flatten(), y.flatten()], dim=1).to(device=device, dtype=dtype)
        
        # dist_matrix: [M, M], M 是块的总数
        dist_matrix = torch.norm(centers.unsqueeze(1) - centers.unsqueeze(0), p=2, dim=-1)

        weights =[]
        for sig in self.sigmas:
            # 使用指数衰减模拟感受野
            mat = torch.exp(-dist_matrix / sig)
            mat = mat / (mat.sum(dim=0, keepdim=True) + 1e-6)
            weights.append(mat)
            
        # 返回形状: [Heads, M, M]
        return torch.stack(weights, dim=0)

class MHLA_Normed_Torch_Dynamic(nn.Module):
    def __init__(self, dim, heads=2, dropout=0., window_size=64):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        inner_dim = self.head_dim * heads
        
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.window_size = window_size
        self.window_len = int(window_size ** 0.5)
        
        # 对应你的 num_head=2，我们设置两个截然不同的感受野尺度
        # 1.5 模拟局部强相关 (类似 baseline 的空洞卷积)
        # 7.0 提供真正的全局图像理解
        self.piece_attn = MultiScaleDistanceWeight(sigmas=[1.5, 7.0])
        
        self.eps = 1e-6
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def _mlp_lepe(self, x, pieces_h, pieces_w):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        v_lepe = rearrange(v, 'b (ph pw) (wh ww) d -> b d (ph wh) (pw ww)', 
                           ph=pieces_h, pw=pieces_w, wh=self.window_len, ww=self.window_len)
        lepe = self.lepe(v_lepe)
        lepe = rearrange(lepe, 'b d (ph wh) (pw ww) -> b (ph pw) (wh ww) d', 
                         ph=pieces_h, pw=pieces_w, wh=self.window_len, ww=self.window_len)
        return q, k, v, lepe

    def forward(self, x, pieces_h, pieces_w):
        x = self.norm(x)
        B, M, W, C = x.shape  # M = num_pieces
        H_head, D = self.num_heads, self.head_dim

        q, k, v, lepe = self._mlp_lepe(x, pieces_h, pieces_w)
        
        k = torch.relu(k) + self.eps
        q = torch.relu(q) + self.eps

        # 核心修改：保留 Head 维度，以便不同 Head 应用不同尺度的距离权重
        # 转换后形状均为 [B, H_head, M, W, D]
        q, k, v = map(lambda t: rearrange(t, "b m w (h d) -> b h m w d", h=H_head, d=D), (q, k, v))
        k = k.transpose(-2, -1) #[B, H_head, M, D, W]

        # 块内 KV 摘要提取 -> [B, H_head, M, D, D]
        kv = torch.matmul(k, v) 
        k_sum = k.sum(dim=-1, keepdim=True) # [B, H_head, M, D, 1]
        
        # 获取多尺度权重矩阵 -> [H_head, M, M]
        dist_weight = self.piece_attn(pieces_h, pieces_w, x.device, x.dtype)
        
        # 爱因斯坦求和：h维度对齐，每个注意力头采用自己的感受野权重矩阵！
        kv_mixed = torch.einsum('h m n, b h n i j -> b h m i j', dist_weight, kv)
        normalizer = torch.einsum('h m n, b h n i j -> b h m i j', dist_weight, torch.matmul(q, k_sum)) + self.eps

        # 输出计算
        out = torch.matmul(q, kv_mixed) / normalizer # [B, H_head, M, W, D]
        out = rearrange(out, "b h m w d -> b m w (h d)")
        
        return self.to_out(out + lepe)