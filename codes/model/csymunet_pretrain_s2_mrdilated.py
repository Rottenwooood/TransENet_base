import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from model import common
#from utils.registry import ARCH_REGISTRY

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MIN_NUM_PATCHES = 12


def make_model(args, parent=False):
    return SymUNet_Pretrain_S2_MRDilated(args)


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


# ============== LAB ==============
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


# ============== MSDWConv ==============
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


# ============== MSConvStar ==============
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
        # x is (B, C, H, W) format
        x = self.fc1(x)
        x = x + self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = self.act(x1) * x2
        x = self.fc2(x)
        return x


# ============== MRDilated DWConv ==============
class MRDilatedDWConv(nn.Module):
    """
    Multi-Range Dilated DWConv:
    - 输入通道切 3 组（处理不能被 3 整除的情况，余数给最后一组）
    - 组1: 5x5 DWConv(d=1, p=2)
    - 组2: 5x5 DWConv(d=2, p=4)
    - 组3: 5x5 DWConv(d=3, p=6)
    - 输出后 Concat，过 1x1 Conv 融合
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Split channels into 3 groups
        channels_per_group = dim // 3
        remainder = dim % 3

        # Calculate channel splits
        self.channel_splits = []
        for i in range(3):
            if i == 2:
                # Last group gets the remainder
                ch = channels_per_group + remainder
            else:
                ch = channels_per_group
            self.channel_splits.append(ch)

        # Group 1: 5x5 DWConv (dilation=1, padding=2)
        self.dwconv1 = nn.Conv2d(self.channel_splits[0], self.channel_splits[0],
                                  kernel_size=5, padding=2, groups=self.channel_splits[0], dilation=1)

        # Group 2: 5x5 DWConv (dilation=2, padding=4)
        self.dwconv2 = nn.Conv2d(self.channel_splits[1], self.channel_splits[1],
                                  kernel_size=5, padding=4, groups=self.channel_splits[1], dilation=2)

        # Group 3: 5x5 DWConv (dilation=3, padding=6)
        self.dwconv3 = nn.Conv2d(self.channel_splits[2], self.channel_splits[2],
                                  kernel_size=5, padding=6, groups=self.channel_splits[2], dilation=3)

        # Fusion 1x1 Conv
        self.fusion = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x_split = torch.split(x, self.channel_splits, dim=1)

        out1 = self.dwconv1(x_split[0])
        out2 = self.dwconv2(x_split[1])
        out3 = self.dwconv3(x_split[2])

        out = torch.cat([out1, out2, out3], dim=1)
        out = self.fusion(out)

        return out


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


# ============== S2_MRDilated Block (并联门控架构) ==============
class S2_MRDilatedBlock(nn.Module):
    """
    S2 Series Block (并联门控架构):
    Token Mixing: x -> LayerNorm -> 1x1 Conv升维2C -> 劈两半 -> 分支A:3x3DWConv, 分支B:MRDilatedDWConv -> 门控融合 -> SCA -> 降维 -> 残差
    Channel Mixing: MSConvStar
    """
    def __init__(self, c, drop_out_rate=0.):
        super().__init__()
        self.norm1 = LayerNorm2d(c)
        self.expand_conv = nn.Conv2d(c, c * 2, 1)
        # 分支A: 3x3 DWConv
        self.branch_a = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c)
        # 分支B: MRDilatedDWConv (组1:5x5 d=1, 组2:5x5 d=2, 组3:5x5 d=3)
        self.branch_b = MRDilatedDWConv(dim=c)
        # SCA (Spatial-Channels Attention)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c // 2, out_channels=c // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.reduce_conv = nn.Conv2d(c, c, 1)
        self.norm2 = LayerNorm2d(c)
        self.msconvstar = MSConvStar(dim=c, mlp_ratio=2., dw_sizes=[1, 3, 5, 7])
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.expand_conv(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.branch_a(x1)
        x2 = self.branch_b(x2)
        out = x1 * x2  # 门控融合
        out = out * self.sca(out)
        out = self.reduce_conv(out)
        x = identity + out * self.beta
        x = x + self.msconvstar(self.norm2(x)) * self.gamma
        return x


# ============== FeedForward & Attention ==============
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


#@ARCH_REGISTRY.register("CSymUNet_Pretrain_S2_MRDilated")
class SymUNet_Pretrain_S2_MRDilated(nn.Module):
    """S2_MRDilated: 并联门控架构 (Token Mixing: 3x3DWConv + MRDilatedDWConv, Channel Mixing: MSConvStar)"""
    def __init__(self, args, conv=common.default_conv):
        super(SymUNet_Pretrain_S2_MRDilated, self).__init__()

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
                S2_MRDilatedBlock(c=chan, drop_out_rate=drop_out_rate) for _ in range(num)
            ]))
            self.downs.append(DownsampleDW(chan))
            chan *= 2

        self.middle_blks = nn.Sequential(*[
            TransformerBlock(
                dim=chan,
                num_heads=restormer_middle_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(middle_blk_num)
        ])

        for i, num in enumerate(dec_blk_nums):
            self.ups.append(UpsampleDW(chan))
            chan //= 2

            self.decoders.append(nn.Sequential(*[
                S2_MRDilatedBlock(c=chan, drop_out_rate=drop_out_rate) for _ in range(num)
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
    model = SymUNet_Pretrain_S2_MRDilated(args)
    model.eval()
    input_lr = torch.rand(1, 3, 48, 48)
    sr = model(input_lr)
    print(f"输入LR尺寸: {input_lr.size()}")
    print(f"输出SR尺寸: {sr.size()}")
