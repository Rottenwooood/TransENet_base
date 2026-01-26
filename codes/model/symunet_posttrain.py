import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from model import common

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MIN_NUM_PATCHES = 12


def make_model(args, parent=False):
    return SymUNet_Posttrain(args)


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
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return rearrange(self.body(rearrange(x, 'b c h w -> b (h w) c')), 'b (h w) c -> b c h w', h=h, w=w)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


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


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class SymUNet_Posttrain(nn.Module):
    """
    后上采样版本SymUNet (简化版)
    - 使用与Pretrain相同的U-Net结构
    - 在最后添加PixelShuffle进行上采样
    """
    def __init__(self, args, conv=common.default_conv):
        super(SymUNet_Posttrain, self).__init__()

        self.args = args
        self.scale = args.scale[0]

        # 基本参数
        img_channel = args.n_colors
        width = getattr(args, 'symunet_posttrain_width', 64)
        middle_blk_num = getattr(args, 'symunet_posttrain_middle_blk_num', 1)
        enc_blk_nums = getattr(args, 'symunet_posttrain_enc_blk_nums', [2, 2, 2])
        dec_blk_nums = getattr(args, 'symunet_posttrain_dec_blk_nums', [2, 2, 2])

        # Transformer 参数
        ffn_expansion_factor = getattr(args, 'symunet_posttrain_ffn_expansion_factor', 2.66)
        bias = getattr(args, 'symunet_posttrain_bias', False)
        LayerNorm_type = getattr(args, 'symunet_posttrain_layer_norm_type', 'WithBias')

        # Restormer 注意力头数
        restormer_heads = getattr(args, 'symunet_posttrain_restormer_heads', [1, 2, 4])
        restormer_middle_heads = getattr(args, 'symunet_posttrain_restormer_middle_heads', 8)

        # 输入层
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for i, num in enumerate(enc_blk_nums):
            self.encoders.append(nn.Sequential(*[
                TransformerBlock(
                    dim=chan,
                    num_heads=restormer_heads[i],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type
                ) for _ in range(num)
            ]))
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
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

        # 解码器
        for i, num in enumerate(dec_blk_nums):
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2

            stage_idx = len(dec_blk_nums) - 1 - i
            current_restormer_heads = restormer_heads[stage_idx]

            self.decoders.append(nn.Sequential(*[
                TransformerBlock(
                    dim=chan,
                    num_heads=current_restormer_heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type
                ) for _ in range(num)
            ]))

        # 最终上采样层：使用PixelShuffle
        self.final_upsample = nn.Sequential(
            nn.Conv2d(chan, img_channel * (self.scale * self.scale), 3, 1, 1, bias=True),
            nn.PixelShuffle(self.scale)
        )

        # LR图像的bicubic插值上采样（作为残差连接）
        self.lr_upsample = nn.Upsample(scale_factor=self.scale, mode='bicubic', align_corners=False)

        self.padder_size = (2 ** len(self.encoders)) * 4

    def forward(self, inp):
        B, C, H, W = inp.shape

        # 检查图像尺寸
        inp_padded = self.check_image_size(inp)
        x = self.intro(inp_padded)

        encs = []

        # 编码器
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        # 解码器
        for decoder_blocks, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder_blocks(x)

        # 最终上采样到HR尺寸
        x = self.final_upsample(x)

        # 残差连接：加上bicubic插值的LR图像
        inp_upsampled = self.lr_upsample(inp_padded)
        x = x + inp_upsampled

        # 裁剪到目标尺寸
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
    model = SymUNet_Posttrain(args)
    model.eval()
    # 输入LR图像，尺寸为48x48，输出HR图像为192x192 (scale=4)
    input_lr = torch.rand(1, 3, 48, 48)
    sr = model(input_lr)
    print(f"输入LR尺寸: {input_lr.size()}")
    print(f"输出SR尺寸: {sr.size()}")