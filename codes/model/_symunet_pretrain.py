import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from model import common

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MIN_NUM_PATCHES = 12


from utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register("SymUNet_Pretrain")
def make_model(args, parent=False):
    # Adapter: Extract args here to keep the class clean
    width = getattr(args, 'symunet_pretrain_width', 64)
    enc_blk_nums = getattr(args, 'symunet_pretrain_enc_blk_nums', [2, 2, 2])
    dec_blk_nums = getattr(args, 'symunet_pretrain_dec_blk_nums', [2, 2, 2])
    
    # Check if we are using the 'restormer' style heads config
    # Note: older versions might not have this, default to [1,2,4]
    heads = getattr(args, 'symunet_pretrain_restormer_heads', [1, 2, 4])
    middle_heads = getattr(args, 'symunet_pretrain_restormer_middle_heads', 8)
    
    ffn_expansion = getattr(args, 'symunet_pretrain_ffn_expansion_factor', 2.66)
    bias = getattr(args, 'symunet_pretrain_bias', False)
    ln_type = getattr(args, 'symunet_pretrain_layer_norm_type', 'WithBias')
    middle_blk_num = getattr(args, 'symunet_pretrain_middle_blk_num', 1)
    
    return SymUNet_Pretrain(
        img_channel=args.n_colors,
        width=width,
        enc_blk_nums=enc_blk_nums, 
        dec_blk_nums=dec_blk_nums,
        middle_blk_num=middle_blk_num,
        heads=heads,
        middle_heads=middle_heads,
        ffn_expansion=ffn_expansion,
        bias=bias,
        ln_type=ln_type,
        scale=args.scale[0]
    )


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


from utils.registry import ARCH_REGISTRY


class SymUNet_Pretrain(nn.Module):
    """
    预上采样版本SymUNet
    - 在最开始将LR图像通过bicubic插值放大到目标尺寸
    - 然后在HR空间进行特征提取和重建
    """
    def __init__(self, 
                 img_channel=3,
                 width=64,
                 enc_blk_nums=[2, 2, 2],
                 dec_blk_nums=[2, 2, 2],
                 middle_blk_num=1,
                 heads=[1, 2, 4],
                 middle_heads=8,
                 ffn_expansion=2.66,
                 bias=False,
                 ln_type='WithBias',
                 scale=4,
                 conv=common.default_conv):
        super(SymUNet_Pretrain, self).__init__()

        self.scale = scale

        # Use passed arguments instead of global args
        img_channel = img_channel
        width = width
        middle_blk_num = middle_blk_num
        enc_blk_nums = enc_blk_nums
        dec_blk_nums = dec_blk_nums

        # Transformer arguments
        ffn_expansion_factor = ffn_expansion
        bias = bias
        LayerNorm_type = ln_type

        # Restormer heads
        restormer_heads = heads
        restormer_middle_heads = middle_heads

        # 预上采样层：使用bicubic插值将LR放大到HR尺寸
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
                TransformerBlock(
                    dim=chan,
                    num_heads=restormer_heads[i],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type
                ) for _ in range(num)
            ]))
            self.downs.append(Downsample(chan))
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
            self.ups.append(Upsample(chan))
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

        self.padder_size = (2 ** len(self.encoders)) * 4

    def forward(self, inp):
        B, C, H, W = inp.shape

        # 预上采样：将LR图像放大到HR尺寸
        inp_upsampled = self.pre_upsample(inp)

        # 检查图像尺寸是否适配网络
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
        # 残差连接：加上预上采样的输入
        x = x + inp_padded

        # 裁剪到原始HR尺寸
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
    from option import args
    # Mock args for testing
    class MockArgs:
        n_colors = 3
        scale = [4]
    
    # Direct instantiation (Best Practice)
    model = SymUNet_Pretrain(width=32, scale=4)
    # OR via Adapter
    # model = make_model(args)
    model.eval()
    # 输入LR图像，尺寸为48x48，输出HR图像为192x192 (scale=4)
    input_lr = torch.rand(1, 3, 48, 48)
    sr = model(input_lr)
    print(f"输入LR尺寸: {input_lr.size()}")
    print(f"输出SR尺寸: {sr.size()}")