import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from model import common
# from utils.registry import ARCH_REGISTRY

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MIN_NUM_PATCHES = 12


def make_model(args, parent=False):
    return SymUNet_Pretrain_CG(args)


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

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


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
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


# ========== CGBlock 依赖模块 (移除 UpsampleWithFlops，已用 F.interpolate 替代) ==========
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=0, stride=1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stride, padding=padding, groups=nin, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class GlobalContextExtractor(nn.Module):
    def __init__(self, c, kernel_sizes=[3, 3, 5], strides=[3, 3, 5], padding=0, bias=False):
        super(GlobalContextExtractor, self).__init__()

        self.depthwise_separable_convs = nn.ModuleList([
            depthwise_separable_conv(c, c, kernel_size, padding, stride, bias)
            for kernel_size, stride in zip(kernel_sizes, strides)
        ])

    def forward(self, x):
        outputs = []
        for conv in self.depthwise_separable_convs:
            x = F.gelu(conv(x))
            outputs.append(x)
        return outputs


# ========== CGBlock (替换 NAFBlock) ==========
class CGBlock(nn.Module):
    """
    CascadedGaze Block - 替换 NAFBlock
    使用 Global Context Extractor (GCE) 替代 Simplified Channel Attention (SCA)
    """
    def __init__(self, c, GCE_Conv=2, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.dw_channel = c * DW_Expand
        self.GCE_Conv = GCE_Conv

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1,
                                padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel,
                                kernel_size=3, padding=1, stride=1, groups=self.dw_channel,
                               bias=True)


        if self.GCE_Conv == 3:
            self.GCE = GlobalContextExtractor(c=c, kernel_sizes=[3, 3, 5], strides=[2, 3, 4])

            self.project_out = nn.Conv2d(int(self.dw_channel*2.5), c, kernel_size=1)

            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=int(self.dw_channel*2.5), out_channels=int(self.dw_channel*2.5), kernel_size=1, padding=0, stride=1,
                        groups=1, bias=True))
        else:
            self.GCE = GlobalContextExtractor(c=c, kernel_sizes=[3, 3], strides=[2, 3])

            self.project_out = nn.Conv2d(self.dw_channel*2, c, kernel_size=1)

            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=self.dw_channel*2, out_channels=self.dw_channel*2, kernel_size=1, padding=0, stride=1,
                        groups=1, bias=True))


        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        b,c,h,w = x.shape

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.gelu(x)


        # Global Context Extractor + Range fusion
        x_1 , x_2 = x.chunk(2, dim=1)
        if self.GCE_Conv == 3:
            x1, x2, x3 = self.GCE(x_1 + x_2)
            x = torch.cat([
                x,
                F.interpolate(x1, size=(h, w), mode='nearest'),
                F.interpolate(x2, size=(h, w), mode='nearest'),
                F.interpolate(x3, size=(h, w), mode='nearest')
            ], dim=1)
        else:
            x1, x2 = self.GCE(x_1 + x_2)
            x = torch.cat([
                x,
                F.interpolate(x1, size=(h, w), mode='nearest'),
                F.interpolate(x2, size=(h, w), mode='nearest')
            ], dim=1)
        x = self.project_out(x)
        x = self.sca(x) * x



        x = self.dropout1(x)
        #channel-mixing
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


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


# @ARCH_REGISTRY.register("CSymUNet_Pretrain_CG")
class SymUNet_Pretrain_CG(nn.Module):
    """
    预上采样版本SymUNet - CGBlock变种
    - 在最开始将LR图像通过bicubic插值放大到目标尺寸
    - 然后在HR空间进行特征提取和重建
    - 使用 CascadedGazeBlock (CGBlock) 替代 NAFBlock
    """
    def __init__(self, args, conv=common.default_conv):
        super(SymUNet_Pretrain_CG, self).__init__()

        self.args = args
        self.scale = args.scale[0]

        # 基本参数
        img_channel = args.n_colors
        width = getattr(args, 'symunet_pretrain_width', 32)
        middle_blk_num = getattr(args, 'symunet_pretrain_middle_blk_num', 1)
        enc_blk_nums = getattr(args, 'symunet_pretrain_enc_blk_nums', [4,6])
        dec_blk_nums = getattr(args, 'symunet_pretrain_dec_blk_nums', [6,4])

        # Transformer 参数
        ffn_expansion_factor = getattr(args, 'symunet_pretrain_ffn_expansion_factor', 2.66)
        bias = getattr(args, 'symunet_pretrain_bias', False)
        LayerNorm_type = getattr(args, 'symunet_pretrain_layer_norm_type', 'WithBias')

        # Restormer 注意力头数
        restormer_heads = getattr(args, 'symunet_pretrain_restormer_heads', [1, 2, 4])
        restormer_middle_heads = getattr(args, 'symunet_pretrain_restormer_middle_heads', 8)

        # CGBlock 参数
        DW_Expand = getattr(args, 'symunet_pretrain_dw_expand', 2)
        FFN_Expand = 4
        drop_out_rate = getattr(args, 'symunet_pretrain_dropout', 0.)
        # CGBlock 特有的 GCE_Conv 参数
        # 支持两种模式:
        # 1. 单值: 所有层使用相同的 GCE_Conv (2 或 3)
        # 2. 列表: 每层使用不同的 GCE_Conv (如 [3,3,2,2])
        gce_conv = getattr(args, 'symunet_pretrain_gce_conv', 2)

        # 预上采样层：使用bicubic插值将LR放大到HR尺寸
        self.pre_upsample = nn.Upsample(scale_factor=self.scale, mode='bicubic', align_corners=False)

        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # 确保 gce_conv 是列表，长度与 encoder 层级数匹配
        if isinstance(gce_conv, int):
            gce_conv = [gce_conv] * len(enc_blk_nums)
        elif isinstance(gce_conv, str):
            gce_conv = [int(x) for x in gce_conv.split(',')]

        chan = width
        for i, num in enumerate(enc_blk_nums):
            # 使用 CGBlock 替代 NAFBlock，每层使用对应的 GCE_Conv
            layer_gce_conv = gce_conv[i] if i < len(gce_conv) else gce_conv[-1]
            self.encoders.append(nn.Sequential(*[
                CGBlock(
                    c=chan,
                    GCE_Conv=layer_gce_conv,
                    DW_Expand=DW_Expand,
                    FFN_Expand=FFN_Expand,
                    drop_out_rate=drop_out_rate
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

        # decoder 部分的 gce_conv 需要反转，与 encoder 对称
        gce_conv_dec = gce_conv[::-1]
        for i, num in enumerate(dec_blk_nums):
            self.ups.append(Upsample(chan))
            chan //= 2

            # 使用 CGBlock 替代 NAFBlock，每层使用对应的 GCE_Conv
            layer_gce_conv = gce_conv_dec[i] if i < len(gce_conv_dec) else gce_conv_dec[-1]
            self.decoders.append(nn.Sequential(*[
                CGBlock(
                    c=chan,
                    GCE_Conv=layer_gce_conv,
                    DW_Expand=DW_Expand,
                    FFN_Expand=FFN_Expand,
                    drop_out_rate=drop_out_rate
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
    model = SymUNet_Pretrain_CG(args)
    model.eval()
    # 输入LR图像，尺寸为48x48，输出HR图像为192x192 (scale=4)
    input_lr = torch.rand(1, 3, 48, 48)
    sr = model(input_lr)
    print(f"输入LR尺寸: {input_lr.size()}")
    print(f"输出SR尺寸: {sr.size()}")
