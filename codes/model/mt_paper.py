import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from natten import NeighborhoodAttention2D as NeighborhoodAttention
from timm.models.layers import to_2tuple


def make_model(args, parent=False):
    return MT(
        img_size=64,
        patch_size=1,
        in_chans=args.n_colors,
        embed_dim=96,
        depths=(6, 6, 6, 6, 6, 6),
        d_state=16,
        mamba_expand=2.0,
        mamba_dropout=0.1,
        mnb_kernel_sizes=(3, 5, 7),
        mnb_dilations=(1, 1, 1),
        mnb_ffn_ratio=4.0,
        mnb_dropout=0.1,
        drop_rate=0.0,
        use_checkpoint=False,
        upscale=args.scale[0],
        img_range=float(args.rgb_range),
        upsampler='pixelshuffle',
        resi_connection='1conv',
    )


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        dim = int(dim[0]) if isinstance(dim, (tuple, list)) else int(dim)
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class Mamba2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2.0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.K = 4
        assert self.d_inner % self.K == 0, "d_inner must be divisible by 4"
        self.d_inner_group = self.d_inner // self.K

        self.channel_down = nn.Conv2d(self.d_inner, self.d_inner_group, kernel_size=1, bias=False, **factory_kwargs)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        x_proj = [
            nn.Linear(self.d_inner_group, (self.dt_rank + 2 * self.d_state), bias=False, **factory_kwargs)
            for _ in range(4)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in x_proj], dim=0))

        dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner_group, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(4)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))

        self.A_logs = self.A_log_init(self.d_state, self.d_inner_group, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner_group, copies=4, merge=True)
        self.selective_scan = selective_scan_fn

        self.restore = nn.Linear(self.d_inner_group, self.d_inner, bias=bias, **factory_kwargs)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n -> r n", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        b, c, h, w = x.shape
        l = h * w
        k = 4

        x_hw = x.clone()
        x_hw[:, :, 1::2] = torch.flip(x_hw[:, :, 1::2], dims=[3])
        x_hw = x_hw.view(b, c, l)

        x_wh = x.permute(0, 1, 3, 2).contiguous()
        x_wh[:, :, 1::2] = torch.flip(x_wh[:, :, 1::2], dims=[3])
        x_wh = x_wh.view(b, c, l)

        x_hwwh = torch.stack([x_hw, x_wh], dim=1)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(b, -1, l)
        dts = dts.contiguous().float().view(b, -1, l)
        Bs = Bs.float().view(b, k, -1, l)
        Cs = Cs.float().view(b, k, -1, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(b, k, -1, l)

        y_hw = out_y[:, 0]
        y_wh = out_y[:, 1]
        y_hw_inv = torch.flip(out_y[:, 2], dims=[-1])
        y_wh_inv = torch.flip(out_y[:, 3], dims=[-1])

        def restore_hw(seq):
            y2d = seq.view(b, c, h, w)
            y2d[:, :, 1::2] = torch.flip(y2d[:, :, 1::2], dims=[3])
            return y2d.view(b, c, l)

        def restore_wh(seq):
            y2d = seq.view(b, c, w, h)
            y2d[:, :, 1::2] = torch.flip(y2d[:, :, 1::2], dims=[3])
            y2d = y2d.permute(0, 1, 3, 2).contiguous()
            return y2d.view(b, c, l)

        return restore_hw(y_hw), restore_hw(y_hw_inv), restore_wh(y_wh), restore_wh(y_wh_inv)

    def forward(self, x: torch.Tensor, **kwargs):
        b, h, w, c = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        x = self.channel_down(x)

        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = y.transpose(1, 2).contiguous().view(b, h, w, -1)
        y = self.restore(y)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class NATTENBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, dilation=1, dropout=0.0, ffn_ratio=4.0, num_heads=8):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = NeighborhoodAttention(dim, kernel_size=kernel_size, dilation=dilation, num_heads=num_heads)
        self.norm2 = LayerNorm2d(dim)
        hidden_dim = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Dropout(dropout),
        )
        self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma2 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.attn(x)
        x = x.permute(0, 3, 1, 2)
        x = residual + self.gamma1 * x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.gamma2 * x
        return x


class MultiScaleNATTENBlock(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        kernel_sizes=(3, 5),
        dilations=(1, 1),
        gate_hidden_ratio=0.25,
        gate_tau=0.8,
        ffn_ratio=4.0,
        dropout=0.0,
        use_ln=False,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations)
        assert in_channels % len(kernel_sizes) == 0

        self.gate_tau = gate_tau
        self.num_branches = len(kernel_sizes)
        self.chunk_channels = in_channels // self.num_branches
        self.embed_dim = 2 * self.chunk_channels

        self.down = nn.ModuleList([nn.Conv2d(self.chunk_channels, self.embed_dim, kernel_size=1) for _ in kernel_sizes])
        self.blocks = nn.ModuleList([
            nn.Sequential(*[
                NATTENBlock(
                    dim=self.embed_dim,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    dropout=dropout,
                    ffn_ratio=ffn_ratio,
                    num_heads=max(1, self.embed_dim // 32),
                )
                for _ in range(depth)
            ])
            for i in range(self.num_branches)
        ])
        gate_hidden = max(4, int(in_channels * gate_hidden_ratio))
        self.gate_pool = nn.AdaptiveAvgPool2d(1)
        self.gate_fc1 = nn.Conv2d(in_channels, gate_hidden, 1)
        self.gate_fc2 = nn.Conv2d(gate_hidden, self.num_branches, 1)
        self.up = nn.ModuleList([nn.Conv2d(self.embed_dim, self.chunk_channels, kernel_size=1) for _ in kernel_sizes])
        self.mix = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.post_norm = LayerNorm2d(in_channels) if use_ln else nn.Identity()

    def forward(self, x):
        residual = x
        chunks = torch.chunk(x, self.num_branches, dim=1)
        gate = self.gate_pool(x)
        gate = F.gelu(self.gate_fc1(gate))
        gate = self.gate_fc2(gate).view(x.shape[0], self.num_branches)
        gate = F.softmax(gate / self.gate_tau, dim=1)

        outs = []
        for i, chunk in enumerate(chunks):
            y = self.down[i](chunk)
            y = self.blocks[i](y)
            y = self.up[i](y)
            y = y * gate[:, i].view(-1, 1, 1, 1)
            outs.append(y)

        x = torch.cat(outs, dim=1)
        x = self.mix(x)
        x = self.post_norm(x)
        return x + residual


class ChannelAttention(nn.Module):
    def __init__(self, dim, squeeze_factor=16):
        super().__init__()
        hidden = max(1, dim // squeeze_factor)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


class ResidualStateSpaceBlock(nn.Module):
    def __init__(self, dim, d_state, mamba_expand, mamba_dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba2D(
            d_model=dim,
            d_state=d_state,
            expand=mamba_expand,
            dropout=mamba_dropout,
        )
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x + self.scale * self.mamba(self.norm(x))


class RSSB_MNB_Block(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        d_state,
        mamba_expand,
        mamba_dropout,
        mnb_kernel_sizes,
        mnb_dilations,
        mnb_ffn_ratio,
        mnb_dropout,
    ):
        super().__init__()
        self.rssb = nn.ModuleList([
            ResidualStateSpaceBlock(dim, d_state, mamba_expand, mamba_dropout)
            for _ in range(depth)
        ])
        self.mnb = MultiScaleNATTENBlock(
            depth=max(1, depth // 2),
            in_channels=dim,
            kernel_sizes=mnb_kernel_sizes,
            dilations=mnb_dilations,
            ffn_ratio=mnb_ffn_ratio,
            dropout=mnb_dropout,
        )
        self.ca = ChannelAttention(dim)
        self.fuse = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        for blk in self.rssb:
            x = blk(x)
        y = x.permute(0, 3, 1, 2).contiguous()
        y = self.mnb(y)
        y = self.ca(y)
        y = self.fuse(y)
        return y.permute(0, 2, 3, 1).contiguous()


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        d_state,
        mamba_expand,
        mamba_dropout,
        mnb_kernel_sizes,
        mnb_dilations,
        mnb_ffn_ratio,
        mnb_dropout,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.block = RSSB_MNB_Block(
            dim=dim,
            depth=depth,
            d_state=d_state,
            mamba_expand=mamba_expand,
            mamba_dropout=mamba_dropout,
            mnb_kernel_sizes=mnb_kernel_sizes,
            mnb_dilations=mnb_dilations,
            mnb_ffn_ratio=mnb_ffn_ratio,
            mnb_dropout=mnb_dropout,
        )
        self.downsample = downsample(input_resolution, dim=dim) if downsample else None

    def forward(self, x, x_size=None):
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.block, x)
        else:
            x = self.block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.patches_resolution = [
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        ]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def forward(self, x, x_size=None):
        return x


class ResidualGroup(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        d_state,
        mamba_expand,
        mamba_dropout,
        mnb_kernel_sizes,
        mnb_dilations,
        mnb_ffn_ratio,
        mnb_dropout,
        downsample=None,
        use_checkpoint=False,
        img_size=None,
        patch_size=1,
        resi_connection='1conv',
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            d_state=d_state,
            mamba_expand=mamba_expand,
            mamba_dropout=mamba_dropout,
            mnb_kernel_sizes=mnb_kernel_sizes,
            mnb_dilations=mnb_dilations,
            mnb_ffn_ratio=mnb_ffn_ratio,
            mnb_dropout=mnb_dropout,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"Invalid resi_connection: {resi_connection}")

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim)
        self.patch_unembed = PatchUnEmbed()

    def forward(self, x, x_size):
        res = self.residual_group(x, x_size)
        res = self.patch_unembed(res, x_size)
        res = res.permute(0, 3, 1, 2)
        res = self.conv(res)
        res = self.patch_embed(res)
        res = res.permute(0, 2, 3, 1).contiguous()
        return res + x


class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch):
        super().__init__(
            nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1),
            nn.PixelShuffle(scale),
        )


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m += [nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1), nn.PixelShuffle(2)]
        elif scale == 3:
            m += [nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1), nn.PixelShuffle(3)]
        else:
            raise ValueError(f"scale {scale} not supported")
        super().__init__(*m)


class MT(nn.Module):
    def __init__(
        self,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=96,
        depths=(6, 6, 6, 6, 6, 6),
        d_state=16,
        mamba_expand=2.0,
        mamba_dropout=0.1,
        mnb_kernel_sizes=(3, 5, 7),
        mnb_dilations=(1, 1, 1),
        mnb_ffn_ratio=4.0,
        mnb_dropout=0.1,
        drop_rate=0.0,
        use_checkpoint=False,
        upscale=4,
        img_range=1.0,
        upsampler='pixelshuffle',
        resi_connection='1conv',
    ):
        super().__init__()
        self.img_range = img_range
        self.upsampler = upsampler
        self.embed_dim = embed_dim

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim, embed_dim)
        self.patch_unembed = PatchUnEmbed()
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.patches_resolution = self.patch_embed.patches_resolution

        self.layers = nn.ModuleList([
            ResidualGroup(
                dim=embed_dim,
                input_resolution=self.patches_resolution,
                depth=d,
                d_state=d_state,
                mamba_expand=mamba_expand,
                mamba_dropout=mamba_dropout,
                mnb_kernel_sizes=mnb_kernel_sizes,
                mnb_dilations=mnb_dilations,
                mnb_ffn_ratio=mnb_ffn_ratio,
                mnb_dropout=mnb_dropout,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
            )
            for d in depths
        ])

        self.norm = LayerNorm2d(embed_dim)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"Invalid resi_connection: {resi_connection}")

        if upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, 64, 3, 1, 1),
                nn.LeakyReLU(True),
            )
            self.upsample = Upsample(upscale, 64)
            self.conv_last = nn.Conv2d(64, in_chans, 3, 1, 1)
        elif upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, in_chans)
        else:
            self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

    def forward_features(self, x):
        x_size = x.shape[2:4]
        x = self.pos_drop(self.patch_embed(x))
        x = x.permute(0, 2, 3, 1).contiguous()
        for layer in self.layers:
            x = layer(x, x_size)
        x = x.permute(0, 3, 1, 2)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.conv_first(x)
        res = self.forward_features(x)
        x = self.conv_after_body(res) + x

        if self.upsampler == 'pixelshuffle':
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            x = self.upsample(x)
        else:
            x = self.conv_last(x)

        return x.clamp_(0.0, self.img_range)
