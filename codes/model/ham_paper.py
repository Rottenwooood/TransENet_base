import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def make_model(args, parent=False):
    return HAM(
        img_size=48,
        patch_size=1,
        in_chans=args.n_colors,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        drop_rate=0.0,
        d_state=16,
        mlp_ratio=2.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        upscale=args.scale[0],
        img_range=float(args.rgb_range),
        upsampler='pixelshuffle',
        resi_connection='1conv',
    )


class HAA(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, n, c, h, w = x.size()
        proj_query = x.view(b, n, -1)
        proj_key = x.view(b, n, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(b, n, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(b, n, c, h, w)
        out = self.gamma * out + x
        return out.view(b, -1, h, w)


class HSFA(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, in_features // 5, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_features // 5, in_features // 5, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_features // 5, out_features, 1, 1, 0),
        )
        self.linear = nn.Conv2d(in_features, out_features, 1, 1, 0)

    def forward(self, x):
        conv_out = self.conv(x)
        linear_out = self.linear(x)
        return conv_out * linear_out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DFSE(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            SELayer(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0),
        )
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            SELayer(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0),
        )

    def forward(self, x):
        _, _, h, w = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')

        ori_mag = torch.abs(x_freq)
        mag = ori_mag + self.processmag(ori_mag)

        ori_pha = torch.angle(x_freq)
        pha = ori_pha + self.processpha(ori_pha)

        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_freq_spatial = torch.fft.irfft2(x_out, s=(h, w), norm='backward')
        return x_freq_spatial + x


class SFIIM(nn.Module):
    def __init__(self, in_nc):
        super().__init__()
        self.spatial_process = HSFA(in_nc, in_nc)
        self.frequency_process = DFSE(in_nc)
        self.cat = nn.Conv2d(2 * in_nc, in_nc, 1, 1, 0)

    def forward(self, x):
        x_freq = self.frequency_process(x)
        x = self.spatial_process(x)
        return self.cat(torch.cat([x, x_freq], 1))


class VSSM(nn.Module):
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

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

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
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        b, c, h, w = x.shape
        l = h * w
        k = 4

        x_hwwh = torch.stack(
            [x.view(b, -1, l), torch.transpose(x, dim0=2, dim1=3).contiguous().view(b, -1, l)],
            dim=1,
        ).view(b, 2, -1, l)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(b, k, -1, l), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(b, k, -1, l), self.dt_projs_weight)

        xs = xs.float().view(b, -1, l)
        dts = dts.contiguous().float().view(b, -1, l)
        Bs = Bs.float().view(b, k, -1, l)
        Cs = Cs.float().view(b, k, -1, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(b, k, -1, l)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(b, 2, -1, l)
        wh_y = torch.transpose(out_y[:, 1].view(b, -1, w, h), dim0=2, dim1=3).contiguous().view(b, -1, l)
        invwh_y = torch.transpose(inv_y[:, 1].view(b, -1, w, h), dim0=2, dim1=3).contiguous().view(b, -1, l)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        b, h, w, c = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, 1, 2).contiguous().view(b, h, w, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class SFMB(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        expand: float = 2.0,
        input_resolution=None,
        is_light_sr=False,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = VSSM(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, expand=expand)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.full((hidden_dim,), 0.1))
        self.conv_blk = SFIIM(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.full((hidden_dim,), 0.1))

    def forward(self, input, x_size):
        b, l, c = input.shape
        input = input.view(b, *x_size, c).contiguous()
        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.self_attention(x))
        x = x * self.skip_scale2 + self.conv_blk(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        return x.view(b, -1, c).contiguous()


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        drop_path=0.0,
        d_state=16,
        mlp_ratio=2.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        is_light_sr=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SFMB(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=nn.LayerNorm,
                    attn_drop_rate=0,
                    d_state=d_state,
                    expand=mlp_ratio,
                    input_resolution=input_resolution,
                    is_light_sr=is_light_sr,
                )
            )
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class SFMG(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        d_state=16,
        mlp_ratio=4.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        img_size=None,
        patch_size=None,
        resi_connection='1conv',
        is_light_sr=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            d_state=d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr=is_light_sr,
        )
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"Unsupported resi_connection: {resi_connection}")

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()
        return flops


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        return x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])

    def flops(self):
        return 0


class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        super().__init__(
            nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1),
            nn.PixelShuffle(scale),
        )


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"scale {scale} is not supported. Supported scales: 2^n and 3.")
        super().__init__(*m)


class HAM(nn.Module):
    def __init__(
        self,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        drop_rate=0.0,
        d_state=16,
        mlp_ratio=2.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        upscale=4,
        img_range=1.0,
        upsampler='pixelshuffle',
        resi_connection='1conv',
        **kwargs,
    ):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            self.mean = torch.Tensor((0.4488, 0.4371, 0.4040)).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio = mlp_ratio

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.is_light_sr = self.upsampler == 'pixelshuffledirect'
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SFMG(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                d_state=d_state,
                mlp_ratio=self.mlp_ratio,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                is_light_sr=self.is_light_sr,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.haa = HAA(embed_dim)
        self.last_conv = nn.Conv2d(embed_dim * 6, embed_dim, 3, 1, 1)
        self.last = nn.Conv2d(embed_dim * 2, embed_dim, 3, 1, 1)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"Unsupported resi_connection: {resi_connection}")

        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        residuals = []
        for layer in self.layers:
            x = layer(x, x_size)
            residuals.append(x.unsqueeze(1))

        x_layer = torch.cat(residuals, dim=1)
        x_layer = x_layer.view(x_layer.size(0), x_layer.size(1), self.embed_dim, x_size[0], x_size[1])
        x_layer = self.haa(x_layer)
        x_layer = self.last_conv(x_layer)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        x = self.conv_after_body(x)

        out = torch.cat([x_layer, x], 1)
        return self.last(out)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)
            x = self.forward_features(x) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean
        return x
