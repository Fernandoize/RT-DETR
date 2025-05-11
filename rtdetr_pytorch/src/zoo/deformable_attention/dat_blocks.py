# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia
# --------------------------------------------------------

import einops
import numpy as np
import torch
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_
from torch import nn
from torch.nn.init import trunc_normal_


class LocalAttention(nn.Module):

    def __init__(self, dim, heads, window_size, attn_drop, proj_drop):
        super().__init__()

        window_size = to_2tuple(window_size)  # 将输入的 window_size 转换为二元组 (Wh, Ww)，即使输入是单个整数。

        self.proj_qkv = nn.Linear(dim, 3 * dim)  # 对输入特征进行线性变换，生成 query (q), key (k), value (v) 三个部分。输出维度是输入维度的 3 倍。
        self.heads = heads  # 注意力头的数量。
        assert dim % heads == 0  # 确保特征维度可以被注意力头的数量整除。
        head_dim = dim // heads  # 每个注意力头的特征维度。
        self.scale = head_dim ** -0.5  # 缩放因子，用于在计算注意力分数时稳定梯度。
        self.proj_out = nn.Linear(dim, dim)  # 对注意力输出进行线性变换，恢复原始特征维度。
        self.window_size = window_size  # 局部窗口的大小 (Wh, Ww)。
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)  # 输出投影后的 dropout 层。
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)  # 注意力权重后的 dropout 层。

        Wh, Ww = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), heads)
        )  # 创建一个可学习的相对位置偏置表。表的大小是 (2*Wh - 1) * (2*Ww - 1) x heads。这个尺寸覆盖了窗口内所有可能的相对位置。
        trunc_normal_(self.relative_position_bias_table, std=0.01)  # 使用截断正态分布初始化相对位置偏置表。

        coords_h = torch.arange(self.window_size[0])  # 生成高度方向的坐标 [0, 1, ..., Wh-1]。
        coords_w = torch.arange(self.window_size[1])  # 生成宽度方向的坐标 [0, 1, ..., Ww-1]。
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww。生成窗口内所有像素的坐标网格。
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww。将坐标网格展平成 (2, Wh*Ww) 的形状，每列代表一个像素的 (h, w) 坐标。
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww。计算窗口内任意两个像素之间的相对坐标。
        relative_coords = relative_coords.permute(1, 2,
                                                  0).contiguous()  # Wh*Ww, Wh*Ww, 2。将相对坐标的维度调整为 (Wh*Ww, Wh*Ww, 2)。
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0。将高度方向的相对坐标平移到 [0, 2*Wh - 2] 的范围。
        relative_coords[:, :, 1] += self.window_size[1] - 1  # shift to start from 0。将宽度方向的相对坐标平移到 [0, 2*Ww - 2] 的范围。
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1  # 将高度方向的相对坐标乘以一个偏移量，以便与宽度方向的相对坐标组合成唯一的索引。
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww。将两个方向的相对坐标相加，得到一个唯一的相对位置索引。
        self.register_buffer("relative_position_index",
                             relative_position_index)  # 将计算得到的相对位置索引注册为 buffer，它不是模型的参数，但会保存在模型的状态中。

    def forward(self, x, mask=None):
        B, C, H, W = x.size()  # 获取输入特征图的批大小、通道数、高度和宽度。
        r1, r2 = H // self.window_size[0], W // self.window_size[1]  # 计算高度和宽度方向上窗口的数量。

        x_total = einops.rearrange(x, 'b c (r1 h1) (r2 w1) -> b (r1 r2) (h1 w1) c', h1=self.window_size[0],
                                   w1=self.window_size[
                                       1])  # 将输入特征图分割成不重叠的局部窗口，并重新排列形状为 (B, Nr*Nw, Wh*Ww, C)，其中 Nr 和 Nw 是高度和宽度方向的窗口数量。
        # b: batch size
        # c: channel dimension
        # r1, r2: number of windows in height and width
        # h1, w1: window height and width
        # m = r1 * r2: total number of windows
        # n = h1 * w1: number of tokens in each window

        x_total = einops.rearrange(x_total, 'b m n c -> (b m) n c')  # 将批大小和窗口数量合并，形状变为 ((B*Nr*Nw), Wh*Ww, C)。

        qkv = self.proj_qkv(x_total)  # 对每个窗口内的特征进行线性变换，得到 query, key, value。形状为 ((B*Nr*Nw), Wh*Ww, 3*C)。
        q, k, v = torch.chunk(qkv, 3, dim=2)  # 将 qkv 分割成 query, key, value，每个形状为 ((B*Nr*Nw), Wh*Ww, C)。

        q = q * self.scale  # 对 query 进行缩放。
        q, k, v = [einops.rearrange(t, 'b n (h c1) -> b h n c1', h=self.heads) for t in
                   [q, k, v]]  # 将特征按注意力头进行分割，形状变为 ((B*Nr*Nw), heads, Wh*Ww, head_dim)。
        attn = torch.einsum('b h m c, b h n c -> b h m n', q, k)  # 计算注意力权重。形状为 ((B*Nr*Nw), heads, Wh*Ww, Wh*Ww)。

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
            -1)  # Wh*Ww,Wh*Ww,nH。根据相对位置索引从可学习的偏置表中获取对应的偏置值。
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww。调整偏置的形状。
        attn_bias = relative_position_bias
        attn = attn + attn_bias.unsqueeze(0)  # 将相对位置偏置添加到注意力权重中。

        if mask is not None:
            # attn : (b * nW) h w w
            # mask : nW ww ww
            nW, ww, _ = mask.size()  # 获取 mask 的形状。mask 通常用于处理可变长度的序列或在窗口注意力中引入额外的约束。
            attn = einops.rearrange(attn, '(b n) h w1 w2 -> b n h w1 w2', n=nW, h=self.heads, w1=ww,
                                    w2=ww) + mask.reshape(1, nW, 1, ww, ww)  # 如果提供了 mask，则将其添加到注意力权重中。
            attn = einops.rearrange(attn, 'b n h w1 w2 -> (b n) h w1 w2')  # 恢复注意力权重的形状。
        attn = self.attn_drop(attn.softmax(dim=3))  # 对注意力权重进行 softmax 归一化，并应用 dropout。

        x = torch.einsum('b h m n, b h n c -> b h m c', attn,
                         v)  # 使用注意力权重对 value 进行加权求和。形状为 ((B*Nr*Nw), heads, Wh*Ww, head_dim)。
        x = einops.rearrange(x, 'b h n c1 -> b n (h c1)')  # 将注意力头的维度合并回特征维度，形状变为 ((B*Nr*Nw), Wh*Ww, C)。
        x = self.proj_drop(self.proj_out(x))  # 对输出进行线性投影和 dropout。形状为 ((B*Nr*Nw), Wh*Ww, C)。
        x = einops.rearrange(x, '(b r1 r2) (h1 w1) c -> b c (r1 h1) (r2 w1)', r1=r1, r2=r2, h1=self.window_size[0],
                             w1=self.window_size[1])  # 将窗口重新组合成原始的特征图形状 (B, C, H, W)。

        return x, None, None  # 返回局部注意力处理后的特征图 x，以及两个 None 值 (通常在自注意力机制中用于返回注意力权重等信息，但在局部注意力中可能不直接返回)。


class ShiftWindowAttention(LocalAttention):

    def __init__(self, dim, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size):

        super().__init__(dim, heads, window_size, attn_drop, proj_drop)

        self.fmap_size = to_2tuple(fmap_size)
        self.shift_size = shift_size

        assert 0 < self.shift_size < min(self.window_size), "wrong shift size."

        img_mask = torch.zeros(*self.fmap_size)  # H W
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[h, w] = cnt
                cnt += 1
        mask_windows = einops.rearrange(img_mask, '(r1 h1) (r2 w1) -> (r1 r2) (h1 w1)', h1=self.window_size[0],
                                        w1=self.window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW ww ww
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):

        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        sw_x, _, _ = super().forward(shifted_x, self.attn_mask)
        x = torch.roll(sw_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        return x, None, None

class DAttentionBaseline(nn.Module):

    def __init__(
            self, q_size, kv_size, n_heads, n_head_channels, n_groups,
            attn_drop, proj_drop, stride,
            offset_range_factor, use_pe, dwc_pe,
            no_off, fixed_pe, ksize, log_cpb
    ):

        super().__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        # self.kv_h, self.kv_w = kv_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        # 将channel分为 n_groups, 每个group多个head,尽可能增加groups中形变的多样性
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe and not self.no_off:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(
                    self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            elif self.log_cpb:
                # Borrowed from Swin-V2
                self.rpe_table = nn.Sequential(
                    nn.Linear(2, 32, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, self.n_group_heads, bias=False)
                )
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, src, value):

        B, C, H, W = src.size()
        dtype, device = src.dtype, src.device

        q = self.proj_q(src)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        # 通过卷积生成 query offset
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        # 2. offset range参数
        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        # 2. 参考点生成
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        # 3. 参考点 + offset
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        # 4. 形变key, value
        if self.no_off:
            x_sampled = F.avg_pool2d(value, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk, f"Size is {x_sampled.size()}"
        else:
            x_sampled = F.grid_sample(
                input=value.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        # 3. 注意力计算
        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        if self.use_pe and (not self.no_off):

            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels,
                                                                              H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
            elif self.log_cpb:
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                            q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                                   n_sample,
                                                                                                   2).unsqueeze(1)).mul(
                    4.0)  # d_y, d_x [-8, +8]
                displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
                attn_bias = self.rpe_table(displacement)  # B * g, H * W, n_sample, h_g
                attn = attn + einops.rearrange(attn_bias, 'b m n h -> (b h) m n', h=self.n_group_heads)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                            q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                                   n_sample,
                                                                                                   2).unsqueeze(1)).mul(
                    0.5)
                attn_bias = F.grid_sample(
                    input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads,
                                           g=self.n_groups),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True)  # B * g, h_g, HW, Ns

                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)

class DAttentionBaselineV1(nn.Module):

    def __init__(
            self,
            stride=8,
            offset_range_factor=-1, use_pe=True, dwc_pe=False,
            no_off=False, fixed_pe=False, ksize=9, log_cpb=False,
            embed_dim=256, num_heads=8, num_groups=4,
            attn_drop=0, proj_drop=0
    ):
        """
            use_pe： 是否使用位置编码
            stride: 可形变注意力的采样步长
            offset_range_factor: 控制可形变注意力中偏移量的范围
            use_pe: 默认使用rpe位置编码
            offset_range_factor： 默认不开


        """

        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_channel = int(self.embed_dim // self.num_heads)
        self.dwc_pe = dwc_pe
        self.scale = self.embed_dim / self.num_heads ** -0.5
        self.num_groups = int(num_groups)
        # 将channel分为 n_groups, 每个group多个head,尽可能增加groups中形变的多样性
        self.group_channel = int(self.embed_dim // self.num_groups)
        self.group_heads = self.num_heads // self.num_groups

        # 位置编码
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor

        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        # offset 生成卷积网络
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.group_channel, self.group_channel, kk, stride, pad_size, groups=self.group_channel),
            LayerNormProxy(self.group_channel),
            nn.GELU(),
            nn.Conv2d(self.group_channel, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        # q, k, v投影
        self.proj_q = nn.Conv2d(
            self.embed_dim, self.embed_dim,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.embed_dim, self.embed_dim,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.embed_dim, self.embed_dim,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.embed_dim, self.embed_dim,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.num_groups, -1, -1, -1)  # B * g H W 2

        return ref

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.num_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, src, spatial_shape, value):
        H, W = spatial_shape[0]
        if self.use_pe and not self.no_off and self.rpe_table is None:
            self.rpe_table = nn.Parameter(
                torch.zeros(self.num_heads, W * 2 - 1, W * 2 - 1)
            )
            trunc_normal_(self.rpe_table, std=0.01)

        src = einops.rearrange(src, 'b (h w) c -> b c h w', h=H, w=W)
        value = einops.rearrange(value, 'b (h w) c -> b c h w', h=H, w=W)
        B, C, H, W = src.size()
        dtype, device = src.dtype, src.device

        q = self.proj_q(src)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.num_groups, c=self.group_channel)
        # 通过卷积生成 query offset
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        # 2. offset range参数
        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        # 2. 参考点生成
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        # 3. 参考点 + offset
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        # 4. 形变key, value
        pos = pos.to(src.device)  # 确保 pos 在正确的设备上
        if self.no_off:
            x_sampled = F.avg_pool2d(value, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk, f"Size is {x_sampled.size()}"
        else:
            x_sampled = F.grid_sample(
                input=value.reshape(B * self.num_groups, self.group_channel, H, W),
                grid=pos[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        # 3. 注意力计算
        q = q.reshape(B * self.num_heads, self.head_channel, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.num_heads, self.head_channel, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.num_heads, self.head_channel, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        if self.use_pe and (not self.no_off):

            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.num_heads, self.head_channel,
                                                                              H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.num_heads, H * W, n_sample)
            elif self.log_cpb:
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                            q_grid.reshape(B * self.num_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.num_groups,
                                                                                                   n_sample,
                                                                                                   2).unsqueeze(1)).mul(
                    4.0)  # d_y, d_x [-8, +8]
                displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
                attn_bias = self.rpe_table(displacement)  # B * g, H * W, n_sample, h_g
                attn = attn + einops.rearrange(attn_bias, 'b m n h -> (b h) m n', h=self.group_heads)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1).to(src.device)
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                            q_grid.reshape(B * self.num_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.num_groups,
                                                                                                   n_sample,
                                                                                                   2).unsqueeze(1)).mul(
                    0.5).to(src.device)
                attn_bias = F.grid_sample(
                    input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.group_heads,
                                           g=self.num_groups),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True)  # B * g, h_g, HW, Ns

                attn_bias = attn_bias.reshape(B * self.num_heads, H * W, n_sample)
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        y = einops.rearrange(y, 'b c h w  -> b (h w) c')
        return y, pos.reshape(B, self.num_groups, Hk, Wk, 2), reference.reshape(B, self.num_groups, Hk, Wk, 2)


class PyramidAttention(nn.Module):

    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., sr_ratio=1):

        super().__init__()

        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Conv2d(dim, dim, 1, 1, 0)
        self.kv = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.proj_ds = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio),
                LayerNormProxy(dim)
            )

    def forward(self, x):

        B, C, H, W = x.size()
        Nq = H * W
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ds = self.proj_ds(x)
            kv = self.kv(x_ds)
        else:
            kv = self.kv(x)

        k, v = torch.chunk(kv, 2, dim=1)
        Nk = (H // self.sr_ratio) * (W // self.sr_ratio)
        q = q.reshape(B * self.num_heads, self.head_dim, Nq).mul(self.scale)
        k = k.reshape(B * self.num_heads, self.head_dim, Nk)
        v = v.reshape(B * self.num_heads, self.head_dim, Nk)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        x = torch.einsum('b m n, b c n -> b c m', attn, v)
        x = x.reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, None, None


class TransformerMLP(nn.Module):

    def __init__(self, channels, expansion, drop):
        super().__init__()

        self.dim1 = channels
        self.dim2 = channels * expansion
        self.chunk = nn.Sequential()
        self.chunk.add_module('linear1', nn.Linear(self.dim1, self.dim2))
        self.chunk.add_module('act', nn.GELU())
        self.chunk.add_module('drop1', nn.Dropout(drop, inplace=True))
        self.chunk.add_module('linear2', nn.Linear(self.dim2, self.dim1))
        self.chunk.add_module('drop2', nn.Dropout(drop, inplace=True))

    def forward(self, x):
        _, _, H, W = x.size()
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = self.chunk(x)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class TransformerMLPWithConv(nn.Module):

    def __init__(self, channels, expansion, drop):
        super().__init__()

        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Sequential(
            nn.Conv2d(self.dim1, self.dim2, 1, 1, 0),
            # nn.GELU(),
            # nn.BatchNorm2d(self.dim2, eps=1e-5)
        )
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.act = nn.GELU()
        # self.bn = nn.BatchNorm2d(self.dim2, eps=1e-5)
        self.linear2 = nn.Sequential(
            nn.Conv2d(self.dim2, self.dim1, 1, 1, 0),
            # nn.BatchNorm2d(self.dim1, eps=1e-5)
        )
        self.drop2 = nn.Dropout(drop, inplace=True)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.drop1(x)
        x = x + self.dwc(x)
        x = self.act(x)
        # x = self.bn(x)
        x = self.linear2(x)
        x = self.drop2(x)

        return x