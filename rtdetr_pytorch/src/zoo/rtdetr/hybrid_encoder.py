'''by lyuwenyu
'''

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rtdetr_decoder import BottleneckGate, MSDeformableAttention
from .utils import get_activation

from src.core import register


__all__ = ['HybridEncoder']

from ..deformable_attention.dat_blocks import DAttentionBaselineV1

from ..deformable_attention.ms_deformable_attention import MSDeformableAttentionGQA


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size-1)//2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    """
    ReceptionVGG: 使用多个不同尺度的卷积核进行并行特征提取，然后再做加法
    """
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    """
    Cross Stage Partial 结合了 RepVggBlock 来实现高效的卷积操作

    Cross Stage Partial (CSP) 是一种卷积神经网络架构设计，旨在提高特征提取的效率和性能。
    CSP 结构的核心思想是将特征图分成两部分，并在不同阶段进行交叉融合，从而增强特征的多样性并减少计算量。
    expansion = 0.5 代表将特征拆分为多个部分, 然后经过不同的操作，一个分支进行卷积特征提取，另一个保持不变，最后进行融合

    1. 降低了复杂度，同时保持特征的丰富性
    2. 广泛应用于YOLO系列中
    """
    def __init__(self,
                 # 输入输出通道
                 in_channels,
                 out_channels,
                 # regvgg block数3
                 num_blocks=3,
                 # 扩展因子 1.0
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()

        # 1. 特征拆分，一般拆分为两部分
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        # 2. 多个rep vgg block进行特征提取
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        # 3. 特征融合
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 n_head,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,
                 deformable_encoder=False):
        super().__init__()
        self.normalize_before = normalize_before
        self.deformable_encoder = deformable_encoder

        if self.deformable_encoder:
            self.self_attn = MSDeformableAttentionGQA(d_model, n_head, num_kv_heads=n_head, num_levels=1, num_points=4)
            # self.self_attn = MSDeformableAttention(d_model, n_head, num_levels=1, num_points=8)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None, reference_points=None, spatial_shapes:torch.Tensor = None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        if self.deformable_encoder:
            src, _ = self.self_attn(q, reference_points, value=src, value_spatial_shapes=spatial_shapes, value_mask=src_mask)
        else:
            src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, deformable_encoder=False):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.deformable_encoder = deformable_encoder

    @staticmethod
    def get_reference_points(spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None]
                     # / (valid_ratios[:, None, lvl, 1] * H_))
            ref_x = ref_x.reshape(-1)[None]
                    # / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
                            # * valid_ratios[:, None])
        return reference_points

    def forward(self, src, src_mask=None, pos_embed=None, spatial_shapes:torch.Tensor = None) -> torch.Tensor:
        output = src

        # preparation and reshape
        reference_points = None
        if self.num_layers > 0:
            if self.deformable_encoder:
                reference_points = self.get_reference_points(spatial_shapes, device=src.device)

        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed, reference_points=reference_points, spatial_shapes=spatial_shapes)

        if self.norm is not None:
            output = self.norm(output)

        return output


class CrossAttentionEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,
                 deformable_encoder=False,
                 num_levels=3,
                 num_points=4,
                 use_cross_attention=False):
        super().__init__()
        self.normalize_before = normalize_before
        self.deformable_encoder = deformable_encoder
        self.use_cross_attention = use_cross_attention

        # Self attention
        if self.deformable_encoder:
            self.self_attn = MSDeformableAttentionGQA(d_model, nhead, num_kv_heads=nhead, num_levels=num_levels, num_points=num_points)
        else:
            # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
            self.self_attn = LocalAttention(d_model, nhead)

        # Cross attention between different feature levels
        if self.use_cross_attention:
            self.cross_attn = MSDeformableAttentionGQA(d_model, nhead, num_kv_heads=nhead, num_levels=num_levels, num_points=num_points)

        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)  # Add norm for local attention
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)  # Add dropout for local attention
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None, reference_points=None, spatial_shapes=None, memory=None, memory_spatial_shapes=None):
        # Self attention
        residual = src
        if self.normalize_before:
            src = self.norm1(src)

        # q = k = self.with_pos_embed(src, pos_embed)
        # if self.deformable_encoder:
        #     src2, _ = self.self_attn(q, reference_points, value=src, value_spatial_shapes=spatial_shapes, value_mask=src_mask)
        # else:
        #     src2 = self.self_attn(q, pos_embed=None, spatial_shapes=spatial_shapes)
        #     # src2, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)
        # src = residual + self.dropout1(src2)
        # if not self.normalize_before:
        #     src = self.norm1(src)

        # Cross attention with other feature levels
        if self.use_cross_attention:
            if memory is not None:
                residual = src
                if self.normalize_before:
                    src = self.norm2(src)

                src2, _= self.cross_attn(
                    self.with_pos_embed(src, pos_embed),
                    reference_points,
                    memory,
                    memory_spatial_shapes,
                    src_mask
                )
                src = residual + self.dropout2(src2)
                if not self.normalize_before:
                    src = self.norm2(src)

        # Feed forward network
        residual = src
        if self.normalize_before:
            src = self.norm4(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout4(src2)
        if not self.normalize_before:
            src = self.norm4(src)

        return src

class CrossAttentionEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, deformable_encoder=False, use_cross_attention=False):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.deformable_encoder = deformable_encoder
        self.use_cross_attention = use_cross_attention

    # @staticmethod
    # def get_reference_points(spatial_shapes, device, grid_size=0.05, eps=1e-2):
    #     reference_points_list = []
    #     for lvl, (H_, W_) in enumerate(spatial_shapes):
    #         ref_y, ref_x = torch.meshgrid(
    #             torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
    #             torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device)
    #         )
    #         ref_y = ref_y.reshape(-1)[None]
    #         ref_x = ref_x.reshape(-1)[None]
    #         ref_xy = torch.stack((ref_x, ref_y), -1)
    #
    #         valid_WH = torch.tensor([W_, H_]).to(torch.float32)
    #         # 计算grid_xy的中心点，然后与特征图的大小进行归一化
    #         ref_xy = (ref_xy.unsqueeze(0) + 0.5) / valid_WH
    #         # x,y代表参考点中心坐标，wh代表宽和高
    #         wh = torch.ones_like(ref_xy) * grid_size * (2.0 ** lvl)
    #         reference_points_list.append(torch.concat([ref_xy, wh], -1).reshape(-1, H_ * W_, 4))
    #
    #     reference_points = torch.concat(reference_points_list, 1).to(device)
    #     # 边界值, 筛选出不合法的边界锚框
    #     valid_mask = ((reference_points > eps) * (reference_points < 1 - eps)).all(-1, keepdim=True)
    #     # 将锚点从[0,1]转换到对数空间，便于后续回归任务的学习，在损失计算中，inf和nan值会被过滤掉
    #     reference_points = torch.log(reference_points / (1 - reference_points))
    #     reference_points = torch.where(valid_mask, reference_points, torch.inf)
    #
    #     reference_points = reference_points[:, :, None]
    #     return reference_points
    @staticmethod
    def get_reference_points(spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device)
            )
            ref_y = ref_y.reshape(-1)[None]
            ref_x = ref_x.reshape(-1)[None]
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points

    def forward(self, src, src_mask=None, pos_embed=None, spatial_shapes=None, memory=None, memory_spatial_shapes=None):
        output = src
        reference_points = None

        if self.num_layers > 0 and (self.deformable_encoder or self.use_cross_attention):
            reference_points = self.get_reference_points(spatial_shapes, device=src.device)

        for layer in self.layers:
            output = layer(
                output,
                src_mask=src_mask,
                pos_embed=pos_embed,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                memory=memory,
                memory_spatial_shapes=memory_spatial_shapes
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


@register
class HybridEncoder(nn.Module):
    def __init__(self,
                 # 输入特征的channel数
                 in_channels=[512, 1024, 2048],
                 # 表示特征图相对于输入图像的缩小倍数
                 feat_strides=[8, 16, 32],
                 # Transformer 和 FPN 中使用的隐藏层维度
                 hidden_dim=256,
                 # Transformer 的多头注意力机制中的头数
                 nhead=8,
                 # ransformer 中前馈网络的维度
                 dim_feedforward = 1024,
                 # Transformer 中的 dropout 比例
                 dropout=0.0,
                 # Transformer 中的激活函数（如 GELU
                 enc_act='gelu',
                 # 指定哪些层级的特征图需要经过 Transformer 编码器处理
                 use_encoder_idx=[1,2,3],
                 # 每个 Transformer 编码器的层数
                 num_encoder_layers=1,
                 # 位置编码的温度参数，用于控制位置编码的频率
                 pe_temperature=10000,
                 # 用于控制 CSPRepLayer 的扩展因子和深度
                 expansion=1.0,
                 depth_mult=1.0,
                 # 卷积层中的激活函数（如 SiLU
                 act='silu',
                 # 评估时输入图像的固定空间尺寸
                 eval_spatial_size=None,
                 # 是否使用deformable_encoder
                 deformable_encoder=False,
                 # 是否使用交叉注意力
                 use_cross_attention=False,
                 # 交叉注意力Deformable Attention中参考点的个数
                 num_cross_attention_points=4,
                 # 是否使用全局注意力
                 use_global_attention=False,
                 # 开启FPN
                 use_fpn=False,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.deformable_encoder = deformable_encoder
        self.use_fpn = use_fpn
        self.use_cross_attention = use_cross_attention
        self.use_global_attention = use_global_attention

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        # self.gate = BottleneckGate()

        # channel projection
        # 使用 input_proj 将每个特征图投影到统一的 hidden_dim 维度

        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )
        #     self.encoder = nn.ModuleList([])
        #     for _ in range(len(use_encoder_idx)):
        encoder_layer = CrossAttentionEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act,
            deformable_encoder=deformable_encoder,
            num_levels=len(self.in_channels),
            num_points=num_cross_attention_points,
            use_cross_attention=self.use_cross_attention,
        )
        self.encoder = CrossAttentionEncoder(
            encoder_layer,
            num_encoder_layers,
            deformable_encoder=deformable_encoder,
            use_cross_attention=self.use_cross_attention
        )

        if self.use_fpn:
            # top-down fpn
            # FPN参数量400W
            # 从高层级到低层级，通过上采样和特征融合逐步生成金字塔特征
            self.lateral_convs = nn.ModuleList()
            self.fpn_blocks = nn.ModuleList()
            for _ in range(len(in_channels) - 1, 0, -1):
                self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
                # 从上到下降维
                self.fpn_blocks.append(
                    CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
                )

            # bottom-up pan
            # 从低层级到高层级，通过下采样和特征融合进一步优化金字塔特征
            self.downsample_convs = nn.ModuleList()
            self.pan_blocks = nn.ModuleList()
            for _ in range(len(in_channels) - 1):
                self.downsample_convs.append(
                    ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
                )
                self.pan_blocks.append(
                    CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
                )
        self._reset_parameters()

        self.level_embed = nn.Parameter(torch.Tensor(len(in_channels), hidden_dim))
        nn.init.normal_(self.level_embed)  # 初始化

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward_global_attention(self, proj_feats):
        memory_list = []
        memory_spatial_shapes = []
        pos_embeds = []
        for lvl, enc_ind in enumerate(self.use_encoder_idx):
            feat = proj_feats[enc_ind]
            B, C, H, W = feat.shape
            # flatten
            src = feat.flatten(2).permute(0, 2, 1)  # [B, HW, C]
            # 1. 位置编码
            if self.training or self.eval_spatial_size is None:
                pos_embed = self.build_2d_sincos_position_embedding(
                    W, H, self.hidden_dim, self.pe_temperature).to(src.device)
            else:
                pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src.device)
            # 2. 加上 level embedding
            lvl_pos = self.level_embed[lvl].view(1, 1, -1)  # [1, 1, C]
            pos_embeds.append(pos_embed + lvl_pos)
            memory_list.append(src)
            memory_spatial_shapes.append((H, W))
        pos_embed = torch.cat(pos_embeds, dim=1)
        memory = torch.cat(memory_list, dim=1)  # [B, sum(HW), C]
        memory_spatial_shapes = torch.tensor(memory_spatial_shapes, device=memory.device)  # [n_levels, 2]


        # 4. 调用 attention
        # 假设 self.global_attn = MSDeformableAttentionGQA(...)
        memory_out = self.encoder(memory,
                                  pos_embed=pos_embed,
                                  spatial_shapes=memory_spatial_shapes,
                                  memory=memory,
                                  memory_spatial_shapes=memory_spatial_shapes)

        # 5. 拆分回各尺度
        split_sizes = [H * W for (H, W) in memory_spatial_shapes]
        outs = torch.split(memory_out, split_sizes, dim=1)
        outs = [o.permute(0, 2, 1).reshape(B, C, H, W) for o, (H, W) in zip(outs, memory_spatial_shapes)]
        return outs

    def forward_cross_attention(self, proj_feats):
        # Cross attention between feature levels
        # Prepare memory for cross attention
        memory_list = []
        # memory_spatial_shapes = []
        # for feat in proj_feats:
        #     h, w = feat.shape[2:]
        #     memory_list.append(feat.flatten(2).permute(0, 2, 1))
        #     memory_spatial_shapes.append((h, w))
        #
        # memory = torch.cat(memory_list, dim=1)
        # memory_spatial_shapes = torch.tensor(memory_spatial_shapes, device=memory.device)

        # Apply cross attention
        for lvl, enc_ind in enumerate(self.use_encoder_idx):
            h, w = proj_feats[enc_ind].shape[2:]
            spatial_shapes = [(h, w)]
            src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)

            # 对 memory_list 里的每个特征做上采样/下采样
            aligned_feats = []
            for feat in proj_feats:
                if feat.shape[2:] != (h, w):
                    # 使用最近邻或双线性插值
                    aligned_feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
                else:
                    aligned_feat = feat
                aligned_feats.append(aligned_feat)
            # flatten 并拼接
            memory_list = [f.flatten(2).permute(0, 2, 1) for f in aligned_feats]
            memory = torch.cat(memory_list, dim=1)
            memory_spatial_shapes = torch.tensor([(h, w)] * len(aligned_feats), device=memory.device)

            if self.training or self.eval_spatial_size is None:
                pos_embed = self.build_2d_sincos_position_embedding(
                    w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
            else:
                pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
            lvl_pos = self.level_embed[lvl].view(1, 1, -1)  # [1, 1, C]
            pos_embed = pos_embed + lvl_pos

            output = self.encoder(
                src_flatten,
                pos_embed=pos_embed,
                spatial_shapes=spatial_shapes,
                memory=memory,
                memory_spatial_shapes=memory_spatial_shapes
            )
            proj_feats[enc_ind] = output.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        return proj_feats

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.use_global_attention:
            proj_feats = self.forward_global_attention(proj_feats)
        else:
            proj_feats = self.forward_cross_attention(proj_feats)

        if self.use_fpn:
        # broadcasting and fusion
            inner_outs = [proj_feats[-1]]
            for idx in range(len(self.in_channels) - 1, 0, -1):
                feat_high = inner_outs[0]
                feat_low = proj_feats[idx - 1]
                feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
                inner_outs[0] = feat_high
                upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
                inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
                inner_outs.insert(0, inner_out)

            outs = [inner_outs[0]]
            for idx in range(len(self.in_channels) - 1):
                feat_low = outs[-1]
                feat_high = inner_outs[idx + 1]
                downsample_feat = self.downsample_convs[idx](feat_low)
                out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
                outs.append(out)
            return outs
        return proj_feats


class LocalAttention(nn.Module):
    """Local attention module that focuses on local regions around each query point."""

    def __init__(self, d_model, nhead, base_window_size=3, min_window_size=3, max_window_size=7, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.base_window_size = base_window_size
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.scale = (d_model // nhead) ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _get_window_size(self, H, W):
        """Dynamically calculate window size based on feature map dimensions."""
        # 计算特征图的较小维度
        min_dim = min(H, W)
        
        # 根据特征图大小动态调整window size
        # 当特征图较小时使用较小的window size，较大时使用较大的window size
        window_size = max(
            self.min_window_size,
            min(
                self.max_window_size,
                int(self.base_window_size * (min_dim / 32))  # 32是基准尺寸
            )
        )
        
        # 确保window size是奇数
        window_size = window_size if window_size % 2 == 1 else window_size + 1
        
        return window_size

    def forward(self, x, pos_embed=None, spatial_shapes=None):
        B, N, C = x.shape
        H, W = spatial_shapes[0] if spatial_shapes is not None else (int(N ** 0.5), int(N ** 0.5))

        # 动态计算window size
        window_size = self._get_window_size(H, W)
        
        # Add position embedding if provided
        if pos_embed is not None:
            x = x + pos_embed

        # Project queries, keys and values
        q = self.q_proj(x).view(B, N, self.nhead, C // self.nhead).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.nhead, C // self.nhead).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.nhead, C // self.nhead).transpose(1, 2)

        # Reshape to 2D grid
        q = q.view(B, self.nhead, H, W, C // self.nhead)
        k = k.view(B, self.nhead, H, W, C // self.nhead)
        v = v.view(B, self.nhead, H, W, C // self.nhead)

        # Calculate padding
        pad_l = pad_t = pad_r = pad_b = window_size // 2
        
        # Pad if necessary
        q = F.pad(q, (0, 0, pad_l, pad_r, pad_t, pad_b))
        k = F.pad(k, (0, 0, pad_l, pad_r, pad_t, pad_b))
        v = F.pad(v, (0, 0, pad_l, pad_r, pad_t, pad_b))

        # Unfold to get local windows
        q = q.unfold(2, window_size, 1).unfold(3, window_size, 1)
        k = k.unfold(2, window_size, 1).unfold(3, window_size, 1)
        v = v.unfold(2, window_size, 1).unfold(3, window_size, 1)

        # Reshape for attention computation
        q = q.contiguous().view(B, self.nhead, H, W, window_size * window_size, C // self.nhead)
        k = k.contiguous().view(B, self.nhead, H, W, window_size * window_size, C // self.nhead)
        v = v.contiguous().view(B, self.nhead, H, W, window_size * window_size, C // self.nhead)

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        x = (attn @ v).view(B, self.nhead, H, W, window_size * window_size, C // self.nhead)
        x = x.mean(dim=4)  # Average over local window
        x = x.transpose(1, 2).reshape(B, N, C)

        # Final projection
        x = self.out_proj(x)
        return x

class FPNDeformableAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_levels=1, num_points=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.attn = MSDeformableAttentionGQA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            num_kv_heads=num_heads,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, q, v):
        # x: [B, C, H, W]
        B, C, H, W = q.shape
        q = q.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        v = v.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        # 构造 reference_points 和 value_spatial_shapes
        reference_points = self._get_reference_points(H, W, q.device, B)
        value_spatial_shapes = torch.tensor([[H, W]], device=v.device)
        out, _ = self.attn(
            q,  # query
            reference_points,  # [B, HW, num_levels, 2]
            v,  # value
            value_spatial_shapes  # [num_levels, 2]
        )
        q = q + out
        q = self.norm(q)
        q = q.permute(0, 2, 1).reshape(B, C, H, W)
        return q

    def _get_reference_points(self, H, W, device, B):
        # 生成归一化的 reference points
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device) / H,
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device) / W,
            indexing='ij'
        )
        ref = torch.stack((grid_x, grid_y), -1)  # [H, W, 2]
        ref = ref.reshape(1, H * W, 1, 2).repeat(B, 1, 1, 1)  # [B, HW, 1, 2]
        return ref
