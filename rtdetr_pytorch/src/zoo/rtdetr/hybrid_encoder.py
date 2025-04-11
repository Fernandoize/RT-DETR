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

from ..deformable_attention.encoder.dat_blocks import DeformableMHAGQA
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
        reference_points = torch.cat(reference_points_list, 2)
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
                 n_head,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,
                 deformable_encoder=False,
                 num_levels=3,
                 num_points=4):
        super().__init__()
        self.normalize_before = normalize_before
        self.deformable_encoder = deformable_encoder

        # Self attention
        if self.deformable_encoder:
            self.self_attn = MSDeformableAttentionGQA(d_model, n_head, num_kv_heads=n_head, num_levels=num_levels, num_points=num_points)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout, batch_first=True)

        # Cross attention between different feature levels
        self.cross_attn = MSDeformableAttentionGQA(d_model, n_head, num_kv_heads=n_head, num_levels=num_levels, num_points=num_points)

        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None, reference_points=None, spatial_shapes=None, memory=None, memory_spatial_shapes=None):
        # Self attention
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        
        q = k = self.with_pos_embed(src, pos_embed)
        if self.deformable_encoder:
            src2, _ = self.self_attn(q, reference_points, value=src, value_spatial_shapes=spatial_shapes, value_mask=src_mask)
        else:
            src2, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)
        src = residual + self.dropout1(src2)
        if not self.normalize_before:
            src = self.norm1(src)

        # Cross attention with other feature levels
        if memory is not None:
            residual = src
            if self.normalize_before:
                src = self.norm2(src)
            
            src2, _ = self.cross_attn(
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
            src = self.norm3(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout3(src2)
        if not self.normalize_before:
            src = self.norm3(src)

        return src

class CrossAttentionEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, deformable_encoder=False):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.deformable_encoder = deformable_encoder

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
        
        if self.num_layers > 0 and self.deformable_encoder:
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
                 # 交叉注意力的层数
                 num_cross_attention_layers=1,
                 # 交叉注意力Deformable Attention中参考点的个数
                 num_cross_attention_points=4,
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
        self.use_cross_attention = use_cross_attention

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

        # encoder transformer
        # 对指定的层级（use_encoder_idx）应用 Transformer 编码器。
        # 将特征图展平为 [B, H*W, C] 的形状，添加位置编码后输入 Transformer。
        # 将 Transformer 的输出恢复为 [B, C, H, W] 的形状。

        # self.encoder = nn.ModuleList([])
        # for _ in range(len(use_encoder_idx)):
        #     encoder_layer = TransformerEncoderLayer(
        #         hidden_dim,
        #         n_head=nhead,
        #         dim_feedforward=dim_feedforward,
        #         dropout=dropout,
        #         activation=enc_act,
        #         deformable_encoder=deformable_encoder)
        #     self.encoder.append(TransformerEncoder(encoder_layer, num_encoder_layers, deformable_encoder=deformable_encoder))

        # self.encoder = TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers, deformable_encoder=deformable_encoder)

        # top-down fpn
        # 从高层级到低层级，通过上采样和特征融合逐步生成金字塔特征
        # self.lateral_convs = nn.ModuleList()
        # self.fpn_blocks = nn.ModuleList()
        # for _ in range(len(in_channels) - 1, 0, -1):
        #     self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
        #     # 从上到下降维
        #     self.fpn_blocks.append(
        #         CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
        #     )

        # bottom-up pan
        # 从低层级到高层级，通过下采样和特征融合进一步优化金字塔特征
        # self.downsample_convs = nn.ModuleList()
        # self.pan_blocks = nn.ModuleList()
        # for _ in range(len(in_channels) - 1):
        #     self.downsample_convs.append(
        #         ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
        #     )
        #     self.pan_blocks.append(
        #         CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
        #     )

        if self.use_cross_attention:
            print("use cross attention")
            # Initialize cross attention encoder
            cross_encoder_layer = CrossAttentionEncoderLayer(
                hidden_dim,
                n_head=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=enc_act,
                deformable_encoder=deformable_encoder,
                    num_levels=len(in_channels),
                num_points=num_cross_attention_points
            )
            self.cross_encoder = CrossAttentionEncoder(
                cross_encoder_layer,
                num_cross_attention_layers,
                deformable_encoder=deformable_encoder
            )

        self._reset_parameters()

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

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        # encoder
        # if self.num_encoder_layers > 0:
        #     for i, enc_ind in enumerate(self.use_encoder_idx):
        #         h, w = proj_feats[enc_ind].shape[2:]
        #         spatial_shapes = [(h, w)]
        #         # flatten [B, C, H, W] to [B, HxW, C]
        #         src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
        #
        #         if self.training or self.eval_spatial_size is None:
        #             pos_embed = self.build_2d_sincos_position_embedding(
        #                 w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
        #         else:
        #             pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
        #
        #         memory = self.encoder[i](src_flatten, pos_embed=pos_embed, spatial_shapes=spatial_shapes)
        #         proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # Cross attention between feature levels
        if self.use_cross_attention:
            # Prepare memory for cross attention
            memory_list = []
            memory_spatial_shapes = []
            for feat in proj_feats:
                h, w = feat.shape[2:]
                memory_list.append(feat.flatten(2).permute(0, 2, 1))
                memory_spatial_shapes.append((h, w))
            
            memory = torch.cat(memory_list, dim=1)
            memory_spatial_shapes = torch.tensor(memory_spatial_shapes, device=memory.device)
            
            # Apply cross attention
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                spatial_shapes = [(h, w)]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
                
                output = self.cross_encoder(
                    src_flatten,
                    pos_embed=pos_embed,
                    spatial_shapes=spatial_shapes,
                    memory=memory,
                    memory_spatial_shapes=memory_spatial_shapes
                )
                proj_feats[enc_ind] = output.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # # 是否可以先融合
        # # broadcasting and fusion
        # inner_outs = [proj_feats[-1]]
        # for idx in range(len(self.in_channels) - 1, 0, -1):
        #     feat_high = inner_outs[0]
        #     feat_low = proj_feats[idx - 1]
        #     feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
        #     inner_outs[0] = feat_high
        #     upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
        #     inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
        #     inner_outs.insert(0, inner_out)
        #
        # outs = [inner_outs[0]]
        # for idx in range(len(self.in_channels) - 1):
        #     feat_low = outs[-1]
        #     feat_high = inner_outs[idx + 1]
        #     downsample_feat = self.downsample_convs[idx](feat_low)
        #     out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
        #     outs.append(out)

        return proj_feats
