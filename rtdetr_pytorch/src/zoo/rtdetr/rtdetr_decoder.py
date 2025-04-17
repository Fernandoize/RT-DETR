"""by lyuwenyu
"""

import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from src.core import register
from .denoising import get_contrastive_denoising_training_group
from .utils import bias_init_with_prob
from .utils import deformable_attention_core_func, get_activation, inverse_sigmoid

__all__ = ['RTDETRTransformer']

from ..deformable_attention.ms_deformable_attention import MSDeformableAttentionGQA

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4,):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2,)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = deformable_attention_core_func

        self._reset_parameters()


    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)


    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask

        # dim = num_head * head_dim
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        # 使用 query samping offset
        # offset和权重都是由query生成
        # [bs, len, heads, level, points, 2] 2代表x, y两个方向
        # mlp生成
        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2)

        # offset的权重 使用 mlp生成
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2)

            # reference_points 的范围在0-1之间, 先将 sampling_offsets 归一化道参考点0-1范围之内
            sampling_locations = reference_points.reshape(
                bs, Len_q, 1, self.num_levels, 1, 2
            ) + sampling_offsets / offset_normalizer

        # shape为4代表参考点的坐标为(x, y, w, h)
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        # value [b, token_len, head, head_dim]
        # value spatial shapes [[a, a], [b, b], [c, c], [d, d]]
        # sampling_locations 已经进行了偏移 [b, query_size, head, level, points_num, location(x, y)]
        # attention_weights[b, query_size, head, level, points_num] 一个location一个weights
        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 n_kv_head=2,
                 use_gqa=False,
                 ):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttentionGQA(d_model, n_head, num_kv_heads=n_head, num_levels=n_levels, num_points=n_points)
        # self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # gate
        # 门控机制gate来控制信息流 于控制自注意力和交叉注意力输出之间的信息流
        # self.gateway = BottleneckGate(d_model)
        # self.gateway2 = BottleneckGate(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # self._reset_parameters()

    # def _reset_parameters(self):
    #     linear_init_(self.linear1)
    #     linear_init_(self.linear2)
    #     xavier_uniform_(self.linear1.weight)
    #     xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)

        # if attn_mask is not None:
        #     attn_mask = torch.where(
        #         attn_mask.to(torch.bool),
        #         torch.zeros_like(attn_mask),
        #         torch.full_like(attn_mask, float('-inf'), dtype=tgt.dtype))

        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2, _ = self.cross_attn(
            # query 中包含了位置信息
            self.with_pos_embed(tgt, query_pos_embed),
            # 参考点
            reference_points,
            # k,v token
            memory,
            memory_spatial_shapes,
            memory_mask)

        # 通过门控机制 gateway 控制自注意力和交叉注意力输出之间的信息流。
        # tgt = self.gateway(tgt, self.dropout2(tgt2))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)

        # tgt = self.gateway2(tgt, self.dropout4(tgt2))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                tgt,
                # 参考点
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_quality = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            # 注意：参考点的位置在不断的更新，因此 query_pos_embed 也在不断编码，其会作为最终的损失
            ref_points_input = ref_points_detach.unsqueeze(2)
            # 查询的位置编码是通过参考点生成的， dino中 ref_points_detach 先生成了正弦位置编码，然后才计算的embed
            # query_pos_embed = query_pos_head(
            #     reference_points=ref_points_detach,
            #     memory=memory
            # )
            query_pos_embed = query_pos_head(ref_points_detach)

            # query_pos_embed 为query中添加位置信息,
            output = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed)

            # 这里使用了多层迭代的思想，第一层的输出作为第二层的输入
            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            # 保存上一层的参考点
            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


class EnhancedPositionEncoding(nn.Module):
    def __init__(self, hidden_dim, num_scales=4, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        
        # 可学习的高斯参数
        self.sigmas = nn.Parameter(torch.ones(num_scales))  # 不同尺度的sigma
        self.centers = nn.Parameter(torch.zeros(num_scales, 2))  # 不同尺度的中心点
        
        # 多尺度特征融合
        self.scale_fusion = nn.Sequential(
            nn.Linear(num_scales, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 位置编码的最终投影
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # self.mlp = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)
        
        # 初始化网络参数
        self._reset_parameters()
        
    def _reset_parameters(self):
        # 为不同尺度设置不同的初始sigma
        sigmas = torch.linspace(0.5, 2.0, self.num_scales)
        with torch.no_grad():
            self.sigmas.copy_(sigmas)
        
        # 为中心点设置不同的初始位置
        # 分别初始化x和y坐标
        centers_x = torch.rand(self.num_scales) * 0.2 - 0.1  # 在[-0.1, 0.1]范围内随机初始化
        centers_y = torch.rand(self.num_scales) * 0.2 - 0.1
        with torch.no_grad():
            self.centers[:, 0] = centers_x
            self.centers[:, 1] = centers_y
        
        # 初始化scale_fusion模块
        for m in self.scale_fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
        
        # 初始化proj模块
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

        # 初始化mlp
        # init.xavier_uniform_(self.mlp.layers[0].weight)
        # init.xavier_uniform_(self.mlp.layers[1].weight)
        
    def gaussian_response(self, points, center, sigma):
        # 考虑方向性的距离计算, 输入数据与高斯函数的中心位置的接近程度
        diff = points - center
        distance = torch.sum(diff**2, dim=-1)
        direction = torch.atan2(diff[..., 1], diff[..., 0])  # 计算方向角
        return torch.exp(-distance / (2 * sigma**2)) * (1 + 0.1 * torch.cos(direction))
    
    def generate_multi_scale_gaussian(self, reference_points):
        """
        生成多尺度高斯响应
        reference_points: [bs, num_queries, 4] (x,y,w,h)
        """
        bs, num_queries, _ = reference_points.shape
        center_points = reference_points[..., :2]  # [bs, num_queries, 2]
        
        # 为每个尺度生成高斯响应
        gaussian_responses = []
        for i in range(self.num_scales):
            # 使用当前尺度的sigma和中心点
            response = self.gaussian_response(
                center_points.reshape(-1, 2),  # 展平所有点
                self.centers[i],  # 当前尺度的中心点
                self.sigmas[i]    # 当前尺度的sigma
            )
            gaussian_responses.append(response.reshape(bs, num_queries, 1))
        
        # 堆叠多尺度响应 [bs, num_queries, num_scales]
        multi_scale_response = torch.cat(gaussian_responses, dim=-1)
        
        return multi_scale_response
    
    def forward(self, reference_points, memory):
        """
        reference_points: [bs, num_queries, 4] (x,y,w,h)
        memory: [bs, num_tokens, hidden_dim] 编码器的输出特征
        """
        # 1. 生成多尺度高斯响应
        multi_scale_response = self.generate_multi_scale_gaussian(reference_points)

        # mlp_response = self.mlp(reference_points)
        
        # 2. 融合多尺度信息
        # scale_features = self.scale_fusion(multi_scale_response)  # [bs, num_queries, hidden_dim]
        
        # 3. 添加层归一化
        pos_embed = self.proj(multi_scale_response)
        
        return pos_embed


@register
class RTDETRTransformer(nn.Module):
    __share__ = ['num_classes']
    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True):

        super(RTDETRTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        # self.query_count = 0
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss

        # backbone feature projection： 将来自于backbone的特征投影到统一的维度 hidden_dim
        self._build_input_proj_layer(feat_channels)

        # 解码器
        # Transformer module 每层包含自注意力机制和交叉注意力机制
        # TODO 只在某些层使用DAT
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # 用于去噪训练中的类别嵌入
        # denoising part
        if num_denoising > 0:
            # self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim, padding_idx=num_classes-1) # TODO for load paddle weights
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)
        # self.query_pos_head = EnhancedPositionEncoding(
        #     hidden_dim=hidden_dim,
        #     num_scales=4,
        #     num_heads=nhead
        # )

        # 编码器
        # encoder head: 对编码器进一步处理，生成编码器的最终输出
        # layernorm可以尝试替换为hekaiming最新提出的模块或者dw卷积
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim,)
        )

        # 生成类别分数和边界框坐标
        # TODO 添加一个物体数量预测头，根据预测的数量作为权重保留query
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)  # Changed to binary classification (background/foreground)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        # 解码器的输出：类别分数、边界框坐标、边界框质量
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)  # Keep multi-class classification
            for _ in range(num_decoder_layers)
        ])
        self.dec_quality_head = nn.ModuleList([
            nn.Linear(hidden_dim, 1)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)

        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for cls_, reg_, quality_ in zip(self.dec_score_head, self.dec_bbox_head,
                                                      self.dec_quality_head):
            init.constant_(cls_.bias, bias)
            init.constant_(quality_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)
        # linear_init_(self.enc_output[0])
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)


    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )
            )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          # 0.05 适合于捕捉中小尺寸的目标
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(\
                torch.arange(end=h, dtype=dtype), \
                torch.arange(end=w, dtype=dtype), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h]).to(dtype)
            # 计算grid_xy的中心点，然后与特征图的大小进行归一化
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            # x,y代表参考点中心坐标，wh代表宽和高
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, 1).to(device)
        # 边界值, 筛选出不合法的边界锚框
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        # 将锚点从[0,1]转换到对数空间，便于后续回归任务的学习，在损失计算中，inf和nan值会被过滤掉
        anchors = torch.log(anchors / (1 - anchors))
        # anchors = torch.where(valid_mask, anchors, float('inf'))
        # anchors[valid_mask] = torch.inf # valid_mask [1, 8400, 1]
        # 将边界锚框中无效的标记为inf, 确保后续不会参与计算
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_decoder_input(self,
                           # 编码器输出
                           memory,
                           # 特征图的形状
                           spatial_shapes,
                           # 去噪训练中的类别嵌入
                           denoising_class=None,
                           # 去噪训练中未激活的边界框
                           denoising_bbox_unact=None):
        """
        准备解码器的输入数据：为解码器生成目标特征、参考点、边界框和类别分数
        """
        bs, _, _ = memory.shape
        # prepare input for decoder
        # 1.生成参考点，仅在训练时或 eval_spatial_size未设置时生成
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        # memory = torch.where(valid_mask, memory, 0)
        # 将token中边界值也置为0
        memory = valid_mask.to(memory.dtype) * memory  # TODO fix type error for onnx export

        output_memory = self.enc_output(memory)

        # 每个token输出一个classes和bboxes
        enc_outputs_class = self.enc_score_head(output_memory)  # Shape: [bs, num_tokens, 2]

        # 参考点是encoder输出的token    + anchors获得的, enc_bbox_head 输出的是偏移量offset, 表示相对于锚点的调整值
        # 直接预测坐标会导致训练不稳定，尤其是目标尺度变化较大时，预测偏移量则可以更好的约束模型的学习范围，使其更容易收敛
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        # 2. topk query select & 归一化得到 reference_points_unact
        # 选择出预测可能性最大的topk class index
        if enc_outputs_class.shape[-1] == 2:  # Binary classification case
            # 对于二分类，分别计算背景和前景的概率
            # enc_outputs_class shape: [bs, num_tokens, 2]
            # 使用softmax计算每个类别的概率
            probs = F.softmax(enc_outputs_class, dim=-1)  # Shape: [bs, num_tokens, 2]
            # 获取前景概率（索引1）
            foreground_probs = probs[:, :, 1] # Shape: [bs, num_tokens]
            # 使用topk选择前景概率最高的num_queries个query
            _, topk_ind = torch.topk(foreground_probs, self.num_queries, dim=1)
        else:  # Multi-class classification case
            # 使用原有的topk选择方式
            _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)

        # 参考点 & 归一化
        # 根据参考点取出对应的值
        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))

        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat(
                [denoising_bbox_unact, reference_points_unact], 1)

        # 对于二分类，我们只需要前景类别的分数
        enc_topk_logits = enc_outputs_class.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, 2))  # Shape: [bs, num_queries, 2]

        # extract region features
        # TODO Topk位置聚合
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(dim=1, \
                index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits


    def forward(self, feats, targets=None):

        # input projection and embedding
        # 1. 获取编码器的输出
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)

        # prepare denoising training
        # 2. 生成去噪训练所需的对比样本。
        if self.training and self.num_denoising > 0:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale, )
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        # 3. 准备解码器的输出
        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact)

        # decoder 4. 解码器
        out_bboxes, out_logits = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask)

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))

            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta

        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b }
                for a, b in zip(outputs_class, outputs_coord)]


class Gate(nn.Module):
    """
    门控机制，用于控制两个输入张量之间的信息流，门控机制可以动态的融合来自不同源的信息
    """
    def __init__(self, d_model):
        super(Gate, self).__init__()
        # d_model：模型的隐藏维度，即输入张量 x1 和 x2 的维度。
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        # 初始化偏置，使得初始门控信号接近0.5
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        # 将线性层的权重初始化为0，确保初始门控信号主要由偏置决定。
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        # (B, L, 2 * d_modle)
        gate_input = torch.cat([x1, x2], dim=-1)
        # 通过线性层 self.gate 处理 gate_input，生成门控信号, 使用Sigmoid激活函数将门控信号归一化到 [0, 1] 范围内。
        gates = torch.sigmoid(self.gate(gate_input))
        # 将门控信号 gates 拆分为两个部分 gate1 和 gate2，每个部分的形状为 (B, L, d_model)。
        gate1, gate2 = gates.chunk(2, dim=-1)
        # 通过层归一化 self.norm 对加权和结果进行归一化，稳定训练过程
        return self.norm(gate1 * x1 + gate2 * x2)


class BottleneckGate(nn.Module):
    """
    优化门控机制 V2：使用瓶颈结构计算单一门控信号。
    先将维度降低，再恢复，进一步减少参数。
    融合方式仍为 g * x1 + (1 - g) * x2。
    """
    def __init__(self, d_model, bottleneck_ratio=0.25): # bottleneck_ratio 控制压缩比例
        super().__init__()
        bottleneck_dim = int(d_model * bottleneck_ratio)
        if bottleneck_dim < 1: # 保证瓶颈维度至少为1
            bottleneck_dim = 1

        # 第一个线性层：降维 (2*d_model -> bottleneck_dim)
        self.linear1 = nn.Linear(2 * d_model, bottleneck_dim)
        self.activation = nn.GELU() # 或 ReLU
        # 第二个线性层：升维 (bottleneck_dim -> d_model)
        self.linear2 = nn.Linear(bottleneck_dim, d_model)

        # 初始化第二个线性层的偏置，使得初始门控 g 接近 0.5
        bias = bias_init_with_prob(0.5)
        init.constant_(self.linear2.bias, bias)
        # 权重初始化为0
        init.constant_(self.linear1.weight, 0)
        init.constant_(self.linear2.weight, 0) # 关键：第二个线性层权重也初始化为0

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        # 拼接输入: (B, L, 2 * d_model)
        gate_input = torch.cat([x1, x2], dim=-1)
        # 通过瓶颈结构计算门控信号
        hidden = self.activation(self.linear1(gate_input))
        g = torch.sigmoid(self.linear2(hidden)) # (B, L, d_model)
        # 融合
        fused_output = g * x1 + (1 - g) * x2
        # 层归一化
        return self.norm(fused_output)