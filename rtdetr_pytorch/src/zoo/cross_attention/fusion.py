import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Conv2d


def get_activation(act: str, inpace: bool = True):
    '''get activation
    '''
    act = act.lower()

    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()

    elif act == 'gelu':
        m = nn.GELU()

    elif act is None:
        m = nn.Identity()

    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')

    if hasattr(m, 'inplace'):
        m.inplace = inpace

    return m


def deformable_attention_core_func_gqa(
        value, value_spatial_shapes, sampling_locations, attention_weights,
        num_heads, num_kv_heads  # Added GQA parameters
):
    """
    Args:
        value (Tensor): [bs, value_length, n_kv_heads, c_head], Input features (already projected)
        value_spatial_shapes (Tensor|List): [n_levels, 2] Spatial shapes of features.
        sampling_locations (Tensor): [bs, query_length, n_heads, n_levels, n_points, 2], Sampling locations derived from Query heads.
        attention_weights (Tensor): [bs, query_length, n_heads, n_levels, n_points], Attention weights derived from Query heads.
        num_heads (int): Number of Query heads.
        num_kv_heads (int): Number of Key/Value heads.

    Returns:
        output (Tensor): [bs, query_length, C (n_heads * c_head)]
    """
    bs, _, n_kv_h, c_head = value.shape
    _, Len_q, n_q_h, n_levels, n_points, _ = sampling_locations.shape

    assert n_q_h == num_heads, "n_heads in sampling_locations doesn't match num_heads"
    assert n_kv_h == num_kv_heads, "n_kv_heads in value doesn't match num_kv_heads"
    assert n_q_h % n_kv_h == 0, "num_heads must be divisible by num_kv_heads"
    num_q_per_kv = n_q_h // n_kv_h

    # Ensure value_spatial_shapes is a tensor for calculations
    if isinstance(value_spatial_shapes, list):
        value_spatial_shapes = torch.as_tensor(value_spatial_shapes, dtype=torch.long, device=value.device)

    # Calculate start indices for each level
    level_start_index = torch.cat((value_spatial_shapes.new_zeros((1,)),
                                   value_spatial_shapes.prod(1).cumsum(0)[:-1]))

    # Split value into list per level
    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)  # List of [bs, H*W, n_kv_h, c_head]

    # Prepare sampling grids (normalize locations to [-1, 1])
    # sampling_locations: [bs, Len_q, n_heads, n_levels, n_points, 2]
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # Get value for the current level: [bs, H*W, n_kv_h, c_head]
        value_l_ = value_list[level]

        # --- GQA Adaptation ---
        # Repeat K/V heads to match Query heads before sampling
        # [bs, H*W, n_kv_h, c_head] -> [bs, H*W, n_heads, c_head]
        value_l_ = value_l_.repeat_interleave(num_q_per_kv, dim=2)
        # --------------------

        # Reshape for grid_sample:
        # [bs, H*W, n_heads, c_head] -> [bs, H*W, n_heads*c_head] -> [bs, n_heads*c_head, H*W]
        value_l_ = value_l_.flatten(2).permute(0, 2, 1)
        # -> [bs * n_heads, c_head, H, W]
        value_l_ = value_l_.reshape(bs * n_q_h, c_head, h, w)

        # Prepare sampling grid for the current level:
        # [bs, Len_q, n_heads, n_points, 2] (select current level)
        sampling_grid_l_ = sampling_grids[:, :, :, level]
        # -> [bs, n_heads, Len_q, n_points, 2]
        sampling_grid_l_ = sampling_grid_l_.permute(0, 2, 1, 3, 4)
        # -> [bs * n_heads, Len_q, n_points, 2]
        sampling_grid_l_ = sampling_grid_l_.flatten(0, 1)

        # Perform sampling: F.grid_sample(input [N,C,Hi,Wi], grid [N,Ho,Wo,2]) -> output [N,C,Ho,Wo]
        # Input: [bs * n_heads, c_head, H, W]
        # Grid: [bs * n_heads, Len_q, n_points, 2]
        # Output: [bs * n_heads, c_head, Len_q, n_points]
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)  # Output: [bs * n_heads, c_head, Len_q, n_points]

        sampling_value_list.append(sampling_value_l_)  # List of [bs * n_heads, c_head, Len_q, n_points]

    # Concatenate sampled values across levels and points
    # Stack: list of [bs*n_h, c_h, Lq, P] -> [bs*n_h, c_h, Lq, L, P]
    # Flatten: [bs*n_h, c_h, Lq, L*P]
    sampled_values = torch.stack(sampling_value_list, dim=-2).flatten(
        -2)  # Shape: [bs * n_heads, c_head, Len_q, n_levels * n_points]

    # Reshape attention weights to match sampled values
    # attention_weights: [bs, Len_q, n_heads, n_levels, n_points]
    # Permute: [bs, n_heads, Len_q, n_levels, n_points]
    # Reshape: [bs * n_heads, 1, Len_q, n_levels * n_points] (add channel dim)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_q_h, 1, Len_q, n_levels * n_points)

    # Perform weighted sum:
    # Element-wise product: [bs*n_h, c_h, Lq, L*P] * [bs*n_h, 1, Lq, L*P] -> [bs*n_h, c_h, Lq, L*P]
    # Sum over last dim (levels * points): [bs * n_heads, c_head, Len_q]
    output = (sampled_values * attention_weights).sum(-1)

    # Reshape output back: [bs * n_heads, c_head, Len_q] -> [bs, n_heads * c_head, Len_q]
    output = output.reshape(bs, n_q_h * c_head, Len_q)

    # Final permute: [bs, Len_q, C]
    return output.permute(0, 2, 1)


class MSDeformableAttentionGQA(nn.Module):  # Renamed class
    def __init__(self, embed_dim=256, num_heads=8, num_kv_heads=None,  # Added num_kv_heads
                 num_levels=4, num_points=4, use_dynamic_range=False):
        """
        Multi-Scale Deformable Attention Module with GQA support
        Args:
            embed_dim (int): Dimension of input features.
            num_heads (int): Number of Query heads.
            num_kv_heads (int): Number of Key/Value heads. If None, defaults to num_heads (standard MHA).
            num_levels (int): Number of feature levels.
            num_points (int): Number of sampling points per query per feature level.
            use_dynamic_range (bool): Whether to use dynamic range prediction. Defaults to False.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        # --- GQA Setup ---
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        self.num_kv_heads = num_kv_heads
        self.num_q_per_kv = num_heads // num_kv_heads
        # --- End GQA Setup ---

        self.embed_dim = embed_dim
        self.num_heads = num_heads  # Query heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points  # Based on Query heads
        self.use_dynamic_range = use_dynamic_range

        self.head_dim = embed_dim // num_heads
        self.kv_embed_dim = self.num_kv_heads * self.head_dim  # Dimension for K/V projection

        # 主要目的是将embeding压缩到4
        # Sampling offsets and attention weights are derived from the Query, so depend on num_heads
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        # Value projection now projects to kv_embed_dim
        self.value_proj = nn.Linear(embed_dim, self.kv_embed_dim)
        # Output projection takes the combined output (embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # Add range predictor if dynamic range is enabled
        if self.use_dynamic_range:
            # Predict x and y ranges separately
            self.range_predictor_x = nn.Linear(embed_dim, num_heads)
            self.range_predictor_y = nn.Linear(embed_dim, num_heads)
            self._reset_range_predictor()

        # Use the GQA-adapted core function
        self.ms_deformable_attn_core = deformable_attention_core_func_gqa

        self._reset_parameters()

    def _reset_range_predictor(self):
        """Initialize range predictor parameters"""
        # Initialize x range predictor
        init.constant_(self.range_predictor_x.weight, 0)
        init.constant_(self.range_predictor_x.bias, 0.5)  # Initialize to middle range

        # Initialize y range predictor
        init.constant_(self.range_predictor_y.weight, 0)
        init.constant_(self.range_predictor_y.bias, 0.5)  # Initialize to middle range

    def _reset_parameters(self):
        # sampling_offsets (depends on num_heads)
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # Normalize grid points to lie inside [-1, 1] square
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        # Repeat for levels and points
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        # Scale points outward
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init = grid_init * scaling
        self.sampling_offsets.bias.data.copy_(grid_init.flatten())  # Use copy_

        # attention_weights (depends on num_heads)
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias,
                       1.0 / (self.num_levels * self.num_points))  # Initialize weights uniformly

        # Projections
        init.xavier_uniform_(self.value_proj.weight)  # Initialize based on new kv_embed_dim
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(self,
                query,  # [bs, query_length, C]
                reference_points,  # [bs, query_length, n_levels, 2] or [bs, query_length, n_levels, 4]
                value,  # [bs, value_length, C]
                value_spatial_shapes,  # Tensor or List: [n_levels, 2]
                value_mask=None):  # [bs, value_length] (optional)
        """
        Args: see original MSDeformableAttention
        Returns:
            output (Tensor): [bs, query_length, C]
        """
        bs, Len_q, _ = query.shape
        bs, Len_v, _ = value.shape
        # Ensure value_spatial_shapes is a tensor
        if isinstance(value_spatial_shapes, list):
            value_spatial_shapes = torch.as_tensor(value_spatial_shapes, dtype=torch.long, device=query.device)

        # Project value to kv_embed_dim
        value = self.value_proj(value)  # [bs, value_length, kv_embed_dim]

        if value_mask is not None:
            # value_mask: [bs, value_length] -> [bs, value_length, 1]
            value_mask = value_mask.unsqueeze(-1).to(value.dtype)
            value = value * value_mask  # Apply mask before reshaping heads

        # Reshape value to include K/V heads dimension
        # [bs, value_length, kv_embed_dim] -> [bs, value_length, num_kv_heads, head_dim]
        value = value.reshape(bs, Len_v, self.num_kv_heads, self.head_dim)

        # Generate sampling offsets and attention weights from Query
        # Offsets: [bs, query_length, total_points * 2] -> [bs, query_length, n_heads, n_levels, n_points, 2]
        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2)

        # Weights: [bs, query_length, total_points] -> [bs, query_length, n_heads, n_levels * n_points]
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points)
        # Softmax over points and levels for each Query head: [bs, query_length, n_heads, n_levels * n_points]
        attention_weights = F.softmax(attention_weights, dim=-1)
        # Reshape weights: [bs, query_length, n_heads, n_levels, n_points]
        attention_weights = attention_weights.reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points)

        # Apply dynamic range prediction if enabled
        if self.use_dynamic_range:
            # Predict x and y attention ranges separately: [bs, query_length, n_heads]
            attention_range_x = torch.sigmoid(self.range_predictor_x(query))
            attention_range_y = torch.sigmoid(self.range_predictor_y(query))

            # Reshape for broadcasting: [bs, query_length, n_heads, 1, 1, 1]
            attention_range_x = attention_range_x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            attention_range_y = attention_range_y.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            # Create a new tensor for scaled offsets instead of modifying inplace
            scaled_offsets = torch.zeros_like(sampling_offsets)
            scaled_offsets[..., 0] = sampling_offsets[..., 0] * attention_range_x.squeeze(-1)  # x direction
            scaled_offsets[..., 1] = sampling_offsets[..., 1] * attention_range_y.squeeze(-1)  # y direction
            sampling_offsets = scaled_offsets

        # Prepare sampling locations based on reference points and offsets
        if reference_points.shape[-1] == 2:  # Top-left (0,0), bottom-right (1,1) format
            # Create normalizer: [n_levels, 2] -> [1, 1, 1, n_levels, 1, 2]
            offset_normalizer = value_spatial_shapes.flip([1]).reshape(  # Use H, W -> W, H format for normalization
                1, 1, 1, self.num_levels, 1, 2).to(query.dtype)
            # Expand reference points: [bs, Len_q, n_levels, 1, 2] -> [bs, Len_q, 1, n_levels, 1, 2]
            reference_points_expanded = reference_points[:, :, None, :, None, :]  # Add head_dim and point_dim
            # Calculate sampling locations: ref + offset / size
            sampling_locations = reference_points_expanded + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:  # Center (cx, cy, w, h) format
            # Use width/height from reference points for normalization
            # ref[:, :, None, :, None, :2]: center points (x,y) [bs, Len_q, 1, n_levels, 1, 2]
            # offsets: [bs, Len_q, n_heads, n_levels, n_points, 2]
            # ref[:, :, None, :, None, 2:]: width/height (w,h) [bs, Len_q, 1, n_levels, 1, 2]
            sampling_locations = (
                    reference_points[:, :, None, :, None, :2]  # Add head_dim and point_dim
                    + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        # --- Call the GQA-adapted core function ---
        # value: [bs, value_length, num_kv_heads, head_dim]
        # value_spatial_shapes: [n_levels, 2]
        # sampling_locations: [bs, Len_q, num_heads, n_levels, n_points, 2]
        # attention_weights: [bs, Len_q, num_heads, n_levels, n_points]
        output = self.ms_deformable_attn_core(
            value, value_spatial_shapes, sampling_locations, attention_weights,
            self.num_heads, self.num_kv_heads  # Pass head counts
        )
        # output: [bs, Len_q, embed_dim]

        # Final output projection
        output = self.output_proj(output)  # [bs, Len_q, embed_dim]

        return output, attention_weights


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
                 use_cross_attention=False,
                 use_local_attention=False,
                 window_size=3):
        super().__init__()
        self.normalize_before = normalize_before
        self.deformable_encoder = deformable_encoder
        self.use_cross_attention = use_cross_attention
        self.use_local_attention = use_local_attention

        # Cross attention between different feature levels
        if self.use_cross_attention:
            self.cross_attn = MSDeformableAttentionGQA(d_model, nhead, num_kv_heads=nhead, num_levels=num_levels,
                                                       num_points=num_points)
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

    def forward(self, src, src_mask=None, pos_embed=None, reference_points=None, spatial_shapes=None, memory=None,
                memory_spatial_shapes=None):

        # Cross attention with other feature levels
        if self.use_cross_attention:
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


class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN"):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = Conv2d(in_chan, in_chan, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv = Conv2d(in_chan, out_chan, kernel_size=1, bias=False)

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat


class CrossAttentionFusion(nn.Module):
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
                 dim_feedforward=1024,
                 # Transformer 中的 dropout 比例
                 dropout=0.0,
                 # Transformer 中的激活函数（如 GELU
                 enc_act='gelu',
                 # 指定哪些层级的特征图需要经过 Transformer 编码器处理
                 use_encoder_idx=[1, 2, 3],
                 # 每个 Transformer 编码器的层数
                 num_encoder_layers=1,
                 # 位置编码的温度参数，用于控制位置编码的频率
                 pe_temperature=10000,
                 # 评估时输入图像的固定空间尺寸
                 eval_spatial_size=None,
                 # 是否使用deformable_encoder
                 deformable_encoder=False,
                 # 是否使用交叉注意力
                 use_cross_attention=False,
                 # 交叉注意力Deformable Attention中参考点的个数
                 num_cross_attention_points=4,
                 # 是否使用全局注意力
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

        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )
        # self.encoder = nn.ModuleList([])
        encoder_layer0 = CrossAttentionEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act,
            deformable_encoder=False,
            num_levels=len(self.in_channels),
            num_points=num_cross_attention_points,
            use_cross_attention=self.use_cross_attention,
        )
        self.encoder = CrossAttentionEncoder(
            encoder_layer0,
            num_encoder_layers,
            deformable_encoder=False,
            use_cross_attention=self.use_cross_attention
        )
        # 添加上采样和下采样卷积层
        self.upsample_convs = nn.ModuleList()
        self.downsample_convs = nn.ModuleList()

        # 为每个可能的尺度差创建上采样和下采样卷积
        max_scale_diff = len(in_channels) - 1
        for i in range(max_scale_diff):
            # 上采样卷积 (使用转置卷积)
            self.upsample_convs.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True)
                )
            )

            # 下采样卷积
            self.downsample_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True)
                )
            )

        # 添加level embedding
        self.level_embed = nn.Parameter(torch.zeros(len(in_channels), hidden_dim))

        # 初始化level embedding
        nn.init.normal_(self.level_embed, std=0.02)

        # 添加特征选择模块
        self.upsample_selection = nn.ModuleList()
        self.downsample_selection = nn.ModuleList()

        for _ in range(max_scale_diff):
            self.upsample_selection.append(
                FeatureSelectionModule(hidden_dim, hidden_dim)
            )
            self.downsample_selection.append(
                FeatureSelectionModule(hidden_dim, hidden_dim)
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

        # 初始化上采样卷积层权重
        for i, upsample_conv in enumerate(self.upsample_convs):
            # 初始化转置卷积权重
            nn.init.kaiming_normal_(upsample_conv[0].weight, mode='fan_out', nonlinearity='relu')
            # 初始化BatchNorm层
            if isinstance(upsample_conv[1], nn.BatchNorm2d):
                nn.init.constant_(upsample_conv[1].weight, 1)
                nn.init.constant_(upsample_conv[1].bias, 0)

        # 初始化下采样卷积层权重
        for i, downsample_conv in enumerate(self.downsample_convs):
            # 初始化卷积权重
            nn.init.kaiming_normal_(downsample_conv[0].weight, mode='fan_out', nonlinearity='relu')
            # 初始化BatchNorm层
            if isinstance(downsample_conv[1], nn.BatchNorm2d):
                nn.init.constant_(downsample_conv[1].weight, 1)
                nn.init.constant_(downsample_conv[1].bias, 0)

        # 初始化特征选择模块权重
        for i, selection in enumerate(self.upsample_selection):
            # 初始化注意力卷积层
            nn.init.kaiming_normal_(selection.conv_atten.weight, mode='fan_out', nonlinearity='relu')
            # 初始化输出卷积层
            nn.init.kaiming_normal_(selection.conv.weight, mode='fan_out', nonlinearity='relu')

        for i, selection in enumerate(self.downsample_selection):
            # 初始化注意力卷积层
            nn.init.kaiming_normal_(selection.conv_atten.weight, mode='fan_out', nonlinearity='relu')
            # 初始化输出卷积层
            nn.init.kaiming_normal_(selection.conv.weight, mode='fan_out', nonlinearity='relu')

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
    def forward_cross_attention(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        for lvl, enc_ind in enumerate(self.use_encoder_idx):
            h, w = proj_feats[enc_ind].shape[2:]
            spatial_shapes = [(h, w)]
            src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)

            # 准备对齐后的特征
            aligned_feats = []
            aligned_spatial_shapes = []

            for i, feat in enumerate(proj_feats):
                if i == enc_ind:
                    # 当前层保持不变
                    aligned_feat = feat
                elif i > enc_ind:
                    # 高层特征需要上采样到当前层
                    scale_diff = i - enc_ind
                    aligned_feat = feat
                    # 多次上采样
                    for j in range(scale_diff):
                        # 先应用特征选择模块
                        aligned_feat = self.upsample_selection[j](aligned_feat)
                        # 再使用双线性插值上采样
                        aligned_feat = F.interpolate(
                            aligned_feat,
                            size=(aligned_feat.shape[2] * 2, aligned_feat.shape[3] * 2),
                            mode='bilinear',
                            align_corners=False
                        )
                else:
                    # 低层特征需要下采样到当前层
                    scale_diff = enc_ind - i
                    aligned_feat = feat
                    # 多次下采样
                    for j in range(scale_diff):
                        # 先应用特征选择模块
                        aligned_feat = self.downsample_selection[j](aligned_feat)
                        # 再使用卷积下采样
                        aligned_feat = self.downsample_convs[j](aligned_feat)

                # 确保特征通道数正确
                if aligned_feat.shape[1] != self.hidden_dim:
                    aligned_feat = self.input_proj[i](aligned_feat)

                aligned_feats.append(aligned_feat)
                aligned_spatial_shapes.append((aligned_feat.shape[2], aligned_feat.shape[3]))

            # 将所有对齐后的特征展平并拼接
            memory_list = []
            for feat in aligned_feats:
                memory_list.append(feat.flatten(2).permute(0, 2, 1))

            memory = torch.cat(memory_list, dim=1)
            memory_spatial_shapes = torch.tensor(aligned_spatial_shapes, device=memory.device)

            # 位置编码
            if self.training or self.eval_spatial_size is None:
                pos_embed = self.build_2d_sincos_position_embedding(
                    w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
            else:
                pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

            # 添加level embedding
            lvl_pos = self.level_embed[lvl].view(1, 1, -1)
            pos_embed = pos_embed + lvl_pos

            # 应用交叉注意力
            output = self.encoder(
                src_flatten,
                pos_embed=pos_embed,
                spatial_shapes=spatial_shapes,
                memory=memory,
                memory_spatial_shapes=memory_spatial_shapes
            )

            # 重塑输出
            proj_feats[enc_ind] = output.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        return proj_feats

    def forward(self, feats):
        return self.forward_cross_attention(feats)
