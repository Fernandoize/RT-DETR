import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import copy # For deep copying spatial shapes

# Assume deformable_attention_core_func is defined later or imported
# We will modify it directly here for clarity

def deformable_attention_core_func_gqa(
    value, value_spatial_shapes, sampling_locations, attention_weights,
    num_heads, num_kv_heads # Added GQA parameters
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
    value_list = value.split(split_shape, dim=1) # List of [bs, H*W, n_kv_h, c_head]

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
            align_corners=False) # Output: [bs * n_heads, c_head, Len_q, n_points]

        sampling_value_list.append(sampling_value_l_) # List of [bs * n_heads, c_head, Len_q, n_points]

    # Concatenate sampled values across levels and points
    # Stack: list of [bs*n_h, c_h, Lq, P] -> [bs*n_h, c_h, Lq, L, P]
    # Flatten: [bs*n_h, c_h, Lq, L*P]
    sampled_values = torch.stack(sampling_value_list, dim=-2).flatten(-2) # Shape: [bs * n_heads, c_head, Len_q, n_levels * n_points]

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


class MSDeformableAttentionGQA(nn.Module): # Renamed class
    def __init__(self, embed_dim=256, num_heads=8, num_kv_heads=None, # Added num_kv_heads
                 num_levels=4, num_points=4):
        """
        Multi-Scale Deformable Attention Module with GQA support
        Args:
            embed_dim (int): Dimension of input features.
            num_heads (int): Number of Query heads.
            num_kv_heads (int): Number of Key/Value heads. If None, defaults to num_heads (standard MHA).
            num_levels (int): Number of feature levels.
            num_points (int): Number of sampling points per query per feature level.
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
        self.num_heads = num_heads # Query heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points # Based on Query heads

        self.head_dim = embed_dim // num_heads
        self.kv_embed_dim = self.num_kv_heads * self.head_dim # Dimension for K/V projection

        # 主要目的是将embeding压缩到4
        # Sampling offsets and attention weights are derived from the Query, so depend on num_heads
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        # Value projection now projects to kv_embed_dim
        self.value_proj = nn.Linear(embed_dim, self.kv_embed_dim)
        # Output projection takes the combined output (embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # Use the GQA-adapted core function
        self.ms_deformable_attn_core = deformable_attention_core_func_gqa

        self._reset_parameters()


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
        self.sampling_offsets.bias.data.copy_(grid_init.flatten()) # Use copy_

        # attention_weights (depends on num_heads)
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 1.0 / (self.num_levels * self.num_points)) # Initialize weights uniformly

        # Projections
        init.xavier_uniform_(self.value_proj.weight) # Initialize based on new kv_embed_dim
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)


    # todo 在query中添加位置编码embedding; 四维坐标，并在decoder每一层对齐优化 来自于 DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    # Mixed Query Selection
    # Mixed Query Selection： content query, position query, denoise query
    def forward(self,
                query,              # [bs, query_length, C]
                reference_points,   # [bs, query_length, n_levels, 2] or [bs, query_length, n_levels, 4]
                value,              # [bs, value_length, C]
                value_spatial_shapes, # Tensor or List: [n_levels, 2]
                value_mask=None):   # [bs, value_length] (optional)
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
        value = self.value_proj(value) # [bs, value_length, kv_embed_dim]

        if value_mask is not None:
            # value_mask: [bs, value_length] -> [bs, value_length, 1]
            value_mask = value_mask.unsqueeze(-1).to(value.dtype)
            value = value * value_mask # Apply mask before reshaping heads

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

        # Prepare sampling locations based on reference points and offsets
        if reference_points.shape[-1] == 2: # Top-left (0,0), bottom-right (1,1) format
             # Create normalizer: [n_levels, 2] -> [1, 1, 1, n_levels, 1, 2]
            offset_normalizer = value_spatial_shapes.flip([1]).reshape( # Use H, W -> W, H format for normalization
                 1, 1, 1, self.num_levels, 1, 2).to(query.dtype)
            # Expand reference points: [bs, Len_q, n_levels, 1, 2] -> [bs, Len_q, 1, n_levels, 1, 2]
            reference_points_expanded = reference_points[:, :, None, :, None, :] # Add head_dim and point_dim
            # Calculate sampling locations: ref + offset / size
            sampling_locations = reference_points_expanded + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4: # Center (cx, cy, w, h) format
            # Use width/height from reference points for normalization
            # ref[:, :, None, :, None, :2]: center points (x,y) [bs, Len_q, 1, n_levels, 1, 2]
            # offsets: [bs, Len_q, n_heads, n_levels, n_points, 2]
            # ref[:, :, None, :, None, 2:]: width/height (w,h) [bs, Len_q, 1, n_levels, 1, 2]
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] # Add head_dim and point_dim
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
            self.num_heads, self.num_kv_heads # Pass head counts
        )
        # output: [bs, Len_q, embed_dim]

        # Final output projection
        output = self.output_proj(output) # [bs, Len_q, embed_dim]

        return output, attention_weights


# Example Usage
if __name__ == '__main__':
    bs, Len_q, C = 2, 100, 256
    n_levels = 4
    n_heads = 8
    # --- GQA Setting ---
    n_kv_heads = 2 # Example: 2 K/V heads shared by 8 Query heads
    # -----------------
    n_points = 4

    # Dummy input tensors
    query = torch.rand(bs, Len_q, C)
    reference_points = torch.rand(bs, Len_q, n_levels, 2) # Example using (x, y) format

    # Generate multi-scale value features and spatial shapes
    value_list = []
    value_spatial_shapes_list = []
    value_len = 0
    current_h, current_w = 32, 32
    for _ in range(n_levels):
        level_len = current_h * current_w
        value_list.append(torch.rand(bs, level_len, C))
        value_spatial_shapes_list.append([current_h, current_w])
        value_len += level_len
        current_h //= 2
        current_w //= 2

    value = torch.cat(value_list, dim=1) # [bs, value_length, C]
    value_spatial_shapes = torch.tensor(value_spatial_shapes_list, dtype=torch.long)
    value_mask = torch.ones(bs, value_len, dtype=torch.bool) # Example: no padding

    print("--- Testing MSDeformableAttention with GQA ---")
    print(f"Query Heads: {n_heads}, K/V Heads: {n_kv_heads}")

    # Instantiate the GQA module
    msda_gqa = MSDeformableAttentionGQA(
        embed_dim=C,
        num_heads=n_heads,
        num_kv_heads=n_kv_heads, # Pass the K/V head count
        num_levels=n_levels,
        num_points=n_points
    )
    msda_gqa.eval()

    # Forward pass
    with torch.no_grad():
        output = msda_gqa(query, reference_points, value, value_spatial_shapes, value_mask)

    print("Input Query Shape:", query.shape)
    print("Input Value Shape:", value.shape)
    print("Input Ref Points Shape:", reference_points.shape)
    print("Value Spatial Shapes:", value_spatial_shapes.tolist())
    print("Output Shape:", output.shape)

    # Verify output shape
    assert output.shape == (bs, Len_q, C)
    print("\nGQA MSDeformableAttention test passed.")

    # Test standard MHA case (num_kv_heads = num_heads)
    print("\n--- Testing MSDeformableAttention GQA with num_kv_heads == num_heads ---")
    msda_mha_via_gqa = MSDeformableAttentionGQA(
        embed_dim=C,
        num_heads=n_heads,
        num_kv_heads=n_heads, # Set kv_heads = heads
        num_levels=n_levels,
        num_points=n_points
    )
    msda_mha_via_gqa.eval()
    with torch.no_grad():
        output_mha = msda_mha_via_gqa(query, reference_points, value, value_spatial_shapes, value_mask)
    print("Output Shape (MHA via GQA):", output_mha.shape)
    assert output_mha.shape == (bs, Len_q, C)
    print("MHA via GQA test passed.")