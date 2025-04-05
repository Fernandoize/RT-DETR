import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Optional, Tuple

# LayerNormProxy might not be needed if we primarily use Linear layers
# but keep it for potential future use if adapting conv layers back.
class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class DeformableMHAGQA(nn.Module):
    """
    Multi-Head Attention with Deformable Sampling (inspired by DAttention/MSDA)
    and Grouped Query Attention (GQA), mimicking nn.MultiheadAttention interface.

    Note:
    - Removes specific positional encodings from DAttentionBaselineGQA.
    - Adopts a mechanism where Query predicts sampling locations and attention weights directly.
    - Uses 1D interpolation for sampling from the Value sequence.
    - `key_padding_mask` primarily affects value masking during sampling.
    - `attn_mask` and `is_causal` have limited applicability due to predictive weights.
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 num_kv_heads: Optional[int] = None,
                 num_points: int = 4, # Number of points to sample per head
                 dropout: float = 0.0,
                 bias: bool = True,
                 batch_first: bool = True): # Match MHA default
        super().__init__()
        if not batch_first:
            raise NotImplementedError("batch_first=False is not supported yet.")
        if embed_dim % num_heads != 0:
             raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads # Query heads
        self.num_points = num_points
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads

        # --- GQA Setup ---
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        self.num_kv_heads = num_kv_heads
        self.num_q_per_kv = num_heads // num_kv_heads
        # --- End GQA Setup ---

        self.kv_embed_dim = self.num_kv_heads * self.head_dim # Dimension for V projection

        # --- Layers ---
        # Query projection (standard MHA style, but not strictly needed if offsets/weights based on original query)
        # Let's keep it for consistency and potential use
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # Value projection (reduced dim for GQA)
        self.v_proj = nn.Linear(embed_dim, self.kv_embed_dim, bias=bias)
        # No Key projection needed as weights are predicted directly

        # Layers to predict sampling locations (offsets) and weights from the query
        self.sampling_offset_proj = nn.Linear(embed_dim, num_heads * num_points * 2, bias=True)
        self.attention_weight_proj = nn.Linear(embed_dim, num_heads * num_points, bias=True)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # Standard MHA-like initialization
        init.xavier_uniform_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            init.constant_(self.q_proj.bias, 0.)
        init.xavier_uniform_(self.v_proj.weight)
        if self.v_proj.bias is not None:
            init.constant_(self.v_proj.bias, 0.)
        init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            init.constant_(self.out_proj.bias, 0.)

        # Initialize offset/weight prediction layers
        init.constant_(self.sampling_offset_proj.weight, 0.)
        # Initialize offsets bias to spread points initially? Or zero? Zero is simpler.
        init.constant_(self.sampling_offset_proj.bias, 0.)

        init.constant_(self.attention_weight_proj.weight, 0.)
        # Initialize weights bias uniformly across points
        init.constant_(self.attention_weight_proj.bias, 1.0 / self.num_points)

    def get_reference_points(self, spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        query: torch.Tensor, # (B, Lq, C) if batch_first else (Lq, B, C)
        key: torch.Tensor,   # (B, Lk, C) if batch_first else (Lk, B, C) - Used only for length info
        value: torch.Tensor, # (B, Lv, C) if batch_first else (Lv, B, C) - Assume Lk=Lv
        key_padding_mask: Optional[torch.Tensor] = None, # (B, Lk) - True where padded
        need_weights: bool = True, # Return attention weights?
        attn_mask: Optional[torch.Tensor] = None, # Not effectively used
        average_attn_weights: bool = True, # Average weights across heads?
        is_causal : bool = False, # Not applicable
        spatial_shape: torch.Tensor = None, # Not used
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if not self.batch_first:
             query = query.permute(1, 0, 2)
             key = key.permute(1, 0, 2)
             value = value.permute(1, 0, 2)

        B, Lq, C = query.shape
        _ , Lk, _ = key.shape
        _ , Lv, C_v = value.shape
        assert C == self.embed_dim
        assert C_v == self.embed_dim
        assert Lk == Lv, "Key and Value sequence lengths must match for this implementation"

        # 1. Project Value (GQA)
        # value: (B, Lv, C) -> v_proj -> (B, Lv, kv_embed_dim)
        v = self.v_proj(value)
        # Reshape for GQA heads: (B, Lv, Nkv, C/N)
        v = v.view(B, Lv, self.num_kv_heads, self.head_dim)

        # --- GQA Adaptation: Repeat K/V heads for sampling ---
        # v: (B, Lv, Nkv, C/N) -> repeat -> (B, Lv, Nq, C/N)
        v_repeat = v.repeat_interleave(self.num_q_per_kv, dim=2)
        # Reshape for grid_sample input:
        # (B, Lv, Nq, C/N) -> (B, Nq, C/N, Lv) -> (B*Nq, C/N, 1, Lv)
        v_repeat = v_repeat.permute(0, 2, 3, 1).reshape(B * self.num_heads, self.head_dim, 1, Lv)
        # ------------------------------------------------------

        # 2. Predict Offsets and Attention Weights from Query
        # Use original query features
        # offsets: (B, Lq, C) -> proj -> (B, Lq, Nq*Np) -> (B, Lq, Nq, Np)
        sampling_offsets = self.sampling_offset_proj(query).view(B, Lq, self.num_heads, self.num_points, 2)
        # weights: (B, Lq, C) -> proj -> (B, Lq, Nq*Np) -> (B, Lq, Nq, Np)
        attention_weights = self.attention_weight_proj(query).view(B, Lq, self.num_heads, self.num_points)
        # Softmax weights over points dimension
        attention_weights = F.softmax(attention_weights, dim=-1)
        # Shape: (B, Lq, Nq, Np)

        # 3. Calculate Sampling Locations (1D)
        # Reference points: centers of the Lq query tokens, normalized to [0, 1]
        ref_points_q = torch.linspace(0.5 / Lq, 1.0 - 0.5 / Lq, Lq, dtype=query.dtype, device=query.device)
        ref_points_q = ref_points_q.view(1, Lq, 1, 1).expand(B, -1, self.num_heads, 1)

        # Normalize offsets by sequence length Lk (or Lv)
        # Add sigmoid? Tanh? Let's just scale for now. Assume offsets are small relative shifts.
        normalized_offsets = sampling_offsets / Lk
        # Calculate sampling locations in [0, 1] range
        sampling_locations = (ref_points_q + normalized_offsets) # Shape: (B, Lq, Nq, Np)
        # Clamp to avoid out-of-bounds
        sampling_locations = sampling_locations.clamp(min=0.0, max=1.0)

        # 4. Prepare for 1D interpolation (grid_sample)
        # grid_sample expects grid in [-1, 1] range and shape (N, H_out, W_out, 2)
        # Our "spatial" dim is Lv (W), H=1.
        # N = B * Nq, H_out = Lq, W_out = Np
        # locations: (B, Lq, Nq, Np)
        # -> (B, Nq, Lq, Np)
        sampling_locations = sampling_locations.permute(0, 2, 1, 3)
        # -> (B*Nq, Lq, Np)
        sampling_locations = sampling_locations.reshape(B * self.num_heads, Lq, self.num_points)

        # Convert locations [0, 1] to grid [-1, 1] for the 'x' coordinate (sequence dim)
        grid_x = 2.0 * sampling_locations - 1.0
        # Set 'y' coordinate to 0 (since H=1)
        grid_y = torch.zeros_like(grid_x)
        # Stack to form grid: (B*Nq, Lq, Np, 2)
        grid = torch.stack((grid_x, grid_y), dim=-1)

        # 5. Perform Deformable Sampling (1D Interpolation)
        # Input v_repeat: (B*Nq, C/N, 1, Lv)
        # Grid: (B*Nq, Lq, Np, 2)
        # Output: (B*Nq, C/N, Lq, Np)
        sampled_v = F.grid_sample(v_repeat, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        # Handle key_padding_mask by zeroing out sampled values from padded locations?
        # This is complex as sampling is interpolated. `padding_mode='zeros'` helps.
        # A more direct mask could be applied to attention_weights *before* softmax,
        # but we need to know which points sample from padded areas, which is hard.
        # Let's rely on padding_mode='zeros' for now.

        # 6. Apply Attention Weights
        # attention_weights: (B, Lq, Nq, Np) -> (B, Nq, Lq, Np) -> (B*Nq, Lq, Np)
        attn_w = attention_weights.permute(0, 2, 1, 3).reshape(B * self.num_heads, Lq, self.num_points)
        # -> (B*Nq, 1, Lq, Np) to broadcast with sampled_v
        attn_w = attn_w.unsqueeze(1)

        # Output = sum(attn_w * sampled_v) over points dim
        # sampled_v: (B*Nq, C/N, Lq, Np)
        # attn_w:    (B*Nq, 1,   Lq, Np)
        # product:   (B*Nq, C/N, Lq, Np)
        # sum:       (B*Nq, C/N, Lq)
        output = (attn_w * sampled_v).sum(dim=-1)

        # 7. Reshape and Project Output
        # output: (B*Nq, C/N, Lq) -> (B, Nq, C/N, Lq) -> (B, Nq*C/N, Lq) -> (B, C, Lq)
        output = output.view(B, self.num_heads, self.head_dim, Lq).permute(0, 3, 1, 2).reshape(B, Lq, C)
        # -> (B, Lq, C) -> out_proj -> (B, Lq, C)
        output = self.out_proj(output)
        output = self.dropout(output)

        # Handle `need_weights` and `average_attn_weights`
        # Note: these weights are *predicted*, not calculated from QK similarity
        attn_weights_output = None
        if need_weights:
            # attention_weights shape: (B, Lq, Nq, Np)
            attn_weights_output = attention_weights # Return per-head weights
            if average_attn_weights:
                attn_weights_output = attn_weights_output.mean(dim=2) # Avg over query heads -> (B, Lq, Np)

        # Final permutation if batch_first=False was requested (but we blocked it)
        # if not self.batch_first:
        #    output = output.permute(1, 0, 2)

        # attn_mask / is_causal are not used effectively here
        if attn_mask is not None:
             print("Warning: attn_mask is not effectively used in DeformableMHAGQA.")
        if is_causal:
             print("Warning: is_causal is not applicable in DeformableMHAGQA.")

        return output, attn_weights_output

# Example Usage:
if __name__ == '__main__':
    B, Lq, Lk, C = 2, 10, 12, 96 # Example dimensions
    n_heads = 8
    n_kv_heads = 2 # GQA
    n_points = 4

    query = torch.randn(B, Lq, C)
    key = torch.randn(B, Lk, C)
    value = torch.randn(B, Lk, C) # Lk = Lv
    padding_mask = torch.zeros(B, Lk, dtype=torch.bool) # Example: no padding
    padding_mask[:, -2:] = True # Pad last two key/value tokens

    print("--- Testing DeformableMHAGQA ---")
    attn = DeformableMHAGQA(
        embed_dim=C,
        num_heads=n_heads,
        num_kv_heads=n_kv_heads,
        num_points=n_points,
        dropout=0.1,
        batch_first=True
    )
    attn.eval()

    output, attn_weights = attn(
        query, key, value,
        key_padding_mask=padding_mask,
        need_weights=True,
        average_attn_weights=False
    )

    output_avg_weights, attn_weights_avg = attn(
        query, key, value,
        key_padding_mask=padding_mask,
        need_weights=True,
        average_attn_weights=True
    )


    print("Query shape:", query.shape)
    print("Key shape:", key.shape)
    print("Value shape:", value.shape)
    print("Output shape:", output.shape)

    assert output.shape == (B, Lq, C)

    if attn_weights is not None:
        print("Attn weights shape (per head):", attn_weights.shape) # Should be (B, Lq, Nq, Np)
        assert attn_weights.shape == (B, Lq, n_heads, n_points)

    if attn_weights_avg is not None:
        print("Attn weights shape (averaged):", attn_weights_avg.shape) # Should be (B, Lq, Np)
        assert attn_weights_avg.shape == (B, Lq, n_points)

    print("\nDeformableMHAGQA test passed.")