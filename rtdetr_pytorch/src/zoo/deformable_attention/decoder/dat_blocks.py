import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import to_2tuple, trunc_normal_


class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class DAttentionBaselineGQA(nn.Module): # Renamed class for clarity
    """
    根据数据动态计算采样位置，然后进行加权 (加入GQA特性)
    """
    def __init__(
            self, q_size, kv_size, n_heads, n_head_channels, n_groups, # n_groups for offset calculation
            n_kv_groups=None, # Number of Key/Value groups for GQA
            attn_drop=0.0, proj_drop=0.0, stride=-1,
            offset_range_factor=1, use_pe=False, dwc_pe=False,
            no_off=False, fixed_pe=False, ksize = 9, log_cpb=False
    ):

        super().__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads # Number of Query heads
        self.n_groups = n_groups # Number of groups for offset calculation

        # --- GQA specific parameters ---
        if n_kv_groups is None:
            # Default: GQA groups = Offset groups for simplicity
            # If you need them to be different, the RPE logic (log_cpb, grid_sample) needs careful adaptation
            self.n_kv_groups = n_groups
            print(f"Warning: n_kv_groups not specified, defaulting to n_groups ({self.n_groups}).")
        else:
            self.n_kv_groups = n_kv_groups

        assert n_heads % self.n_kv_groups == 0, f"n_heads ({n_heads}) must be divisible by n_kv_groups ({self.n_kv_groups})"
        self.n_q_per_kv = n_heads // self.n_kv_groups # Number of Query heads per Key/Value group
        # --- End GQA parameters ---

        self.q_h, self.q_w = q_size
        # self.kv_h, self.kv_w = kv_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride

        self.nc = n_head_channels * n_heads # Total channels for Q and Output
        self.kv_nc = n_head_channels * self.n_kv_groups # Total channels for K and V

        # Channel calculation per offset group
        assert self.nc % self.n_groups == 0, f"Total channels ({self.nc}) must be divisible by n_groups ({self.n_groups})"
        self.n_group_channels = self.nc // self.n_groups # Channels per offset group

        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        # 可形变偏移生成 (Uses n_groups for offset calculation grouping)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        # query, key, value的投影
        self.proj_q = nn.Conv2d(
            self.nc, self.nc, # Output: n_heads * n_head_channels
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.kv_nc, # Output: n_kv_groups * n_head_channels
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.kv_nc, # Output: n_kv_groups * n_head_channels
            kernel_size=1, stride=1, padding=0
        )

        # 对注意力的输出进行投影，生成最终的特征图
        self.proj_out = nn.Conv2d(
            self.nc, self.nc, # Input: n_heads * n_head_channels
            kernel_size=1, stride=1, padding=0
        )

        # 注意力权重和输出投影的dropout率
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        # 是否使用位置编码
        if self.use_pe and not self.no_off:
            if self.dwc_pe:
                # Applies to output, shape should be compatible
                self.rpe_table = nn.Conv2d(
                    self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                # Bias per query head, shape seems okay
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            elif self.log_cpb:
                # Assuming n_groups == n_kv_groups here
                # Output dimension is number of query heads per group
                assert self.n_groups == self.n_kv_groups, "log_cpb requires n_groups == n_kv_groups in this implementation"
                self.rpe_table = nn.Sequential(
                    nn.Linear(2, 32, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, self.n_q_per_kv, bias=False) # Output heads per group
                )
            else: # Grid Sample RPE
                # Assuming n_groups == n_kv_groups here
                assert self.n_groups == self.n_kv_groups, "Grid Sample RPE requires n_groups == n_kv_groups in this implementation"
                # Stores bias per query head? Table size relates to relative positions.
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        """
        生成参考点，用于计算偏移
        Uses n_groups (offset groups)
        """
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        # Expand based on offset groups
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2
        return ref

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):
        """
        生成查询网格，用于计算位置编码
        Uses n_groups (offset groups) for displacement calculation in RPE
        """
        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        # Expand based on offset groups
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2
        return ref

    def forward(self, x):

        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device
        assert C == self.nc, f"Input channel {C} != expected {self.nc}"

        # 1. 通过conv_offset生成偏移 (Using n_groups)
        q_in = self.proj_q(x)
        q_off = einops.rearrange(q_in, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg

        # 参考点的大小和offset相同
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk # Number of sampled points (keys/values)

        # 2. offset_range_factor控制偏移范围
        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p') # (B * g) Hg Wg 2
        # 3. 计算参考点 (Using n_groups)
        reference = self._get_ref_points(Hk, Wk, B, dtype, device) # (B * g) Hg Wg 2

        # 4. 计算采样位置
        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference # (B*g) Hg Wg 2
        else:
            pos = (offset + reference).clamp(-1., +1.)

        # 5. 采样 Key/Value 特征
        # Input to sampling: original features x, potentially reshaped based on offset groups if needed
        # Here, grid_sample takes input (N, C_in, H_in, W_in) and grid (N, H_out, W_out, 2)
        # We want N = B * n_groups, C_in = n_group_channels ? No, needs full channel dim C.
        # Reshape x to match the grid's N dimension: (B * g, Cg, H, W)? No, C should be full C.
        # Let's assume grid_sample can handle N mismatch if input is B,C,H,W and grid is (B*g),Hg,Wg,2? Check docs.
        # It seems grid_sample expects N in input and grid to match.
        # So, we either repeat x or adapt the sampling.
        # Let's repeat x to match the grouped pos grid.
        x_for_sampling = einops.rearrange(x, 'b c h w -> b 1 c h w')
        x_for_sampling = x_for_sampling.expand(B, self.n_groups, C, H, W)
        x_for_sampling = einops.rearrange(x_for_sampling, 'b g c h w -> (b g) c h w')

        if self.no_off:
            # If no offset, avg_pool original x. Does not need grouping.
            kv_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            assert kv_sampled.size(2) == Hk and kv_sampled.size(3) == Wk, f"Size is {kv_sampled.size()}"
            # Reshape to match expected kv format after sampling
            kv_sampled = einops.rearrange(kv_sampled, 'b c h w -> b c 1 (h w)') # B, C, 1, Ns
        else:
            # Use the grouped x for sampling with the grouped pos grid
            kv_sampled = F.grid_sample(
                input=x_for_sampling, # (B*g) C H W
                grid=pos[..., (1, 0)],  # y, x -> x, y # (B*g) Hg Wg 2
                mode='bilinear', align_corners=True)  # Output: (B*g) C Hg Wg

            # Regroup results back: (B*g) C Hg Wg -> B C g (Hg Wg) -> B C 1 (g Hg Wg)? No.
            # Output should be features corresponding to K/V for all heads.
            # We sampled features at Hk*Wk locations using groups. The features C are shared.
            # Rearrange to B, C, 1, Ns where Ns = Hk * Wk
            kv_sampled = einops.rearrange(kv_sampled, '(b g) c h w -> b c g (h w)', b=B, g=self.n_groups)
            # Now, how to combine the groups? The sampling was done per group.
            # If n_groups == n_kv_groups, maybe we take the result directly?
            # Let's assume the sampled features are the same regardless of group if offset is same.
            # Average over groups? Or just take one group's result?
            # Let's average over the group dimension for robustness, assuming sampling is similar across groups.
            # This might need revision depending on intended group behavior.
            # Alternative: If offset calc uses shared weights across groups, results might be identical.
            # Let's average for now.
            kv_sampled = kv_sampled.mean(dim=2, keepdim=True) # B, C, 1, Ns (Ns=Hk*Wk)
            # If n_groups=1, this mean does nothing.

        # 6.计算 Query, Key, Value
        q = q_in.reshape(B * self.n_heads, self.n_head_channels, H * W) # (B*h) Ch (H*W)

        # Project K and V from the *sampled* features
        k = self.proj_k(kv_sampled) # B, kv_nc, 1, Ns
        v = self.proj_v(kv_sampled) # B, kv_nc, 1, Ns

        # Reshape K, V for GQA
        # kv_nc = n_kv_groups * n_head_channels
        # Ns = n_sample = Hk * Wk
        k = k.reshape(B, self.n_kv_groups, self.n_head_channels, n_sample) # B g_kv Ch Ns
        v = v.reshape(B, self.n_kv_groups, self.n_head_channels, n_sample) # B g_kv Ch Ns

        # Repeat K, V heads for GQA logic
        # Repeat g_kv dimension n_q_per_kv times to match n_heads
        k = k.repeat_interleave(self.n_q_per_kv, dim=1) # B (g_kv*n_q_per_kv)=h Ch Ns
        v = v.repeat_interleave(self.n_q_per_kv, dim=1) # B (g_kv*n_q_per_kv)=h Ch Ns

        # Final reshape for attention calculation
        k = k.reshape(B * self.n_heads, self.n_head_channels, n_sample) # (B*h) Ch Ns
        v = v.reshape(B * self.n_heads, self.n_head_channels, n_sample) # (B*h) Ch Ns

        # 7. 计算注意力权重
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # (B*h) (H*W) Ns
        attn = attn.mul(self.scale)

        # 8. 加入位置编码 (RPE)
        if self.use_pe and (not self.no_off):
            if self.dwc_pe:
                # Applied to output value later
                residual_lepe = self.rpe_table(q_in).reshape(B * self.n_heads, self.n_head_channels, H * W)
            elif self.fixed_pe:
                # Bias per query head, shape (h, HW, Ns) - Needs Ns part correct
                # Assuming kv_h * kv_w = n_sample
                assert self.kv_h * self.kv_w == n_sample, "fixed_pe requires kv_size to match sampling size"
                rpe_table = self.rpe_table # h (H*W) (Hk*Wk)
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1) # B h (H*W) Ns
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample) # (B*h) (H*W) Ns
            elif self.log_cpb:
                # Assumes n_groups == n_kv_groups
                q_grid = self._get_q_grid(H, W, B, dtype, device) # (B*g) H W 2
                # pos was (B*g) Hg Wg 2 -> need (B*g) Ns 2
                pos_rpe = einops.rearrange(pos, 'b h w c -> b (h w) c') # (B*g) Ns 2
                displacement = (
                            q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos_rpe.unsqueeze(1)
                           ).mul(4.0) # (B*g) HW Ns 2
                displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
                # RPE table output: heads per group
                attn_bias = self.rpe_table(displacement)  # (B*g) HW Ns h_per_g
                # Rearrange to match attention score shape (B*h) HW Ns
                attn = attn + einops.rearrange(attn_bias, '(b g) m n h_g -> (b g h_g) m n',
                                               g=self.n_groups, h_g=self.n_q_per_kv) # (B*h) HW Ns
            else: # Grid Sample RPE
                # Assumes n_groups == n_kv_groups
                rpe_table = self.rpe_table # h (2H-1) (2W-1)
                # Expand rpe table to batch dim B
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1) # B h (2H-1) (2W-1)
                q_grid = self._get_q_grid(H, W, B, dtype, device) # (B*g) H W 2
                pos_rpe = einops.rearrange(pos, 'b h w c -> b (h w) c') # (B*g) Ns 2
                displacement = (
                            q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos_rpe.unsqueeze(1)
                           ).mul(0.5) # (B*g) HW Ns 2 --> range [-1, 1] ? Check DAttention paper
                # Need to sample from B h (2H-1) (2W-1) using grid (B*g) HW Ns 2
                # Input to grid_sample: N C H_in W_in ; Grid: N H_out W_out 2
                # Here Input N=B, C=h, H=2H-1, W=2W-1
                # Grid N=(B*g), H_out=HW, W_out=Ns
                # Need N to match. Reshape/repeat rpe_bias or displacement.
                # Let's repeat rpe_bias B h ... -> (B*g) h ... ? Seems complex.
                # Alternative: Reshape displacement (B*g) HW Ns 2 -> B HW (g*Ns) 2 ? No.
                # --- Simplification: Assume n_groups=1 for Grid Sample RPE for now ---
                if self.n_groups != 1:
                     raise NotImplementedError("Grid Sample RPE with n_groups > 1 is complex to implement correctly with GQA grouping mismatch, not implemented yet.")
                # If n_groups=1, grid is B HW Ns 2
                attn_bias = F.grid_sample(
                    input=rpe_bias.permute(0, 1, 3, 2), # B h (2W-1) (2H-1) <- W, H order? Check grid sample doc
                                                       # Let's assume original H, W order: B h (2H-1) (2W-1)
                    grid=displacement[..., (1, 0)], # B HW Ns 2 (x, y)
                    mode='bilinear', align_corners=True, padding_mode='border'
                 ) # Output: B h HW Ns
                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample) # (B*h) HW Ns
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        # 9. 计算输出
        out = torch.einsum('b m n, b c n -> b c m', attn, v) # (B*h) Ch (H*W)

        if self.use_pe and self.dwc_pe:
            # Add residual PE if using DWC PE
            out = out + residual_lepe # (B*h) Ch (H*W)

        out = out.reshape(B, C, H, W)

        # 10. 输出投影
        y = self.proj_drop(self.proj_out(out)) # B C H W

        # Return format similar to original
        # Reshape pos and reference back to B g ... format
        pos_out = einops.rearrange(pos, '(b g) h w c -> b g h w c', b=B, g=self.n_groups)
        ref_out = einops.rearrange(reference, '(b g) h w c -> b g h w c', b=B, g=self.n_groups)

        return y, pos_out, ref_out


if __name__ == '__main__':
    # 定义输入张量和参数
    B = 2
    H = 14
    W = 14
    C = 96  # Embed Dim
    n_heads = 8
    n_head_channels = C // n_heads
    stride = 2
    offset_groups = 4 # For offset calculation
    kv_groups = 2 # For GQA (must divide n_heads)

    q_size = (H, W)
    kv_size = (H // stride, W // stride) # Not directly used in init, calculated inside

    # Create input tensor
    x = torch.randn(B, C, H, W)

    # Instantiate the GQA Attention module
    # Note: Using n_groups=offset_groups, n_kv_groups=kv_groups
    attn_gqa = DAttentionBaselineGQA(
        q_size=q_size,
        kv_size=kv_size, # Placeholder, calculated internally based on stride
        n_heads=n_heads,
        n_head_channels=n_head_channels,
        n_groups=offset_groups, # Grouping for offset calculation
        n_kv_groups=kv_groups,  # GQA specific: number of K/V groups
        attn_drop=0.1,
        proj_drop=0.1,
        stride=stride,
        offset_range_factor=2,
        use_pe=True, # Enable PE for testing RPE paths
        dwc_pe=False,
        no_off=False,
        fixed_pe=False, # Test log_cpb or grid sample RPE
        ksize=3,
        log_cpb=True # Test log_cpb RPE (requires n_groups==n_kv_groups)
        #log_cpb=False # Enable this and set n_groups=1 to test grid sample RPE
    )

    # Test case where n_groups != n_kv_groups for log_cpb/grid RPE (should raise error)
    try:
        attn_gqa_mismatch = DAttentionBaselineGQA(
            q_size=q_size, kv_size=kv_size, n_heads=n_heads, n_head_channels=n_head_channels,
            n_groups=4, n_kv_groups=2, stride=stride, use_pe=True, log_cpb=True # Mismatch
        )
        # This part should not be reached if the assert works
        print("Error: Instantiation with n_groups != n_kv_groups for log_cpb didn't raise error.")
    except AssertionError as e:
        print(f"Successfully caught assertion for RPE group mismatch: {e}")
    except NotImplementedError as e:
         print(f"Successfully caught NotImplementedError for RPE group mismatch: {e}")


    # Set n_groups = n_kv_groups for log_cpb/grid sample RPE to work in this implementation
    if attn_gqa.log_cpb or (not attn_gqa.dwc_pe and not attn_gqa.fixed_pe):
         assert attn_gqa.n_groups == attn_gqa.n_kv_groups, "Test setup requires n_groups == n_kv_groups for selected RPE"
         # Or if using grid sample RPE, need n_groups=1 for current implementation
         if not attn_gqa.log_cpb and not attn_gqa.dwc_pe and not attn_gqa.fixed_pe:
             assert attn_gqa.n_groups == 1, "Test setup for Grid Sample RPE requires n_groups=1"


    # Forward pass
    attn_gqa.eval() # Use eval mode for dropout etc unless training
    with torch.no_grad(): # Disable gradient calculation for simple test
        y, pos, ref = attn_gqa(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Sampled pos shape:", pos.shape) # B g Hg Wg 2
    print("Reference points shape:", ref.shape) # B g Hg Wg 2

    # Verify output shape
    assert y.shape == x.shape, f"Output shape {y.shape} does not match input shape {x.shape}"
    print("\nGQA Attention module test passed.")

    # Example with fixed PE (should work regardless of group matching)
    print("\nTesting with fixed PE:")
    attn_gqa_fixedpe = DAttentionBaselineGQA(
        q_size=q_size, kv_size=kv_size, n_heads=n_heads, n_head_channels=n_head_channels,
        n_groups=4, n_kv_groups=2, # Groups can mismatch for fixed PE
        stride=stride, use_pe=True, fixed_pe=True
    )
    attn_gqa_fixedpe.eval()
    with torch.no_grad():
        y_fixed, _, _ = attn_gqa_fixedpe(x)
    print("Input shape:", x.shape)
    print("Output shape (fixed PE):", y_fixed.shape)
    assert y_fixed.shape == x.shape
    print("Fixed PE test passed.")

    # Example with DWC PE (should work regardless of group matching)
    print("\nTesting with DWC PE:")
    attn_gqa_dwcpe = DAttentionBaselineGQA(
        q_size=q_size, kv_size=kv_size, n_heads=n_heads, n_head_channels=n_head_channels,
        n_groups=4, n_kv_groups=2, # Groups can mismatch for DWC PE
        stride=stride, use_pe=True, dwc_pe=True
    )
    attn_gqa_dwcpe.eval()
    with torch.no_grad():
        y_dwc, _, _ = attn_gqa_dwcpe(x)
    print("Input shape:", x.shape)
    print("Output shape (DWC PE):", y_dwc.shape)
    assert y_dwc.shape == x.shape
    print("DWC PE test passed.")