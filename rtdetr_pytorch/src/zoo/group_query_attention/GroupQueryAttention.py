import math
from typing import Optional

import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_
from torchtune.modules import attention as gqa, KVCache

class GroupQueryAttention(nn.Module):
    def __init__(self, *args, embed_dim: int, num_heads: int, num_kv_heads: int,
                 pos_embeddings: Optional[nn.Module] = None, q_norm: Optional[nn.Module] = None,
                 k_norm: Optional[nn.Module] = None, kv_cache: Optional[KVCache] = None, max_seq_len: int = 4096,
                 is_causal: bool = True, attn_dropout: float = 0.0, device=None, dtype=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        factory_kwargs = {"device": device, "dtype": dtype}
        # 将 q_proj, k_proj, v_proj 转换为 nn.Linear 模块
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)
        self._reset_parameters()
        self.gqa = gqa.MultiHeadAttention(embed_dim=embed_dim,
                         num_heads=num_heads,
                         num_kv_heads=num_kv_heads,
                         head_dim=self.head_dim,
                         q_proj=self.q_proj,
                         k_proj=self.k_proj,
                         v_proj=self.v_proj,
                         output_proj=self.out_proj,
                         attn_dropout=attn_dropout)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.constant_(self.k_proj.bias, 0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0)


    def forward(self, **kwargs):
        self.gqa.forward(**kwargs)