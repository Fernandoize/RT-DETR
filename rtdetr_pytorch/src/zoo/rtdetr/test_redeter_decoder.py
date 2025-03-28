import unittest

import torch
from src.zoo.rtdetr.rtdetr_decoder import MSDeformableAttention

class TestRtDetrDecoder(unittest.TestCase):

    # 思路1: 使用select topk query， 在注意力机制中，通常会生成大量的查询（query），但并非所有查询都对最终结果有显著贡献。
    # 通过选择Top-K个最重要的查询（例如根据得分或权重排序），可以减少计算量并聚焦于关键信息；提高计算效率：只保留最重要的K个查询，减少不必要的计算。
    # 提升模型性能：专注于最相关的特征，可能有助于提升模型的准确性和鲁棒性。

    # 思路2: 对query 进行分组 增强模型表达能力：不同组的查询可以关注不同的特征或区域，从而捕获更丰富的信息；
    # 减少冗余计算：通过分组，避免所有查询同时处理相同的信息，降低计算冗余。

    # 思路3: 动态权重, 使得模型能够根据输出数据自适应的调整注意力分布
    # 思路4: 参考点的生成方式，使用预测网络，结合任务的特定信息，提高网络收敛的效率

    # 思路5: 局部注意力和全局注意力的结合，在局部使用形变注意力; 在全局范围使用标准注意力，提高模型对复杂场景的理解

    def test_deformable_attention(self):
        # 初始化 MSDeformableAttention
        attn = MSDeformableAttention(embed_dim=256, num_heads=8, num_levels=4, num_points=4)

        # 定义输入张量
        batch_size = 1
        query_length = 100
        value_length = 1024
        embed_dim = 256
        n_levels = 4
        n_points = 4

        # 查询张量，形状为 (batch_size, query_length, embed_dim)
        query = torch.randn(batch_size, query_length, embed_dim)

        # 参考点张量，形状为 (batch_size, query_length, n_levels, 2)，范围在 [0, 1] 之间
        reference_points = torch.rand(batch_size, query_length, n_levels, 2)

        # 值张量，形状为 (batch_size, value_length, embed_dim)
        value = torch.randn(batch_size, value_length, embed_dim)

        # 每个层级的空间形状，形状为 (n_levels, 2)
        value_spatial_shapes = torch.tensor([[32, 32], [16, 16], [8, 8], [4, 4]], dtype=torch.long)

        # 标记出有效的value，形状为 (batch_size, value_length)，类型为 bool
        value_mask = None

        # 计算输出
        output = attn(query, reference_points, value, value_spatial_shapes, value_mask)

        # 验证输出形状
        self.assertEqual(output.shape, (1, 256, 100))




