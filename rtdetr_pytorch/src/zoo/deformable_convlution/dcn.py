import torch
import torch.nn as nn
from mmcv.ops import DeformConv2d as MMCVDeformConv2d

# 检查是否可以使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义一个简单的 Deformable Convolution 层
class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DeformableConv2d, self).__init__()

        # offset 卷积，用于生成偏移量
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,  # 2 表示 x 和 y 方向的偏移
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # 使用 mmcv 的 DeformConv2d
        self.dconv = MMCVDeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            deform_groups=1
        )

    def forward(self, x):
        # 生成偏移量
        offset = self.offset_conv(x)
        # 通过 deformable convolution 前向传播
        out = self.dconv(x, offset)
        return out

# 创建一个简单的网络
class SimpleDCN(nn.Module):
    def __init__(self):
        super(SimpleDCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.dconv = DeformableConv2d(64, 64, 3)
        self.conv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.dconv(x))
        x = self.conv2(x)
        return x

# 测试 DEMO
def main():
    # 创建模型
    model = SimpleDCN().to(device)

    # 创建一个模拟输入 (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 224, 224).to(device)

    # 前向传播
    output = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")

    # 简单训练示例
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 模拟目标
    target = torch.randn(1, 1, 224, 224).to(device)

    # 训练一步
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()