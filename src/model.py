
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        return x + residual * 0.2


class CNNWienerDeblur(nn.Module):
    def __init__(self, features=64, num_blocks=8):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(6, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(features) for _ in range(num_blocks)]
        )

        self.output_layer = nn.Conv2d(
            features, 3, kernel_size=3, padding=1
        )

        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, blur, wiener):
        x = torch.cat([blur, wiener], dim=1)

        x = self.input_layer(x)
        x = self.residual_blocks(x)

        correction = self.output_layer(x)

        output = blur + correction

        return torch.clamp(output, 0.0, 1.0)
