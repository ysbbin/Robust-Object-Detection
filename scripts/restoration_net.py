"""
Lightweight U-Net for image restoration (denoising / deblurring / super-resolution).

Architecture:
  - Encoder: 4 downsample blocks [32, 64, 128, 256]
  - Bottleneck: 256 channels
  - Decoder: 4 upsample blocks with skip connections
  - Residual learning: output = input + learned_residual

Fully convolutional — accepts any input size at inference.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.conv(x)
        return self.pool(feat), feat  # downsampled, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from odd dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear",
                                          align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class RestorationUNet(nn.Module):
    """
    Lightweight U-Net with residual learning for image restoration.
    output = input + model(input)  (learns the residual/correction)
    """

    def __init__(self, channels=(32, 64, 128, 256)):
        super().__init__()
        c1, c2, c3, c4 = channels

        # Encoder
        self.down1 = DownBlock(3, c1)    # -> c1, skip c1
        self.down2 = DownBlock(c1, c2)   # -> c2, skip c2
        self.down3 = DownBlock(c2, c3)   # -> c3, skip c3
        self.down4 = DownBlock(c3, c4)   # -> c4, skip c4

        # Bottleneck
        self.bottleneck = ConvBlock(c4, c4)  # c4 -> c4

        # Decoder
        self.up4 = UpBlock(c4, c4, c3)   # c4 + skip c4 -> c3
        self.up3 = UpBlock(c3, c3, c2)   # c3 + skip c3 -> c2
        self.up2 = UpBlock(c2, c2, c1)   # c2 + skip c2 -> c1
        self.up1 = UpBlock(c1, c1, c1)   # c1 + skip c1 -> c1

        # Output: residual
        self.out_conv = nn.Conv2d(c1, 3, 1)

    def forward(self, x):
        # Encoder
        d1, s1 = self.down1(x)
        d2, s2 = self.down2(d1)
        d3, s3 = self.down3(d2)
        d4, s4 = self.down4(d3)

        # Bottleneck
        b = self.bottleneck(d4)

        # Decoder
        u4 = self.up4(b, s4)
        u3 = self.up3(u4, s3)
        u2 = self.up2(u3, s2)
        u1 = self.up1(u2, s1)

        # Residual learning
        residual = self.out_conv(u1)
        return torch.clamp(x + residual, 0.0, 1.0)
