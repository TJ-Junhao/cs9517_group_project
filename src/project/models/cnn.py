from typing import Any

import torch
import torch.nn as nn


class UNet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, channel_in: int, channel_out: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(
                channel_in, channel_out, kernel_size=3, stride=1, padding=1
            )
            self.conv2 = nn.Conv2d(
                channel_out, channel_out, kernel_size=3, stride=1, padding=1
            )
            self.activation = nn.ReLU(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.activation(self.conv1(x))
            x = self.activation(self.conv2(x))
            return x

    class EncoderBlock(nn.Module):
        def __init__(
            self, channel_in: int, channel_out: int, stride_downsampling: bool
        ) -> None:
            super().__init__()
            self.do_stride = stride_downsampling
            self.down = (
                nn.Conv2d(channel_in, channel_in, kernel_size=3, stride=2, padding=1)
                if stride_downsampling
                else nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.conv = UNet.ConvBlock(channel_in, channel_out)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(self.down(x))

    class DecoderBlock(nn.Module):
        def __init__(self, channel_in: int, channel_out: int) -> None:
            super().__init__()
            self.up = nn.ConvTranspose2d(
                channel_in, channel_out, kernel_size=2, stride=2
            )
            self.conv = UNet.ConvBlock(channel_out * 2, channel_out)

        def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
            x = self.up(x)
            x = torch.cat((x, skip), dim=1)
            return self.conv(x)

    def __init__(self, channel_in: int, stride: bool = False) -> None:
        super().__init__()

        self.enc1 = self.ConvBlock(channel_in, 64)
        self.enc2 = self.EncoderBlock(64, 128, stride_downsampling=stride)
        self.enc3 = self.EncoderBlock(128, 256, stride_downsampling=stride)
        self.enc4 = self.EncoderBlock(256, 512, stride_downsampling=stride)
        self.enc5 = self.EncoderBlock(512, 1024, stride_downsampling=stride)

        self.dec1 = self.DecoderBlock(1024, 512)
        self.dec2 = self.DecoderBlock(512, 256)
        self.dec3 = self.DecoderBlock(256, 128)
        self.dec4 = self.DecoderBlock(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        x6 = self.dec1(x5, x4)
        x7 = self.dec2(x6, x3)
        x8 = self.dec3(x7, x2)
        x9 = self.dec4(x8, x1)

        return self.out(x9)


class ResUNet(nn.Module):
    class ResidualEncoderBlock(nn.Module):
        def __init__(self, channel_in: int, channel_out: int, stride: int = 1) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(
                channel_in,
                channel_out,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.conv2 = nn.Conv2d(
                channel_out, channel_out, kernel_size=3, stride=1, padding=1, bias=False
            )

            self.gn1 = nn.GroupNorm(num_groups=8, num_channels=channel_out)
            self.gn2 = nn.GroupNorm(num_groups=8, num_channels=channel_out)
            # self.alpha = nn.Parameter(torch.ones(1, channel_out, 1, 1) * 0.01)
            self.activation = nn.ReLU(inplace=True)

            self.proj = (
                nn.Identity()
                if channel_in == channel_out and stride == 1
                else nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=stride)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            identity = self.proj(x)
            out = self.activation(self.gn1(self.conv1(x)))
            out = self.gn2(self.conv2(out))
            return self.activation(out + identity)

    class ResidualDecoderBlock(nn.Module):
        def __init__(self, channel_in, channel_out) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(
                channel_in, channel_out, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.conv2 = nn.Conv2d(
                channel_out, channel_out, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.proj = (
                nn.Identity()
                if channel_in == channel_out
                else nn.Conv2d(channel_in, channel_out, kernel_size=1)
            )
            self.up_conv = nn.ConvTranspose2d(
                channel_in, channel_out, kernel_size=2, stride=2
            )
            self.gn1 = nn.GroupNorm(num_groups=8, num_channels=channel_out)
            self.gn2 = nn.GroupNorm(num_groups=8, num_channels=channel_out)
            # self.alpha = nn.Parameter(torch.ones(1, channel_out, 1, 1) * 0.1)
            self.activation = nn.ReLU(inplace=True)

        def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            x = torch.cat((self.up_conv(x1), x2), dim=1)
            identity = self.proj(x)
            out = self.activation(self.gn1(self.conv1(x)))
            out = self.gn2(self.conv2(out))
            return self.activation(out + identity)

    def __init__(self, channel_in: int) -> None:
        super().__init__()
        self.enc1 = self.ResidualEncoderBlock(channel_in, 64, stride=1)
        self.enc2 = self.ResidualEncoderBlock(64, 128, stride=2)
        self.enc3 = self.ResidualEncoderBlock(128, 256, stride=2)
        self.enc4 = self.ResidualEncoderBlock(256, 512, stride=2)
        self.enc5 = self.ResidualEncoderBlock(512, 1024, stride=2)

        self.dec1 = self.ResidualDecoderBlock(1024, 512)
        self.dec2 = self.ResidualDecoderBlock(512, 256)
        self.dec3 = self.ResidualDecoderBlock(256, 128)
        self.dec4 = self.ResidualDecoderBlock(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        x6 = self.dec1(x5, x4)
        x7 = self.dec2(x6, x3)
        x8 = self.dec3(x7, x2)
        x9 = self.dec4(x8, x1)

        return self.out(x9)


class ASPPResUNet(nn.Module):
    class ASPPBlock(nn.Module):
        def __init__(self, channel_in: int, channel_out: int, stride: int = 1) -> None:
            super().__init__()
            self.conv = nn.Conv2d(
                channel_in,
                channel_out,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.branches = nn.ModuleList(
                [
                    nn.Conv2d(channel_out, channel_out, kernel_size=1, bias=False),
                    nn.Conv2d(
                        channel_out,
                        channel_out,
                        kernel_size=3,
                        dilation=3,
                        padding=3,
                        bias=False,
                    ),
                    nn.Conv2d(
                        channel_out,
                        channel_out,
                        kernel_size=3,
                        dilation=6,
                        padding=6,
                        bias=False,
                    ),
                    nn.Conv2d(
                        channel_out,
                        channel_out,
                        kernel_size=3,
                        dilation=9,
                        padding=9,
                        bias=False,
                    ),
                ]
            )

            self.gn1 = nn.GroupNorm(num_groups=8, num_channels=channel_out)
            self.gn2 = nn.GroupNorm(num_groups=8, num_channels=channel_out)
            self.gn3 = nn.GroupNorm(num_groups=8, num_channels=channel_out)

            self.activation = nn.ReLU(inplace=True)

            self.proj1 = (
                nn.Identity()
                if channel_in == channel_out and stride == 1
                else nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=stride)
            )
            self.proj2 = nn.Conv2d(channel_out * 4, channel_out, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            identity = self.proj1(x)
            out = self.activation(self.gn1(self.conv(x)))
            out = self.gn3(
                self.proj2(
                    torch.cat(
                        [self.gn2(self.activation(b(out))) for b in self.branches],
                        dim=1,
                    )
                )
            )
            return self.activation(out + identity)

    def __init__(self, channel_in: int) -> None:
        super().__init__()
        self.enc1 = ResUNet.ResidualEncoderBlock(channel_in, 64, stride=1)
        self.enc2 = ResUNet.ResidualEncoderBlock(64, 128, stride=2)
        self.enc3 = ResUNet.ResidualEncoderBlock(128, 256, stride=2)
        self.enc4 = self.ASPPBlock(256, 512, stride=2)
        self.bottleneck = self.ASPPBlock(512, 1024, stride=2)

        self.dec1 = ResUNet.ResidualDecoderBlock(1024, 512)
        self.dec2 = ResUNet.ResidualDecoderBlock(512, 256)
        self.dec3 = ResUNet.ResidualDecoderBlock(256, 128)
        self.dec4 = ResUNet.ResidualDecoderBlock(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.bottleneck(x4)

        x6 = self.dec1(x5, x4)
        x7 = self.dec2(x6, x3)
        x8 = self.dec3(x7, x2)
        x9 = self.dec4(x8, x1)

        return self.out(x9)


class AttentionGateASPPResUNet(nn.Module):
    class AttentionGate(nn.Module):
        def __init__(self, channel) -> None:
            super().__init__()
            intermediate_channel = channel // 2
            self.W_upsample = nn.Conv2d(channel, intermediate_channel, kernel_size=1)
            self.W_shortcut = nn.Conv2d(channel, intermediate_channel, kernel_size=1)
            self.attention_map = nn.Conv2d(intermediate_channel, 1, kernel_size=1)
            self.relu = nn.ReLU(inplace=True)
            self.sigmoid = nn.Sigmoid()

        def forward(self, shortcut, upsample):
            att = self.relu(self.W_shortcut(shortcut) + self.W_upsample(upsample))
            # Map the attention to 0-1
            att = self.sigmoid(self.attention_map(att))
            return shortcut * att

    class AttentionGateResDecoderBlock(nn.Module):
        def __init__(self, channel_in, channel_out) -> None:
            super().__init__()
            self.attention = AttentionGateASPPResUNet.AttentionGate(channel_out)
            self.conv1 = nn.Conv2d(
                channel_in, channel_out, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.conv2 = nn.Conv2d(
                channel_out, channel_out, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.proj = (
                nn.Identity()
                if channel_in == channel_out
                else nn.Conv2d(channel_in, channel_out, kernel_size=1)
            )
            self.up_conv = nn.ConvTranspose2d(
                channel_in, channel_out, kernel_size=2, stride=2
            )
            self.gn1 = nn.GroupNorm(num_groups=8, num_channels=channel_out)
            self.gn2 = nn.GroupNorm(num_groups=8, num_channels=channel_out)
            self.activation = nn.ReLU(inplace=True)

        def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            # x1: decoder branch, x2: shortcut branch
            up = self.up_conv(x1)
            x2 = self.attention(x2, up)
            x = torch.cat((up, x2), dim=1)
            identity = self.proj(x)
            out = self.activation(self.gn1(self.conv1(x)))
            out = self.gn2(self.conv2(out))
            return self.activation(out + identity)

    def __init__(self, channel_in: int) -> None:
        super().__init__()
        self.enc1 = ResUNet.ResidualEncoderBlock(channel_in, 64, stride=1)
        self.enc2 = ResUNet.ResidualEncoderBlock(64, 128, stride=2)
        self.enc3 = ResUNet.ResidualEncoderBlock(128, 256, stride=2)
        self.enc4 = ASPPResUNet.ASPPBlock(256, 512, stride=2)
        self.bottleneck = ASPPResUNet.ASPPBlock(512, 1024, stride=2)

        self.dec1 = self.AttentionGateResDecoderBlock(1024, 512)
        self.dec2 = self.AttentionGateResDecoderBlock(512, 256)
        self.dec3 = self.AttentionGateResDecoderBlock(256, 128)
        self.dec4 = self.AttentionGateResDecoderBlock(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.bottleneck(x4)

        x6 = self.dec1(x5, x4)
        x7 = self.dec2(x6, x3)
        x8 = self.dec3(x7, x2)
        x9 = self.dec4(x8, x1)

        return self.out(x9)
