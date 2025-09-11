import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.inc = ConvBlock(n_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(512, 1024)
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = ConvBlock(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = ConvBlock(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = ConvBlock(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path with skip connections
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.up_conv1(x)

        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.up_conv2(x)

        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.up_conv3(x)

        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.up_conv4(x)

        return self.outc(x)
