import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 8)

        self.down1 = Down(8, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)

        self.down4 = Down(128, 128)

        self.up1 = Up(256, 64)
        self.up2 = Up(128, 32)
        self.up3 = Up(64, 8)
        self.up4 = Up(16, 8)

        self.outc = OutConv(8, n_classes)

    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        x = self.up1(x5, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        logits = self.outc(x)

        return logits


model = UNet(1, 4)