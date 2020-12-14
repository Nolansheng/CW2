import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 4)

        self.down1 = Down(4, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)

        self.down4 = Down(64, 64)

        self.up1 = Up(128, 32)
        self.up2 = Up(64, 16)
        self.up3 = Up(32, 4)
        self.up4 = Up(8, 4)

        self.outc = OutConv(4, n_classes)

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