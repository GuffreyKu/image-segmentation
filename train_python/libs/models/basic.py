import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


def activation_func(activation):
    return nn.ModuleDict({
        'relu': nn.ReLU(inplace=True),
        'leaky_relu': nn.LeakyReLU(negative_slope=0.01, inplace=True),
        'sigmoid': nn.Sigmoid(),
        'prelu': nn.PReLU(),
        'softmax': nn.Softmax(dim=1),
        'gelu': nn.GELU()})[activation]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channel, k_size, activation='relu', pad=1, s=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channel,
                              k_size, padding=pad, stride=s, dilation=dilation)
        if activation != 'softmax':
            self.batchNorm = nn.BatchNorm2d(out_channel)
        self.actfunction = activation_func(activation)
        self.act_name = activation

    def forward(self, x):
        x = self.conv(x)

        if self.act_name != 'softmax':
            x = self.batchNorm(x)

        x = self.actfunction(x)
        return x


class DeCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channel, k_size, pad=1, s=1, dilation=1, op=0):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channel, k_size, padding=pad, stride=s, dilation=dilation, output_padding=op)
        # self.batchNorm = nn.BatchNorm2d(out_channel)
        # self.actfunction = activation_func(activation)

    def forward(self, x):
        x = self.deconv(x)
        # x = self.batchNorm(x)
        # x = self.actfunction(x)
        return x


class GWCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channel, k_size, activation='relu', pad=1, s=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, k_size,
                              groups=in_channels, padding=pad, stride=s, dilation=dilation)
        self.pointConv = nn.Conv2d(
            in_channels, out_channel, kernel_size=1, stride=1, padding=0, groups=1)
        self.batchNorm = nn.BatchNorm2d(out_channel)
        self.actfunction = activation_func(activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.pointConv(x)
        x = self.actfunction(x)
        x = self.batchNorm(x)
        return x


class Fullyconnect(nn.Module):
    def __init__(self, in_channels, out_channel, activation):
        super().__init__()
        self.act_name = activation
        self.fc = nn.Linear(in_channels, out_channel)
        nn.init.kaiming_normal_(self.fc.weight)
        if self.act_name != 'linear':
            self.actfunction = activation_func(activation)
            self.bn = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        if self.act_name == 'linear':
            return x
        return self.bn(self.actfunction(x))


class FcRes(nn.Module):
    def __init__(self, in_channels, activation):
        super().__init__()
        self.fc = Fullyconnect(in_channels, in_channels, activation)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x + self.fc(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channel, activation='relu', s=1):
        super().__init__()
        self.convR1 = CNNBlock(in_channels,
                               int(in_channels/4),
                               1, activation, 0, s)
        self.convR2 = CNNBlock(
            int(in_channels/4), int(in_channels/4), 3, activation, 1, s)
        self.convR3 = CNNBlock(
            int(in_channels/4), out_channel, 1, activation, 0, s)
        self.actfunctionR = activation_func(activation)

    def forward(self, x):
        x1 = self.convR1(x)
        x2 = self.convR2(x1)
        x3 = self.convR3(x2)
        res = x + x3
        res = self.actfunctionR(res)
        return res


class ResBlockA(nn.Module):
    def __init__(self, in_channels, out_channel, activation='relu', s=1):
        super().__init__()
        self.conv1 = CNNBlock(in_channels, int(
            in_channels/2), 1, activation, 0, s)
        self.conv2 = CNNBlock(int(in_channels/2),
                              int(in_channels/2), 3, activation, s=1)
        self.conv3 = CNNBlock(int(in_channels/2),
                              out_channel, 1, activation, 0, s=1)
        self.conv4 = CNNBlock(in_channels, out_channel, 1, activation, 0, s)

        self.actfunction = activation_func(activation)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x)
        res = x3 + x4
        res = self.actfunction(res)
        return res


if __name__ == "__main__":
    import torch
    input = torch.randn(1, 16, 14, 14)
    downsample = CNNBlock(16, 16, 3, s=2, pad=1)
    upsample = DeCNNBlock(16, 16, 4, s=2, pad=1, dilation=1)
    h = downsample(input)
    print(input.size())
    output = upsample(h)
    print(output.size())
