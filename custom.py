# -*- coding: utf-8 -*-
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class PixShuff(th.nn.Module):
    def __init__(self, up):
        super().__init__()
        self.pixshuff = th.nn.PixelShuffle(up)

    def forward(self, x):
        o = self.pixshuff(x)
        return o


class Relu(th.nn.Module):
    def forward(self, x):
        return th.nn.functional.relu(x, True)


class AdaptiveAvgPool2d(th.nn.Module):
    def __init__(self, sz):
        super().__init__()
        self.pool = th.nn.AdaptiveAvgPool2d(sz)

    def forward(self, x):
        return self.pool(x)


class Bnorm2d(th.nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.bn = th.nn.BatchNorm2d(ch)

    def forward(self, x):
        o = self.bn(x)
        return o


class Drop2d(th.nn.Module):
    def __init__(self, prob):
        super().__init__()
        self.drop = th.nn.Dropout2d(prob)

    def forward(self, x):
        o = self.drop(x)
        return o


class LinearBnRelu(th.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.main = th.nn.Sequential(
            th.nn.Linear(n_in, n_out),
            th.nn.BatchNorm1d(n_out),
            th.nn.ReLU(True),
        )

    def forward(self, x):
        o = self.main(x)
        return o


class LinearBnReluDrop(th.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.main = th.nn.Sequential(
            th.nn.Linear(n_in, n_out),
            th.nn.BatchNorm1d(n_out),
            th.nn.ReLU(True),
            th.nn.Dropout(0.1),
        )

    def forward(self, x):
        o = self.main(x)
        return o


class Conv2d(th.nn.Module):
    def __init__(self, ch_in, ch_out, k, s, p):
        super().__init__()
        self.conv = th.nn.Conv2d(ch_in, ch_out, k, s, p)

    def forward(self, x):
        o = self.conv(x)
        return o


class Conv2dRelu(th.nn.Module):
    def __init__(self, ch_in, ch_out, k, s, p):
        super().__init__()
        self.conv = th.nn.Sequential(
            th.nn.Conv2d(ch_in, ch_out, k, s, p),
            th.nn.ReLU(True),
        )

    def forward(self, x):
        o = self.conv(x)
        return o


class Conv2dBn(th.nn.Module):
    def __init__(self, ch_in, ch_out, k, s, p):
        super().__init__()
        self.conv = th.nn.Sequential(
            th.nn.Conv2d(ch_in, ch_out, k, s, p, bias=False),
            th.nn.BatchNorm2d(ch_out),
        )

    def forward(self, x):
        o = self.conv(x)
        return o


class Conv2dBnRelu(th.nn.Module):
    def __init__(self, ch_in, ch_out, k, s, p):
        super().__init__()
        self.conv = th.nn.Sequential(
            th.nn.Conv2d(ch_in, ch_out, k, s, p, bias=False),
            th.nn.BatchNorm2d(ch_out),
            th.nn.ReLU(True),
        )

    def forward(self, x):
        o = self.conv(x)
        return o


class Conv2dBnLrelu(th.nn.Module):
    def __init__(self, ch_in, ch_out, k, s, p, slope):
        super().__init__()
        self.conv = th.nn.Sequential(
            th.nn.Conv2d(ch_in, ch_out, k, s, p, bias=False),
            th.nn.BatchNorm2d(ch_out),
            th.nn.LeakyReLU(slope, True),
        )

    def forward(self, x):
        o = self.conv(x)
        return o


class TConv2d(th.nn.Module):
    def __init__(self, ch_in, ch_out, k, s, p):
        super().__init__()
        self.conv = th.nn.ConvTranspose2d(ch_in, ch_out, k, s, p)

    def forward(self, x):
        o = self.conv(x)
        return o


class TConv2dRelu(th.nn.Module):
    def __init__(self, ch_in, ch_out, k, s, p):
        super().__init__()
        self.conv = th.nn.Sequential(
            th.nn.ConvTranspose2d(ch_in, ch_out, k, s, p),
            th.nn.ReLU(True),
        )

    def forward(self, x):
        o = self.conv(x)
        return o


class TConv2dBnRelu(th.nn.Module):
    def __init__(self, ch_in, ch_out, k, s, p):
        super().__init__()
        self.conv = th.nn.Sequential(
            th.nn.ConvTranspose2d(ch_in, ch_out, k, s, p, bias=False),
            th.nn.BatchNorm2d(ch_out),
            th.nn.ReLU(True),
        )

    def forward(self, x):
        o = self.conv(x)
        return o


class ResidualRelu(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super().__init__()
        self._block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1,
            ),
        )

    def forward(self, x):
        return F.relu(x + self._block(x))


class ResidualReluStack(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
    ):
        super().__init__()
        self._res_stack = nn.Sequential(
            *[
                ResidualRelu(in_channels, num_hiddens, num_residual_hiddens)
                for _ in range(num_residual_layers)
            ]
        )

    def forward(self, x):
        o = self._res_stack(x)
        return o


class Conv2dReluStack(nn.Module):
    def __init__(self, n_filters):
        super().__init__()

        modules = []
        for i in range(len(n_filters[:-1])):
            in_ch, out_ch = n_filters[i], n_filters[i + 1]
            modules += [TConv2dRelu(in_ch, out_ch, 4, 2, 1)]

        self.conv_stack = th.nn.Sequential(*modules)

    def forward(self, x):
        o = self.conv_stack(x)

        return o


class TConv2dReluStack(nn.Module):
    def __init__(self, n_filters):
        super().__init__()

        modules = []
        for i in range(len(n_filters[:-1])):
            in_ch, out_ch = n_filters[i], n_filters[i + 1]
            modules += [TConv2dRelu(in_ch, out_ch, 4, 2, 1)]

        self.tconv_stack = th.nn.Sequential(*modules)

    def forward(self, x):
        o = self.tconv_stack(x)

        return o


class Builder(th.nn.Module):
    def __init__(self, conf):
        super().__init__()
        modules = []
        for module in conf:
            mod_type, args = module
            modules += [eval(mod_type)(*args)]

        self.main = th.nn.Sequential(*modules)

    def forward(self, x):
        o = self.main(x)
        return o
