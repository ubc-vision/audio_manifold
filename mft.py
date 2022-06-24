# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torchvision as thv

from custom import Builder


class Tran(th.nn.Module):
    def __init__(self, dim, num=3, drop=0.2):
        super().__init__()
        tran = []
        for i in range(num):
            tran.append(["Conv2dBnRelu", [dim, dim, 1, 1, 0]])
            if i < num - 1:
                tran.append(["Drop2d", [drop]])

        self.tran = Builder(tran)

    def forward(self, x):
        o = self.tran(x)
        return o


class Encoder(th.nn.Module):
    def __init__(self, in_ch, in_size=128, conv="Conv2dBnRelu"):
        super().__init__()
        n_down = int(np.log2(in_size))
        encoder = [["Bnorm2d", [in_ch]]]
        encoder.append([conv, [in_ch, 64, 4, 2, 1]])
        encoder.append([conv, [64, 128, 4, 2, 1]])
        encoder.append([conv, [128, 256, 4, 2, 1]])
        encoder.append([conv, [256, 512, 4, 2, 1]])
        encoder.append([conv, [512, 1024, 4, 2, 1]])
        for i in range(n_down - len(encoder[1:])):
            encoder.append([conv, [1024, 1024, 4, 2, 1]])

        self.encoder = Builder(encoder)

    def forward(self, x):
        o = self.encoder(x)
        return o


class Shallow(th.nn.Module):
    def __init__(
        self,
        in_ch,
        vq_dim,
        in_size,
        out_size,
        conv_down="Conv2dBnRelu",
        conv_up="TConv2dBnRelu",
        n_bf=64,
        max_f=256,
        inner_size=1,
    ):
        super().__init__()
        n_down = int(np.log2(in_size / inner_size))
        out_ch = n_bf
        enc = [["Bnorm2d", [in_ch]]]
        for _ in range(n_down):
            enc.append([conv_down, [in_ch, out_ch, 4, 2, 1]])
            in_ch, out_ch = out_ch, min(max_f, out_ch * 2)
        enc.append(["AdaptiveAvgPool2d", [1]])
        self.enc = Builder(enc)

        self.tra = Tran(out_ch)

        n_up = int(np.log2(out_size / inner_size))
        dec = []
        in_ch, out_ch = out_ch, out_ch // 2
        for _ in range(n_up):
            dec.append([conv_up, [in_ch, out_ch, 4, 2, 1]])
            in_ch, out_ch = out_ch, max(n_bf, out_ch // 2)
        dec.append(["Conv2d", [in_ch, vq_dim, 1, 1, 0]])
        self.dec = Builder(dec)

    def forward(self, x):
        o = self.enc(x)
        o = self.tra(o)
        o = self.dec(o)
        return o


class Decoder(th.nn.Module):
    def __init__(self, in_ch, out_ch, out_size, conv="TConv2dBnRelu"):
        super().__init__()
        n_up = int(np.log2(out_size))
        ch = in_ch
        dec = []
        for _ in range(n_up):
            dec.append([conv, [ch, max(64, ch // 2), 4, 2, 1]])
            ch = max(64, ch // 2)
        dec.append(["TConv2d", [ch, out_ch, 1, 1, 0]])

        self.dec = Builder(dec)

    def forward(self, x):
        o = self.dec(x)
        return o


class OldTransform(th.nn.Module):
    def __init__(self, in_ch, out_ch, in_size, out_size, dim=1024):
        super().__init__()
        self.enc = Encoder(in_ch, in_size)
        self.tra = Tran(dim, 3)
        self.dec = Decoder(dim, out_ch, out_size)

    def forward(self, x):
        o = self.enc(x)
        o = self.tra(o)
        o = self.dec(o)
        return o


class ResTransform(th.nn.Module):
    def __init__(
        self, in_ch, out_ch, in_size, out_size, dim=512, pretrained=False
    ):
        super().__init__()
        self.pretrained = pretrained
        pre = th.nn.Sequential(
            th.nn.BatchNorm2d(in_ch),
            th.nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False),
        )
        encoder = thv.models.resnet18(pretrained=False)
        self.encoder = th.nn.Sequential(pre, *list(encoder.children())[1:-1])
        self.tran = Tran(dim)
        self.decoder = Decoder(dim, out_ch, out_size)
        # if n_up == 3:
        #     self.tran = Builder(
        #         [
        #             ["Conv2dBnRelu", [512, 512, 1, 1, 0]],
        #             ["Drop2d", [0.2]],
        #             ["Conv2dBnRelu", [512, 512, 1, 1, 0]],
        #             ["Drop2d", [0.2]],
        #             ["Conv2dBnRelu", [512, 512, 1, 1, 0]],
        #             ["Drop2d", [0.2]],
        #             ["TConv2dBnRelu", [512, 256, 4, 2, 1]],
        #             ["TConv2dBnRelu", [256, 128, 4, 2, 1]],
        #             ["TConv2d", [128, out_dim, 4, 2, 1]],
        #         ]
        #     )
        # elif n_up == 4:
        #     self.tran = Builder(
        #         [
        #             ["Conv2dBnRelu", [512, 512, 1, 1, 0]],
        #             ["Drop2d", [0.2]],
        #             ["Conv2dBnRelu", [512, 512, 1, 1, 0]],
        #             ["Drop2d", [0.2]],
        #             ["Conv2dBnRelu", [512, 512, 1, 1, 0]],
        #             ["Drop2d", [0.2]],
        #             ["TConv2dBnRelu", [512, 256, 4, 2, 1]],
        #             ["TConv2dBnRelu", [256, 128, 4, 2, 1]],
        #             ["TConv2dBnRelu", [128, 128, 4, 2, 1]],
        #             ["TConv2d", [128, out_dim, 4, 2, 1]],
        #         ]
        #     )
        # elif n_up == 5:
        #     self.tran = Builder(
        #         [
        #             ["Conv2dBnRelu", [512, 512, 1, 1, 0]],
        #             ["Drop2d", [0.2]],
        #             ["Conv2dBnRelu", [512, 512, 1, 1, 0]],
        #             ["Drop2d", [0.2]],
        #             ["Conv2dBnRelu", [512, 512, 1, 1, 0]],
        #             ["Drop2d", [0.2]],
        #             ["TConv2dBnRelu", [512, 256, 4, 2, 1]],
        #             ["TConv2dBnRelu", [256, 128, 4, 2, 1]],
        #             ["TConv2dBnRelu", [128, 128, 4, 2, 1]],
        #             ["TConv2dBnRelu", [128, 128, 4, 2, 1]],
        #             ["TConv2d", [128, out_dim, 4, 2, 1]],
        #         ]
        #     )
        # else:
        #     raise ValueError
        # self.model = th.nn.Sequential(self.pre, self.encoder, self.tran)

    def forward(self, x):
        o = self.encoder(x)
        o = self.tran(o)
        o = self.decoder(o)
        return o

    def params(self):
        if self.pretrained:
            params = list(self.model[0].parameters()) + list(
                self.model[-1].parameters()
            )
        else:
            params = self.model.parameters()
        return params
