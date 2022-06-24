import cv2
import numpy as np
import torch as th

from cityscapes import Cityscapes


def readlines(x):
    with open(x, "r") as f:
        out = f.read().split("\n")
    out = [o for o in out if len(o)]
    return out


def minmax_norm(x):
    return (x - x.min()) / (x.max() - x.min())


def decode(x, transpose=True):
    x = x.numpy().astype(np.int32)
    x = np.array([Cityscapes.decode_target(xx[0]) for xx in x])
    if transpose:
        x = x.transpose(0, 3, 1, 2)
    x = th.Tensor(x)
    return x


def load_frame(x, norm=True, size=None, mode=cv2.INTER_LINEAR):
    x = cv2.imdecode(x, -1)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    if size:
        x = cv2.resize(x, size, interpolation=mode)
    if norm:
        x = np.float32(x) / 255.0
    x = x.transpose(2, 0, 1)
    x = th.Tensor(x)
    return x


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, (th.nn.Conv2d, th.nn.Linear)):
                th.nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, th.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class Normalize:
    def __call__(self, x):
        o = (x - x.min()) / (x.max() - x.min())
        return o


class AudioResizeNormalize:
    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, x):
        n_dim = len(x.shape)
        if n_dim == 3:
            x = x.unsqueeze(0)
        o = th.nn.functional.interpolate(x, size=self.size)
        o = (o - o.max()) / (o.max() - o.min())
        if n_dim == 3:
            o = o[0]
        return o
