import numpy as np
import torch as th

from custom import Builder
from vq import VAE, VQVAE

device = th.device("cuda" if th.cuda.is_available() else "cpu")


def encoder(
    in_ch,
    hid_dim,
    emb_dim,
    in_size,
    out_size,
    nbf=128,
    n_res=2,
    res_dim=32,
    conv="Conv2dRelu",
    res="ResidualReluStack",
):
    modules = [["Bnorm2d", [in_ch]]]

    n_pool = int(np.log2(in_size / out_size))
    in_ch = [in_ch, nbf // 2] + [nbf] * (n_pool - 2)
    out_ch = in_ch[1:] + [nbf]

    for ic, oc in zip(in_ch, out_ch):
        modules.append([conv, [ic, oc, 4, 2, 1]])
    modules.append([res, [nbf, hid_dim, n_res, res_dim]])
    modules.append(["Conv2d", [hid_dim, emb_dim, 1, 1, 0]])
    return modules


def decoder(
    out_ch,
    hid_dim,
    emb_dim,
    in_size,
    out_size,
    nbf=128,
    n_res=2,
    res_dim=32,
    conv="TConv2dRelu",
    res="ResidualReluStack",
):
    n_up = int(np.log2(out_size / in_size))
    in_chs = [out_ch, nbf // 2] + [nbf] * (n_up - 2)
    out_chs = in_chs[1:] + [nbf]

    modules = [
        ["Conv2d", [emb_dim, hid_dim, 3, 1, 1]],
        [res, [hid_dim, nbf, n_res, res_dim]],
    ]
    for ic, oc in zip(out_chs[::-1][:-1], in_chs[::-1][:-1]):
        modules.append([conv, [ic, oc, 4, 2, 1]])
    modules.append(["TConv2d", [nbf // 2, out_ch, 4, 2, 1]])
    return modules


class Manifold(th.nn.Module):
    def __init__(self, in_ch, out_ch, in_size, out_size, ckpt_f=None):
        super().__init__()
        ckpt = th.load(ckpt_f, map_location=th.device("cpu"))
        self.vq = VQVAE(
            Builder(
                encoder(
                    in_ch,
                    ckpt["hid_dim"],
                    ckpt["emb_dim"],
                    in_size,
                    out_size,
                )
            ),
            Builder(
                decoder(
                    out_ch,
                    ckpt["hid_dim"],
                    ckpt["emb_dim"],
                    out_size,
                    in_size,
                )
            ),
            ckpt["emb_num"],
            ckpt["emb_dim"],
        )
        if ckpt_f:
            self.vq.load_state_dict(ckpt["model"])
            self.vq.eval()

    @th.no_grad()
    def indices(self, x):
        o = self.vq._vq_vae(self.vq._enc(x))[-1]
        return o

    @th.no_grad()
    def encode(self, x):
        o = self.vq._enc(x)
        return o

    @th.no_grad()
    def decode(self, x):
        # o = self.vq.test(x)
        o = self.vq._vq_vae(x)[1]
        o = self.vq._dec(o)
        return o

    def decode_with_gradients(self, x):
        # o = self.vq.test(x)
        o = self.vq._vq_vae(x)[1]
        o = self.vq._dec(o)
        return o


class VAEManifold(th.nn.Module):
    def __init__(self, in_ch, out_ch, in_size, out_size, ckpt_f=None):
        super().__init__()
        ckpt = th.load(ckpt_f, map_location=th.device("cpu"))
        self.vq = VAE(
            Builder(
                encoder(
                    in_ch,
                    ckpt["hid_dim"],
                    ckpt["emb_dim"],
                    in_size,
                    out_size,
                )
            ),
            Builder(
                decoder(
                    out_ch,
                    ckpt["hid_dim"],
                    ckpt["emb_dim"],
                    out_size,
                    in_size,
                )
            ),
            ckpt["emb_num"],
            ckpt["emb_dim"],
        )
        if ckpt_f:
            self.vq.load_state_dict(ckpt["model"])
            self.vq.eval()

    @th.no_grad()
    def indices(self, x):
        raise NotImplementedError

    @th.no_grad()
    def encode(self, x):
        o = self.vq._enc(x)
        return o

    @th.no_grad()
    def decode(self, x):
        # o = self.vq.test(x)
        # o = self.vq._vq_vae(x)[1]
        o = self.vq._dec(x)
        return o


class Manifolds(th.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.main = {k: Manifold(v) for k, v in kwargs.items()}

    def indices(self, **kwargs):
        out = {
            k: self.main[k].indices(v) for k, v in kwargs.items() if v.numel()
        }
        return out

    def decode(self, **kwargs):
        pass
