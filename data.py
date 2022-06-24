#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import h5py
import numpy as np
import torch as th
import torchvision as thv

from util import load_frame

null = th.Tensor([])


class Dataset(th.utils.data.Dataset):
    def __init__(
        self,
        audio_f=None,
        video_f=None,
        depth_f=None,
        seg_f=None,
        bg_f=None,
        mask_f=None,
        ids=None,
        dataset="eth",
        train=True,
        audio_wl=2,
        audio_tran=None,
        audio_sr=22050,
        audio_norm=True,
        video_tran=None,
        video_fps=30,
        audio_size=(256, 256),
        fix_t=False,
        ret_id=False,
    ):
        self.audio_f = audio_f
        self.video_f = video_f
        self.depth_f = depth_f
        self.seg_f = seg_f
        self.bg_f = bg_f
        self.mask_f = mask_f
        print(train, len(ids))
        if ids is not None:
            self.ids = ids
        else:
            with h5py.File(audio_f, "r") as fil:
                ids = list(fil.keys())
        self.dataset = dataset
        self.train = train
        self.audio_wl = audio_wl
        self.audio_hwl = audio_wl / 2.0
        self.audio_tran = audio_tran
        self.audio_norm = audio_norm
        self.audio_size = audio_size
        self.audio_sr = audio_sr
        self.video_tran = video_tran
        self.video_fps = video_fps
        self.fix_t = fix_t
        self.ret_id = ret_id

    def load_audio(self, fil, vid, t=None, norm=False, mode=cv2.INTER_AREA):
        with h5py.File(fil, "r") as in_f:
            N_samples = in_f[vid].shape[-1]
            if t is None:
                dur = N_samples / float(self.audio_sr)
                t_s, t_e = self.audio_hwl, dur - self.audio_hwl
                t = (t_e - t_s) * th.rand(1) + t_s
                # if self.dataset == "eth" and self.depth_f:
                #    t = th.randint(0, int(dur) - 1, size=(1,)) + 0.5
            p1 = int((t - self.audio_hwl) * self.audio_sr)
            p2 = int((t + self.audio_hwl) * self.audio_sr)
            x = in_f[vid][:, p1:p2]
        if self.audio_tran:
            x = self.audio_tran(th.from_numpy(x))
            x /= 80.0
            # if norm:
            #     x = (x - x.min()) / (x.max() - x.min())
            # x = np.stack(
            #     [
            #         cv2.resize(xx.numpy(), self.audio_size, interpolation=mode)
            #         for xx in x
            #     ]
            # )
            # x = th.from_numpy(x)
        if not self.fix_t:
            t = None
        return x, t

    def load_frame(self, fil, vid, t=None, norm=True, mode=cv2.INTER_AREA):
        with h5py.File(fil, "r") as in_f:
            N_frames = len(in_f[vid])
            if t is None:
                t = int(th.randint(N_frames - 1, size=(1,)))
                t = t / self.video_fps
            p = int(t * self.video_fps)
            p = min(p, N_frames - 1)
            x = in_f[vid][p]
            if x.dtype == np.float32:
                if x.shape[1:] != self.audio_size:
                    x = cv2.resize(x[0], self.audio_size, interpolation=mode)
                    x = x[np.newaxis]
                x = th.from_numpy(x)
            else:
                x = load_frame(
                    np.frombuffer(x, np.uint8),
                    norm=norm,
                    size=self.audio_size,
                    mode=mode,
                )
        if not self.fix_t:
            t = None
        return x, t

    def __getitem__(self, i):
        # if self.train:
        if False:
            i, t = self.ids[i], None
        else:
            if self.dataset != "mavd":
                i, t = self.ids[i]
            else:
                i, t = self.ids[i], None

        try:
            if self.ret_id:
                x_id = "{}_{}".format(i, str(int(t * 1000)).zfill(10))
                x_id = np.array(x_id)
            else:
                x_id = null
            if self.audio_f:
                x_a, t = self.load_audio(self.audio_f, i, t=t)
            else:
                x_a = null
            if self.video_f:
                x_v, t = self.load_frame(self.video_f, i, t=t)
            else:
                x_v = null
            if self.depth_f:
                x_d, t = self.load_frame(self.depth_f, i, t=t)
            else:
                x_d = null
            if self.seg_f:
                x_s, t = self.load_frame(
                    self.seg_f, i, t=t, mode=cv2.INTER_NEAREST
                )
                x_s = x_s[:1]
            else:
                x_s = null
        except KeyError:
            return {
                "id": null,
                "audio": null,
                "video": null,
                "depth": null,
                "seg": null,
            }

        if self.bg_f:
            with h5py.File(self.bg_f, "r") as b_f:
                x_b = load_frame(b_f[i][...])
            x_mo = th.zeros_like(x_s)
            for i in [13, 17, 16]:
                x_mo[x_s == i] = 1
            x_s *= x_mo
            x_s *= (x_v != x_b).sum(axis=0) > 0

        if self.mask_f:
            x_m, t = self.load_frame(
                self.mask_f, i, t=t, mode=cv2.INTER_NEAREST
            )
            x_m[x_m > 0] = 1.0
            x_v = x_v * x_m
            x_v = thv.transforms.functional.gaussian_blur(x_v, (3, 3))

        out = {"id": x_id, "audio": x_a, "video": x_v, "depth": x_d, "seg": x_s}
        return out

    def __len__(self):
        return len(self.ids)

    def collate_fn(batch):
        keys = ["audio", "video", "depth", "seg"]
        ids = np.array([x["id"] for x in batch])
        asd = np.array([[x[k].numel() for k in keys] for x in batch])
        asd = np.where(asd.sum(axis=1) != 0)[0]
        data = {d: th.stack([batch[i][d] for i in asd]) for d in keys}
        data["ids"] = ids[asd]
        return data


class MavdDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_audio(self, fil, vid, t=None, norm=False):
        with h5py.File(fil, "r") as in_f:
            x = in_f[vid][:]
        if self.audio_tran:
            x = self.audio_tran(th.from_numpy(x))
        if self.audio_size:
            x = np.stack([cv2.resize(xx.numpy(), self.audio_size) for xx in x])
            x = th.from_numpy(x)
        return x, t

    def load_frame(self, fil, vid, t=None, norm=True, mode=cv2.INTER_LINEAR):
        with h5py.File(fil, "r") as in_f:
            x = in_f[vid][:]
            if x.dtype == "float32":
                if self.audio_size != x.shape[2:]:
                    x = cv2.resize(x, self.audio_size, interpolation=mode)
                x = th.from_numpy(x)
                x = x.unsqueeze(0)
            else:
                x = load_frame(x, norm=norm, size=self.audio_size, mode=mode)
        return x, t
