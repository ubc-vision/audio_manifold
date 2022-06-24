#!/usr/bin/env python
# -*- coding=utf-8 -*-

import os
from glob import glob

import h5py
import numpy as np
import torch as th
import torchaudio as tha
import torchvision as thv
from torch.utils.tensorboard import SummaryWriter

from bin_sound.eth_networks import OmniNet_DepthSeg, OmniNet_Depth, OmniNet_Seg
from bin_sound.eth_optimizers import get_optimizer_depth_seg
from bin_sound.eth_loss import get_loss
from echo2depth import echo2depth, echo2depth_loss
from custom import Builder
from data import Dataset, MavdDataset
from manifold import Manifold, VAEManifold, decoder, encoder
from metrics import depth_metrics, seg_metrics
from mft import OldTransform, ResTransform, Shallow
from util import decode, readlines
from vq import VAE, VQVAE

device = th.device("cuda" if th.cuda.is_available() else "cpu")

CH = {
    "ytasmr": {"audio": 2, "video": 3, "seg": 1, "depth": 3},
    "mavd": {"audio": 8, "video": 3, "seg": 1, "depth": 1},
    "eth": {"audio": 8, "video": 3, "seg": 1, "depth": 1},
}
AUDIO_WL = {"ytasmr": 2, "eth": 1, "mavd": 1}
TRANSFORM = {
    "old": OldTransform,
    "res": ResTransform,
    "hm-res": ResTransform,
    "shallow": Shallow,
    "hm-shallow": Shallow,
}
BEST = {"depth": -1, "seg": -1, "depth-seg": -1}
BEST_FN = {"depth": lambda x, y: x > y, "seg": lambda x, y: x > y, "depth-seg": lambda x, y: x > y}
MANIFOLD = {"vqvae": VQVAE, "vae": VAE}
AUDIO_SR = {"ytasmr": 11025.0, "eth": 22050.0, "mavd": 16000.0}
VIDEO_FPS = {"ytasmr": 25.0, "eth": 30.0, "mavd": 1.0}
RES_SIZE = {"depth": (320, 1024), "seg": (480, 960)}
METRIC_LOG_KEYS = {
    "depth": ["auc", "abs_rel", "seq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "rel_err",],
    "seg": ["iou"],
    "depth-seg": [
        "auc",
        "abs_rel",
        "seq_rel",
        "rmse",
        "rmse_log",
        "a1",
        "a2",
        "a3",
        "rel_err",
        ["iou"],
    ],
}


class Logger:
    def __init__(self, data, writer, keys, verbose=True):
        self.data = data
        self.writer = writer
        self.keys = keys
        self.verbose = verbose
        self.depth_keys = METRIC_LOG_KEYS["depth"]
        self.seg_keys = METRIC_LOG_KEYS["seg"]

    def __call__(self, i, tr_loss, tr_rec_loss, te_loss, te_rec_loss, te_metrics):
        tr_loss = tr_loss.mean()
        te_loss = te_loss.mean()
        tr_rec_loss = tr_rec_loss.mean()
        te_rec_loss = te_rec_loss.mean()
        if self.verbose:
            print(i, tr_loss, te_loss)

        score = te_rec_loss
        if self.writer is not None:
            loss = {
                "loss/train": tr_loss,
                "loss/test": te_loss,
                "rec/train": tr_rec_loss,
                "rec/test": te_rec_loss,
            }
            if self.data == "depth":
                te_metrics = te_metrics.mean(axis=0)
                score = te_metrics[0]
                asd = {"metric/" + k: v for k, v in zip(self.keys[:-1], te_metrics[:8])}
                qwe = {"metric/rel_err_" + str(i): v for i, v in enumerate(te_metrics[8:])}
                metric = {**asd, **qwe}

            elif self.data == "seg":
                miou = np.nanmean(te_metrics, axis=-1).mean()
                iou = np.nanmean(te_metrics, axis=0)
                score = miou
                metric = {"metric/iou_" + str(i): v for i, v in enumerate(iou)}
                metric = {**metric, **{"metric/miou": miou}}
            else:
                raise ValueError

            log = {**loss, **metric}
            for k, v in log.items():
                self.writer.add_scalar(k, v, i)

        return score


class Metrics:
    def __init__(self, data, size=None, size_depth=None, size_seg=None):
        self.data = data
        self.size = size
        self.size_depth = size_depth
        self.size_seg = size_seg
        self.resize = th.nn.functional.interpolate

    def __call__(self, pred, gt):
        pred = pred.detach()

        if self.size:
            if self.data == "seg":
                pred = pred.argmax(1, keepdims=True).float()
            pred = self.resize(pred, self.size, mode="nearest")
            gt = self.resize(gt, self.size, mode="nearest")

        if self.data == "depth":
            pred = pred.cpu().numpy()
            gt = gt.cpu().numpy()
            out = np.array([depth_metrics(g, p) for g, p in zip(gt, pred)])

        elif self.data == "seg":
            gt = (gt * 255).long().squeeze()
            pred = pred.long().squeeze()
            out = np.array([seg_metrics(p, g) for p, g in zip(pred, gt)])

        else:
            raise ValueError

        return out


class ManifoldTrain:
    def __init__(self, args, writer, ckpt_f, best_f):
        self.args = args
        self.writer = writer
        self.ckpt_f = ckpt_f
        self.best_f = best_f

        lr = args.learning_rate
        manifold = MANIFOLD[args.manifold]
        self.ch = CH[args.dataset][args.data]
        out_ch = 19 if args.data == "seg" else self.ch
        enc = Builder(
            encoder(self.ch, args.vq_hid_dim, args.vq_emb_dim, args.in_size, args.out_size,)
        )
        dec = Builder(
            decoder(out_ch, args.vq_hid_dim, args.vq_emb_dim, args.out_size, args.in_size,)
        )
        self.model = manifold(enc, dec, args.vq_emb_num, args.vq_emb_dim).to(device)
        self.opt = th.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = th.nn.functional.mse_loss
        if args.data == "seg":
            self.loss_fn = th.nn.functional.cross_entropy

    def restore(self, ckpt_f):
        ckpt = th.load(ckpt_f)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["opt"])
        iter = ckpt["iter"]
        best = ckpt["best"]
        return iter, best

    def save_image(self, true, pred, fname, N=16, nrow=1):
        args = self.args
        true, pred = true[:N, ...].cpu(), pred[:N, ...].detach().cpu()
        if args.data == "seg":
            pred = pred.argmax(dim=1, keepdims=True)
            true = decode(true * 255) / 255
            pred = decode(pred) / 255
        elif args.data == "audio":
            from util import minmax_norm

            true = true[:, :1, ...]
            pred = pred[:, :1, ...]
            true = minmax_norm(true)
            pred = minmax_norm(pred)
        else:
            pass
        out = th.cat((true, pred), -1)
        thv.utils.save_image(out, os.path.join(args.log_dir, fname), nrow=1)

    def __call__(self, tr_loader):
        args = self.args
        model = self.model
        opt = self.opt

        i, best = 0, 1e32
        if args.restore and os.path.isfile(self.ckpt_f):
            i, best = self.restore(model, opt)

        model.train()
        tr_loss = list()
        while True:
            for data in tr_loader:
                x = data[args.data].to(device)
                if args.data == "audio":
                    x = th.nn.functional.interpolate(x, args.in_size)
                out = model(x)
                loss_val = model.loss(
                    *out,
                    (x * 255).long().squeeze() if args.data == "seg" else x,
                    opt=opt,
                    loss_fn=self.loss_fn,
                    top_k=args.top_k
                )
                tr_loss.append(loss_val)
                i += 1
                if (i % args.ifr) == 0:
                    model.eval()
                    if args.log_dir:
                        self.save_image(x, out[1], "train.jpg")
                    tr_loss = np.average(tr_loss)
                    print(i, tr_loss)
                    if args.log_dir:
                        self.writer.add_scalar("loss/train", tr_loss, i)
                        if tr_loss < best:
                            best = tr_loss
                            th.save(
                                {
                                    "ch": self.ch,
                                    "hid_dim": args.vq_hid_dim,
                                    "emb_dim": args.vq_emb_dim,
                                    "emb_num": args.vq_emb_num,
                                    "model": model.state_dict(),
                                    "opt": opt.state_dict(),
                                    "best": best,
                                    "iter": i,
                                },
                                self.ckpt_f,
                            )
                    tr_loss = list()
                    model.train()


class TransformTrain:
    def __init__(self, args, writer, ckpt_f, best_f):
        self.args = args
        self.writer = writer
        self.ckpt_f = ckpt_f
        self.best_f = best_f
        self.metrics = Metrics(args.to, size=RES_SIZE[args.to])
        self.logger = Logger(args.to, writer, METRIC_LOG_KEYS[args.to])

        lr = args.learning_rate
        in_ch = CH[args.dataset][args.frm]
        out_ch = CH[args.dataset][args.to]
        transform = TRANSFORM[args.transform]
        manifold = VAEManifold if args.manifold == "vae" else Manifold
        self.vq = (
            manifold(
                out_ch,
                19 if args.to == "seg" else out_ch,
                args.in_size,
                args.out_size,
                ckpt_f=args.vq_to,
            )
            .to(device)
            .eval()
        )
        self.model = transform(in_ch, args.vq_emb_dim, args.in_size, args.out_size,).to(device)
        self.opt = th.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = th.nn.functional.mse_loss

    def save_image(self, r, t, y, out_f, N=16, nrow=1):
        r, t, y = r.detach().cpu(), t.detach().cpu(), y.cpu()
        if self.args.to == "seg":
            r = r.argmax(1, keepdims=True)
            t = t.argmax(1, keepdims=True)
            r = decode(r) / 255
            t = decode(t) / 255
            y = decode(y * 255) / 255
        thv.utils.save_image(
            th.cat((y[:N, :3], t[:N, :3], r[:N, :3]), -1),
            os.path.join(self.args.log_dir, out_f),
            nrow=nrow,
        )

    def restore(self, ckpt_f):
        ckpt = th.load(ckpt_f, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["opt"])
        iter = ckpt["iter"]
        best = ckpt["best"]
        return iter, best

    def criterion(self, o, t):
        loss_fn = self.loss_fn
        if self.args.top_k:
            loss = loss_fn(o, t, reduce=False)
            loss = loss.mean(axis=list(range(len(loss.shape)))[1:])
            loss, _ = th.topk(loss, self.args.top_k)
            loss = loss.mean()
        else:
            loss = loss_fn(o, t)
        return loss

    @th.no_grad()
    def rec_criterion(self, p, t):
        if self.args.to == "seg":
            p = p.argmax(axis=1, keepdims=True) / 255.0
        val = self.loss_fn(p.detach().clone(), t.detach().clone()).item()
        return val

    @th.no_grad()
    def evaluate(self, te_loader):
        model, vq = self.model, self.vq
        te_loss, te_rec_loss, te_metrics = list(), list(), list()
        for i, data in enumerate(te_loader):
            x = data[args.frm].to(device)
            y = data[args.to].to(device)
            if args.frm == "audio":
                x = th.nn.functional.interpolate(x, args.in_size)
            o, t = model(x), vq.encode(y)
            loss_val = self.loss_fn(o, t).item()
            te_loss.append(loss_val)
            r = vq.decode(o)
            te_rec_loss.append(self.rec_criterion(r, y))
            te_metrics.append(self.metrics(r, y))
            if i == 0:
                out_r = r.detach().clone()
                out_t = vq.decode(t.detach().clone())
                out_y = y.detach().clone()
        te_loss = np.array(te_loss)
        te_rec_loss = np.array(te_rec_loss)
        te_metrics = np.concatenate(te_metrics)
        return te_loss, te_rec_loss, te_metrics, out_r, out_t, out_y

    @th.no_grad()
    def test(self, te_loader):
        self.restore()
        self.model.eval()
        # _, _, metrics, _, _, _ = self.evaluate(te_loader)
        # iou = np.nanmean(metrics, axis=0)
        # asd = " ".join(["{:.4f}".format(i) for i in iou])
        # print(args.log_dir, asd)

    def __call__(self, tr_loader, te_loader):
        args, model, vq, opt = self.args, self.model, self.vq, self.opt
        out_image = self.save_image
        i, best, best_fn = 0, BEST[args.to], BEST_FN[args.to]
        if self.args.restore and os.path.isfile(self.ckpt_f):
            i, best = self.restore()

        model.train()
        tr_loss, tr_rec_loss = list(), list()
        while True:
            for data in tr_loader:
                x = data[args.frm].to(device)
                y = data[args.to].to(device)
                if args.frm == "audio":
                    x = th.nn.functional.interpolate(x, args.in_size)
                o, t = model(x), vq.encode(y)
                opt.zero_grad()
                loss = self.criterion(o, t)
                loss.backward()
                opt.step()
                tr_loss.append(loss.item())
                r = vq.decode(o)
                tr_rec_loss.append(self.rec_criterion(r, y))
                i += 1
                if (i % args.ifr) == 0:
                    model.eval()
                    tr_loss = np.array(tr_loss)
                    tr_rec_loss = np.array(tr_rec_loss)
                    if args.log_dir:
                        out_image(r, vq.decode(t), y, "train.jpg")
                    te_loss, te_rec_loss, te_metrics, o, t, y = self.evaluate(te_loader)
                    score = self.logger(i, tr_loss, tr_rec_loss, te_loss, te_rec_loss, te_metrics,)
                    if best_fn(score, best):
                        best = score
                    if args.log_dir:
                        out_image(r, t, y, "test.jpg")
                        ckpt = {
                            "model": model.state_dict(),
                            "opt": opt.state_dict(),
                            "iter": i,
                            "best": best,
                        }
                        th.save(ckpt, self.ckpt_f)
                        if score == best:
                            th.save(ckpt, self.best_f)
                            out_image(r, t, y, "test_best.jpg")
                    tr_loss, tr_rec_loss = list(), list()
                    model.train()

    @th.no_grad()
    def dump(self, te_loader):
        import tqdm

        args = self.args
        dump_dir = os.path.join(args.log_dir, "dump")
        if not os.path.isdir(dump_dir):
            os.makedirs(dump_dir)

        self.restore(self.ckpt_f)
        self.model.eval()
        for data in tqdm.tqdm(te_loader):
            x = data[args.frm].to(device)
            y = data[args.to].to(device)
            ids = data["ids"]
            if args.frm == "audio":
                x = th.nn.functional.interpolate(x, args.in_size)
            r = self.vq.decode(self.model(x))
            r, y = r.cpu(), y.cpu()
            for i, rec, gt in zip(ids, r, y):
                out_f = os.path.join(dump_dir, i + ".jpg")
                if args.to == "seg":
                    rec = rec.argmax(0, keepdims=True)
                    rec = rec.unsqueeze(0)
                    gt = gt.unsqueeze(0)
                    rec = decode(rec) / 255
                    gt = decode(gt * 255) / 255
                    rec = rec.squeeze(0)
                    gt = gt.squeeze(0)
                thv.utils.save_image(th.cat((gt, rec), -1), out_f)


class EndtoendTrain:
    def __init__(self, args, writer, ckpt_f, best_f):
        self.args = args
        self.writer = writer
        self.ckpt_f = ckpt_f
        self.best_f = best_f
        self.metrics = Metrics(args.to, size=RES_SIZE[args.to])
        self.logger = Logger(args.to, self.writer, METRIC_LOG_KEYS[args.to])
        in_ch = CH[args.dataset][args.frm]
        out_ch = CH[args.dataset][args.to]
        out_ch = 19 if args.to == "seg" else out_ch
        model = TRANSFORM[args.transform](in_ch, args.vq_emb_dim, args.in_size, args.out_size,).to(
            device
        )
        dec = Builder(
            decoder(out_ch, args.vq_hid_dim, args.vq_emb_dim, args.out_size, args.in_size,)
        ).to(device)
        self.model = th.nn.Sequential(model, dec)
        self.opt = th.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.loss_fn = th.nn.functional.mse_loss
        if args.to == "seg":
            self.loss_fn = th.nn.functional.cross_entropy
        if args.data == "seg":
            self.loss_fn = th.nn.functional.cross_entropy

    def restore(self):
        if os.path.isfile(self.ckpt_f):
            ckpt = th.load(self.ckpt_f)
            self.model.load_state_dict(ckpt["model"])
            self.opt.load_state_dict(ckpt["opt"])
            iter = ckpt["iter"]
            best = ckpt["best"]
        else:
            print("Restore failed")
        return iter, best

    def save_image(self, r, t, y, out_f, N=16, nrow=1):
        r, t, y = r.detach().cpu(), t.detach().cpu(), y.cpu()
        if self.args.to == "seg":
            r = r.argmax(1, keepdims=True)
            t = t.argmax(1, keepdims=True)
            r = decode(r) / 255
            t = decode(t * 255) / 255
            y = decode(y * 255) / 255
        thv.utils.save_image(
            th.cat((y[:N, :3], t[:N, :3], r[:N, :3]), -1),
            os.path.join(self.args.log_dir, out_f),
            nrow=nrow,
        )

    def criterion(self, o, t):
        loss_fn = self.loss_fn
        if self.args.top_k:
            loss = loss_fn(o, t, reduce=False)
            loss = loss.mean(axis=list(range(len(loss.shape)))[1:])
            loss, _ = th.topk(loss, self.args.top_k)
            loss = loss.mean()
        else:
            if self.args.to == "seg":
                loss = loss_fn(o, (t * 255).type(th.long).squeeze(1))
            else:
                loss = loss_fn(o, t)
        return loss

    @th.no_grad()
    def rec_criterion(self, p, t):
        if self.args.to == "seg":
            p = p.argmax(axis=1, keepdims=True) / 255.0
        val = self.loss_fn(p.detach().clone(), t.detach().clone()).item()
        return val

    @th.no_grad()
    def evaluate(self, te_loader):
        model = self.model
        criterion = self.criterion
        metrics = self.metrics
        te_loss, te_metrics = list(), list()
        for i, data in enumerate(te_loader):
            x = data[args.frm].to(device)
            y = data[args.to].to(device)
            if args.frm == "audio":
                x = th.nn.functional.interpolate(x, args.in_size)
            o = model(x)
            loss = criterion(o, y)
            te_loss.append(loss.item())
            te_metrics.append(metrics(o, y))
            if i == 0:
                out_o = o.detach().clone()
                out_y = y.detach().clone()
        te_loss = np.array(te_loss)
        te_metrics = np.concatenate(te_metrics)
        return te_loss, te_metrics, out_o, out_y

    def __call__(self, tr_loader, te_loader):
        args = self.args
        model = self.model
        opt = self.opt
        out_image = self.save_image

        best, best_fn = BEST[args.to], BEST_FN[args.to]
        i, tr_loss = 0, list()
        if args.restore and os.path.isfile(self.ckpt_f):
            i, best = self.restore()

        model.train()
        tr_loss = list()
        while True:
            for data in tr_loader:
                x = data[args.frm].to(device)
                y = data[args.to].to(device)
                if args.frm == "audio":
                    x = th.nn.functional.interpolate(x, args.in_size)
                o = model(x)
                opt.zero_grad()
                loss = self.criterion(o, y)
                loss.backward()
                opt.step()
                tr_loss.append(loss.item())
                i += 1
                if (i % args.ifr) == 0:
                    model.eval()
                    tr_loss = np.array(tr_loss)
                    if args.log_dir:
                        out_image(o, y, y, "train.jpg")
                    te_loss, te_metrics, o, y = self.evaluate(te_loader)
                    score = self.logger(
                        i, tr_loss, np.array([-1]), te_loss, np.array([-1]), te_metrics,
                    )
                    if best_fn(score, best):
                        best = score
                    if args.log_dir:
                        out_image(o, y, y, "test.jpg")
                        ckpt = {
                            "model": model.state_dict(),
                            "opt": opt.state_dict(),
                            "iter": i,
                            "best": best,
                        }
                        th.save(ckpt, self.ckpt_f)
                        if score == best:
                            out_image(o, y, y, "test_best.jpg")
                            th.save(ckpt, self.best_f)
                    tr_loss = list()
                    model.train()


class ECHO2DEPTHTrain:
    def __init__(self, args, writer, ckpt_f, best_f):
        self.args = args
        self.writer = writer
        self.ckpt_f = ckpt_f
        self.best_f = best_f
        self.metrics = Metrics(args.to, size=RES_SIZE[args.to])
        self.logger = Logger(args.to, self.writer, METRIC_LOG_KEYS[args.to])
        in_ch = CH[args.dataset][args.frm]
        out_ch = CH[args.dataset][args.to]
        out_ch = 19 if args.to == "seg" else out_ch

        assert in_ch == args.input_spec_channels

        self.model = echo2depth(num_channels=args.input_spec_channels).to(device)

        self.loss_fn = echo2depth_loss

        self.opt = th.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0005)

        """
        self.loss_fn = th.nn.functional.mse_loss
        if args.data == "seg":
            self.loss_fn = th.nn.functional.cross_entropy
        """

    def restore(self):
        if os.path.isfile(self.ckpt_f):
            ckpt = th.load(self.ckpt_f)
            self.model.load_state_dict(ckpt["model"])
            self.opt.load_state_dict(ckpt["opt"])
            iter = ckpt["iter"]
            best = ckpt["best"]
        else:
            print("Restore failed")
        return iter, best

    def save_image(self, r, t, y, out_f, N=16, nrow=1):
        r, t, y = r.detach().cpu(), t.detach().cpu(), y.cpu()
        if self.args.to == "seg":
            r = r.argmax(1, keepdims=True)
            t = t.argmax(1, keepdims=True)
            r = decode(r) / 255
            t = decode(t * 255) / 255
            y = decode(y * 255) / 255
        thv.utils.save_image(
            th.cat((y[:N, :3], t[:N, :3], r[:N, :3]), -1),
            os.path.join(self.args.log_dir, out_f),
            nrow=nrow,
        )

    def criterion(self, o, t):
        loss_fn = self.loss_fn
        if self.args.top_k:
            loss = loss_fn(o, t, reduce=False)
            loss = loss.mean(axis=list(range(len(loss.shape)))[1:])
            loss, _ = th.topk(loss, self.args.top_k)
            loss = loss.mean()
        else:
            loss = loss_fn(o, t)
        return loss

    @th.no_grad()
    def rec_criterion(self, p, t):
        if self.args.to == "seg":
            p = p.argmax(axis=1, keepdims=True) / 255.0
        val = self.loss_fn(p.detach().clone(), t.detach().clone()).item()
        return val

    @th.no_grad()
    def evaluate(self, te_loader):
        model = self.model
        criterion = self.criterion
        metrics = self.metrics
        te_loss, te_metrics = list(), list()
        for i, data in enumerate(te_loader):
            x = data[args.frm].to(device)
            y = data[args.to].to(device)
            if args.frm == "audio":
                x = th.nn.functional.interpolate(x, args.in_size)
            o = model(x)
            loss = criterion(o, y)
            te_loss.append(loss.item())
            te_metrics.append(metrics(o, y))
            if i == 0:
                out_o = o.detach().clone()
                out_y = y.detach().clone()
        te_loss = np.array(te_loss)
        te_metrics = np.concatenate(te_metrics)
        return te_loss, te_metrics, out_o, out_y

    def __call__(self, tr_loader, te_loader):
        args = self.args
        model = self.model
        opt = self.opt
        out_image = self.save_image

        best, best_fn = BEST[args.to], BEST_FN[args.to]
        i, tr_loss = 0, list()
        if args.restore and os.path.isfile(self.ckpt_f):
            i, best = self.restore()

        model.train()
        tr_loss = list()
        while True:
            for data in tr_loader:
                x = data[args.frm].to(device)
                y = data[args.to].to(device)
                if args.frm == "audio":
                    x = th.nn.functional.interpolate(x, args.in_size)
                o = model(x)
                opt.zero_grad()
                loss = self.criterion(o, y)
                loss.backward()
                opt.step()
                tr_loss.append(loss.item())
                i += 1
                if (i % args.ifr) == 0:
                    model.eval()
                    tr_loss = np.array(tr_loss)
                    if args.log_dir:
                        out_image(o, y, y, "train.jpg")
                    te_loss, te_metrics, o, y = self.evaluate(te_loader)
                    score = self.logger(
                        i, tr_loss, np.array([-1]), te_loss, np.array([-1]), te_metrics,
                    )
                    if best_fn(score, best):
                        best = score
                    if args.log_dir:
                        out_image(o, y, y, "test.jpg")
                        ckpt = {
                            "model": model.state_dict(),
                            "opt": opt.state_dict(),
                            "iter": i,
                            "best": best,
                        }
                        th.save(ckpt, self.ckpt_f)
                        if score == best:
                            out_image(o, y, y, "test_best.jpg")
                            th.save(ckpt, self.best_f)
                    tr_loss = list()
                    model.train()


class Train:
    def __init__(self, args):
        self.args = args
        self.writer = None
        self.ckpt_f = None
        self.best_f = None
        if args.log_dir:
            if not args.restore and not args.mode == "dump":
                print("Deleting previous training runs")
                for fil in glob(os.path.join(args.log_dir, "*tfevents*")):
                    os.remove(fil)
                for fil in glob(os.path.join(args.log_dir, "*.pyth")):
                    os.remove(fil)
                for fil in glob(os.path.join(args.log_dir, "*.jpg")):
                    os.remove(fil)
            self.writer = SummaryWriter(args.log_dir)
            self.ckpt_f = os.path.join(args.log_dir, "ckpt.pyth")
            self.best_f = os.path.join(args.log_dir, "best.pyth")

    def trans_setup(self):
        args = self.args
        audio_sr = AUDIO_SR[args.dataset]
        if args.dataset == "ytasmr":
            audio_tran = th.nn.Sequential(
                tha.transforms.Spectrogram(
                    n_fft=512,
                    hop_length=256 // 2,
                    power=2,
                    normalized=True,
                    window_fn=th.hann_window,
                ),
                tha.transforms.AmplitudeToDB(),
                # AudioResizeNormalize((256, 256)),
            )
        elif args.dataset == "eth" or args.dataset == "mavd":
            audio_tran = th.nn.Sequential(
                tha.transforms.MelSpectrogram(
                    sample_rate=audio_sr, n_fft=2048, hop_length=256, n_mels=256
                ),
                tha.transforms.AmplitudeToDB(),
            )
        elif args.dataset == "mavd":
            audio_tran = th.nn.Sequential(
                tha.transforms.MelSpectrogram(
                    sample_rate=audio_sr, n_fft=512, hop_length=128, n_mels=256
                ),
                tha.transforms.AmplitudeToDB(),
            )
        else:
            raise ValueError
        video_tran = None
        return audio_tran, video_tran

    def ids_setup(self, sort=True):
        args = self.args
        audio_sr = AUDIO_SR[args.dataset]
        audio_wl = AUDIO_WL[args.dataset]
        tr_ids = readlines(os.path.join(args.dataset, "data/tr_ids.txt"))
        te_ids = readlines(os.path.join(args.dataset, "data/te_ids.txt"))
        if args.toy:
            tr_ids = tr_ids[:1] * 1000

        if args.mode == "manifold":
            tr_ids += te_ids

        if args.dataset != "mavd":
            ids = list()
            audio_f = os.path.join(args.dataset, "data/audio.h5")
            skip = 0  # if args.dataset == "ytasmr" else 1
            with h5py.File(audio_f, "r") as a_f:
                for vid in te_ids:
                    dur = a_f[vid].shape[1] / audio_sr
                    pos = np.arange(skip, dur - audio_wl, audio_wl) + audio_wl / 2
                    ids += [[vid, p] for p in pos]
            te_ids = ids

            ids = list()
            audio_f = os.path.join(args.dataset, "data/audio.h5")
            skip = 0  # if args.dataset == "ytasmr" else 1
            with h5py.File(audio_f, "r") as a_f:
                for vid in tr_ids:
                    dur = a_f[vid].shape[1] / audio_sr
                    pos = np.arange(skip, dur - audio_wl, audio_wl) + audio_wl / 2
                    ids += [[vid, p] for p in pos]
            tr_ids = ids
        else:
            if args.mode == "video":
                moving = readlines("mavd/data/te_moving.txt")
                te_ids = [x for x in te_ids if x.split(os.sep)[0] in moving]

        if sort:
            sorted(te_ids)

        return tr_ids, te_ids

    def loader_setup(self, tr_ids, te_ids):
        args = self.args
        audio_sr = AUDIO_SR[args.dataset]
        audio_wl = AUDIO_WL[args.dataset]
        video_fps = VIDEO_FPS[args.dataset]
        if args.dataset == "eth" and (
            args.to == "depth" or args.to == "depth-seg" or args.data == "depth"
        ):
            video_fps = 1
        print("video_fps", video_fps)

        audio_tran, video_tran = self.trans_setup()
        dataset = MavdDataset if args.dataset == "mavd" else Dataset

        if tr_ids:
            print("num tr_ids", len(tr_ids))
        if te_ids:
            print("num te_ids", len(te_ids))

        if tr_ids is not None:
            tr_dataset = dataset(
                args.audio_f,
                args.video_f,
                args.depth_f,
                args.seg_f,
                args.bg_f,
                args.mask_f,
                ids=tr_ids,
                train=True,
                dataset=args.dataset,
                audio_wl=audio_wl,
                audio_tran=audio_tran,
                audio_sr=audio_sr,
                video_fps=video_fps,
                audio_size=(args.in_size, args.in_size),
                fix_t=True,
            )
            tr_loader = th.utils.data.DataLoader(
                tr_dataset,
                batch_size=args.batch_size,
                drop_last=True,
                shuffle=True,
                pin_memory=True,
                num_workers=args.cpus,
                collate_fn=Dataset.collate_fn,
            )
        else:
            tr_loader = None

        if te_ids is not None:
            te_dataset = dataset(
                args.audio_f,
                args.video_f,
                args.depth_f,
                args.seg_f,
                args.bg_f,
                args.mask_f,
                ids=te_ids,
                train=False,
                dataset=args.dataset,
                audio_wl=audio_wl,
                audio_tran=audio_tran,
                audio_sr=audio_sr,
                video_fps=video_fps,
                audio_size=(args.in_size, args.in_size),
                fix_t=True,
                ret_id=True if args.mode == "dump" else False,
            )
            te_loader = th.utils.data.DataLoader(
                te_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=args.cpus,
                collate_fn=Dataset.collate_fn,
            )
        else:
            te_loader = None

        return tr_loader, te_loader

    def manifold_train(self):
        args = self.args
        tr_ids, _ = self.ids_setup()
        tr_loader, _ = self.loader_setup(tr_ids, None)
        train = ManifoldTrain(args, self.writer, self.ckpt_f, self.best_f)
        train(tr_loader)

    def transform_train(self):
        args = self.args
        tr_ids, te_ids = self.ids_setup()
        tr_loader, te_loader = self.loader_setup(tr_ids, te_ids)
        train = TransformTrain(args, self.writer, self.ckpt_f, self.best_f)
        train(tr_loader, te_loader)

    def endtoend_train(self):
        args = self.args
        tr_ids, te_ids = self.ids_setup()
        tr_loader, te_loader = self.loader_setup(tr_ids, te_ids)
        train = EndtoendTrain(args, self.writer, self.ckpt_f, self.best_f)
        train(tr_loader, te_loader)

    def evaluate(self):
        args = self.args
        _, te_ids = self.ids_setup()
        _, te_loader = self.loader_setup(_, te_ids)
        test = TransformTrain(args, self.writer, self.ckpt_f, self.best_f)
        test.test(te_loader)

    def dump(self):
        args = self.args
        _, te_ids = self.ids_setup()
        _, te_loader = self.loader_setup(_, te_ids)
        TransformTrain(args, self.writer, self.ckpt_f, self.best_f).dump(te_loader)

    @th.no_grad()
    def video(self):
        args = self.args
        in_ch = CH[args.dataset][args.frm]
        out_ch = CH[args.dataset][args.to]
        out_root = os.path.join(args.log_dir, "out2")

        vq = (
            Manifold(
                args.vq_to, out_ch, 19 if args.to == "seg" else out_ch, args.in_size, args.out_size,
            )
            .to(device)
            .eval()
        )
        model = TRANSFORM[args.transform](in_ch, args.vq_emb_dim, args.in_size, args.out_size,)
        ckpt = th.load(self.best_f)
        model.load_state_dict(ckpt["model"])
        model = model.to(device).eval()

        tr_ids, te_ids = self.ids_setup()
        from tqdm import tqdm

        if args.dataset == "mavd":
            vid = list(set([x.split(os.sep)[0] for x in te_ids]))
        elif args.dataset == "ytasmr":
            vid = list(set([x[0] for x in te_ids]))

        for tid in tqdm(vid[:10]):
            if args.dataset == "mavd":
                ids = [x for x in te_ids if tid in x]
            elif args.dataset == "ytasmr":
                ids = [x for x in te_ids if tid in x[0]]

            _, te_loader = self.loader_setup(None, ids)
            for data in te_loader:
                x = data[args.frm].to(device)
                y = data[args.to].to(device)
                o = model(x)
                r = vq.decode(o)
                if args.to == "seg":
                    r = r.argmax(1, keepdims=True).cpu()
                    r = decode(r) / 255
                    y = decode(y.cpu() * 255) / 255

                out_imgs = th.cat((y[:, :3], r[:, :3]), -1).cpu()
                i = 0

                out_path = os.path.join(out_root, tid)
                if not os.path.isdir(out_path):
                    os.makedirs(out_path)

                for out_img in out_imgs:
                    out_f = os.path.join(out_path, str(i).zfill(8) + ".jpg")
                    thv.utils.save_image(out_img, out_f)
                    i += 1


def main(args):
    if args.mode == "manifold":
        Train(args).manifold_train()
    elif args.mode == "transform":
        Train(args).transform_train()
    elif args.mode == "endtoend":
        Train(args).endtoend_train()
    elif args.mode == "video":
        Train(args).video()
    elif args.mode == "eval":
        Train(args).evaluate()
    elif args.mode == "dump":
        Train(args).dump()
    else:
        raise ValueError


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_f", type=str, default=None)
    parser.add_argument("--video_f", type=str, default=None)
    parser.add_argument("--depth_f", type=str, default=None)
    parser.add_argument("--seg_f", type=str, default=None)
    parser.add_argument("--bg_f", type=str, default=None)
    parser.add_argument("--mask_f", type=str, default=None)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--transform", type=str, default="res")
    parser.add_argument("--data", type=str, required=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--frm", type=str, default=None)
    parser.add_argument("--num_channels", type=str, default=8)
    parser.add_argument("--to", type=str, default=None)
    parser.add_argument("--vq_frm", type=str, default=None)
    parser.add_argument("--vq_to", type=str, default=None)
    parser.add_argument("--in_size", type=int, default=256)
    parser.add_argument("--out_size", type=int, default=32)
    parser.add_argument("--vq_emb_num", type=int, default=64)
    parser.add_argument("--vq_emb_dim", type=int, default=64)
    parser.add_argument("--vq_hid_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--manifold", type=str, default="vqvae")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--toy", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--ifr", type=int, default=1000)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--restore", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
