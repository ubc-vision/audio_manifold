#!/usr/bin/env python
import os

import cv2
import h5py
import numpy as np
import tqdm


def proc(vid, in_f, out_f, size=(256, 256)):
    frame_ids = sorted(map(int, in_f[vid].keys()))
    frame_ids = list(map(str, frame_ids))

    def dec_resize_enc(x):
        x = cv2.imdecode(x, -1)
        x = cv2.resize(x, size, interpolation=cv2.INTER_NEAREST)
        _, x = cv2.imencode(".png", x)
        x = x.tobytes()
        return x

    frames = [dec_resize_enc(in_f[vid][i][...]) for i in frame_ids]
    frames = np.array(frames, np.bytes_)
    with h5py.File(out_f, "a") as out:
        out[vid] = frames


def main(args):
    if os.path.isfile(args.out_f):
        with h5py.File(args.out_f, "r") as out_f:
            out_vids = list(out_f.keys())
    else:
        out_vids = []

    with h5py.File(args.in_f, "r") as in_f:
        vids = sorted(in_f.keys())
        if args.vid is not None:
            pass
        else:
            for vid in tqdm.tqdm(vids):
                if vid not in out_vids:
                    proc(vid, in_f, args.out_f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_f", type=str, required=True)
    parser.add_argument("--out_f", type=str, required=True)
    parser.add_argument("--vid", type=int, default=None)
    args = parser.parse_args()
    main(args)
