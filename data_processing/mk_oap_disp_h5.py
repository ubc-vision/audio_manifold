#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import os
from glob import glob
import argparse

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--oap_path', type=str, required=True)

    return parser.parse_args()

def main(args):
    out_file = args.oap_path + "/disp.h5"

    done = []
    if os.path.isfile(out_file):
        with h5py.File(out_file, "r") as f:
            done = list(f.keys())

    for scene in tqdm(sorted(glob(args.oap_path + "/data/scene*"))):
        vid = os.path.basename(scene)
        if vid in done:
            continue
        fold = os.path.join(scene, args.oap_path + "monodepth2_front")
        asd = []
        for frame in sorted(glob(os.path.join(fold, "*.npy"))):
            x = np.load(frame)[0, 0]
            x = cv2.resize(x, (256, 256), interpolation=cv2.INTER_LINEAR)
            # _, x = cv2.imencode(".png", x)
            asd.append(x[np.newaxis])
        asd = np.array(asd)
        with h5py.File(out_file, "a") as f:
            f[vid] = asd

if __name__ == "__main__":
    args = parse_args()
    main(args)