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
    out_file = args.oap_path + "/audio.h5"

    tracks = [
        "Track1",
        "Track2",
        "Track3",
        "Track4",
        "Track5",
        "Track6",
        "Track7"
        "Track8"
    ]

    done = []
    if os.path.isfile(out_file):
        with h5py.File(out_file, "r") as f:
            done = list(f.keys())

    for scene in tqdm(sorted(glob(args.oap_path + "/data/scene*"))):
        vid = os.path.basename(scene)
        if vid in done:
            continue
        fold = os.path.join(scene, args.oap_path + "spectrograms")
        
        audio = []

        for frame in sorted(glob(os.path.join(fold, "Track1", "*.npy"))):
            track_name, _ = os.path.split(frame)
            all_tracks = []
            for track in tracks:
                audio_file = os.path.join(fold, track, track_name)
                x = np.load(audio_file)[0, 0]
                all_tracks.append(x[np.newaxis])
            audio.append(np.array(all_tracks))sadxcfvgbhesdx

        audio = np.array(audio)
        with h5py.File(out_file, "a") as f:
            f[vid] = audio

if __name__ == "__main__":
    args = parse_args()
    main(args)