from io import BytesIO
import argparse
import glob
import h5py
import pickle
import os
import cv2
import librosa

import numpy as np

from pprint import pprint
from tqdm import tqdm

def readlines(in_f):
    with open(in_f, "r") as f:
        x = f.read().split("\n")[:-1]

    return x

def main(args):
    jpg_files = glob.glob(os.path.join(
        args.dir,
        'fl_rgb/*jpg' if 'drive' in args.dir else '*/fl_rgb/*jpg',
    ))

    print("Number of jpg files")
    print(len(jpg_files))

    rgb_frames = {}
    rgb_frame_names = []

    for jpg_file in tqdm(jpg_files):
        jpg_file_id = ''
        jpg_file_path, jpg_file_name = os.path.split(jpg_file)
        jpg_file_path, _ = os.path.split(jpg_file_path)
        jpg_file_path = os.path.split(jpg_file_path)[1]
        jpg_id = jpg_file_path + '/' +  jpg_file_name[7:-4]
        # print(jpg_id)
        # print(jpg_file_name)
        # print(jpg_file)
        rgb_frame_names.append(jpg_id)
        if jpg_id in rgb_frames.keys():
            rgb_frames[jpg_id].append(jpg_file_name)
        else:
            rgb_frames[jpg_id] = [jpg_file_name]

    for value in rgb_frames.values():
        assert len(value) == 1
    
    with h5py.File(os.path.join(args.dir,'mavd_rgb_dayhdf5'), 'a') as f:
        for i, key in enumerate(ids_to_process):
            drive_name, timestamp = os.path.split(key)
            current_recordings = glob.glob(os.path.join(
                args.dir,
                drive_name,
                'fl_rgb',
                '*' + timestamp + '*' + '.jpg'
            ))
            
            assert len(current_recordings) == 1

            for j, file_name in enumerate(current_recordings):
                rgb = cv2.imread(file_name)
                f[key] = rgb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get the audio inputs')
    parser.add_argument('--dir', help='The drive directory or the dataset path')
    parser.add_argument('--ids', help='A text file containing the ids to process')

    args = parser.parse_args()

    main(args)

