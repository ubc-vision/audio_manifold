from io import BytesIO
import argparse
import glob
import h5py
import pickle
import os
import cv2
import librosa
import torch
import torchaudio

import numpy as np

from tqdm import tqdm

a_t = torch.nn.Sequential(
                    torchaudio.transforms.Spectrogram(
                        n_fft=512,
                        hop_length=128,
                        power=2,
                        normalized=True,
                        window_fn=torch.hann_window,
                    ),
                torchaudio.transforms.AmplitudeToDB(),
                )

def readlines(in_f):
    with open(in_f, "r") as f:
        x = f.read().split("\n")[:-1]

    return x

def main(args):
    mp3_files = glob.glob(os.path.join(
        args.dir,
        'audio/*mp3' if 'drive' in args.dir else '*/audio/*mp3',
    ))

    print("Number of mp3 files")
    print(len(mp3_files))
    
    mp3_frames = {}
    mp3_frame_names = []
    for mp3_file in mp3_files:
        mp3_file_id = ''
        mp3_file_path, mp3_file_name = os.path.split(mp3_file)
        mp3_file_path, _ = os.path.split(mp3_file_path)
        mp3_file_path = os.path.split(mp3_file_path)[1]
        mp3_id = mp3_file_path + '/' +  mp3_file_name[8:-4]
        if mp3_id in mp3_frames.keys():
            mp3_frames[mp3_id].append(mp3_file)
        else:
            mp3_frame_names.append(mp3_id)
            mp3_frames[mp3_id] = [mp3_file]

    for key, value in mp3_frames.items():
        value.sort()
        mp3_frames[key] = value
        if len(value) != 8:
            print(key)
        
        assert len(value) == 8

    with h5py.File(os.path.join(args.dir,'mavd_audio_day.hdf5'), 'w') as f:
        for i, key in enumerate(ids_to_process):
            assert key in mp3_frames.keys()

            value = mp3_frames[key] 

            assert len(value) == 8

            audio_np = np.array([])
            drive_name, timestamp = os.path.split(key)
            current_recordings = glob.glob(os.path.join(
                args.dir,
                drive_name,
                'audio',
                '*' + timestamp + '*' + '.mp3'
            ))

            # Make sure there are eight audio recordings per key
            if len(current_recordings) != 8:
                print(current_recordings)
                assert len(current_recordings) == 8
            try:
                for j, file_name in enumerate(current_recordings):
                    y, _ = librosa.load(file_name, sr=44100)
                    # unsqueeze
                    y = y[np.newaxis, :]
                    if j == 0:
                        audio_np = y
                    else:
                        audio_np = np.concatenate((audio_np, y), axis=0)

                # apply spectrogram representation with torchaudio
                audio = torch.from_numpy(audio_np)
                audio = a_t(audio)

                f[key] = audio.numpy()
            except:
                print("WARNING: unable to load {} fron numpy".format(str(key)))
                print(str(key))

            if i % 10000 == 0:
                print('Processed {} files'.format(i))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get the audio inputs')
    parser.add_argument('--dir', help='The drive directory or the dataset path')
    parser.add_argument('--ids', help='A text file containing the ids to process')
    parser.add_argument('--cpus', type=int)
    args = parser.parse_args()

    main(args)

