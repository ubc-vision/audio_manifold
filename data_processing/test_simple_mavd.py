# Taken and modified from the monodepth2 repository
# Credit to that repository and the authors

from __future__ import absolute_import, division, print_function

import os
import sys
import cv2
import h5py
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=False)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')
    parser.add_argument("--rgb_f",
                        required=True)
    parser.add_argument("--depth_f",
                        required=True)

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.pred_metric_depth and "stereo" not in args.model_name:
        print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not in metric space.")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()
    
    rgb_ids = []
    processed_ids = []
    
    with h5py.File(args.rgb_f, 'r') as f:
        for j, group in enumerate(f.keys()):
            for k, key in enumerate(list(f[group].keys())):
                path = os.path.join(group, key)
                if not 'night' in str(path):
                    # print(path)
                    rgb_ids.append(
                        os.path.join(group, key)
                    )
    
    rgb_ids = set(rgb_ids)
    rgb_ids = list(rgb_ids)
    
    print('Number of RGB ids')
    print(len(rgb_ids))
    print("-> Predicting on {:d} test images".format(len(rgb_ids)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        with h5py.File(args.rgb_f, 'r') as f:
            with h5py.File(args.depth_f, 'a') as g:
                for idx, image_path in enumerate(rgb_ids):
                    # print(image_path)
                    image = f[image_path][...]
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    # print(image.shape)
                    # image = image[:, :, [2, 1, 0]]
                    
                    # cv2.imwrite('test.png', image) 
                    
                    input_image = pil.fromarray(image)
                    original_width, original_height = input_image.size
                    input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
                    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

                    # PREDICTION
                    input_image = input_image.to(device)
                    features = encoder(input_image)
                    outputs = depth_decoder(features)

                    disp = outputs[("disp", 0)]
                    disp_resized = torch.nn.functional.interpolate(
                        disp, (original_height, original_width), mode="bilinear", align_corners=False)

                    # Saving numpy file
                    # output_name = os.path.splitext(os.path.basename(image_path))[0]
                    scaled_disp, depth = disp_to_depth(disp, 0.1, 100)

                    if args.pred_metric_depth:
                        # name_dest_npy = os.path.join(output_directory, "{}_depth.npy".format(output_name))
                        metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
                        # np.save(name_dest_npy, metric_depth)
                        g[image_path] = metric_depth
                    else:
                        # name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
                        # np.save(name_dest_npy, scaled_disp.cpu().numpy())
                        g[image_path] = scaled_disp.cpu().numpy()
                    
                    processed_ids.append(image_path)

    print('Number of valid ids')
    print(len(rgb_ids))
    print('Number of depth files')
    print(len(processed_ids))
    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
