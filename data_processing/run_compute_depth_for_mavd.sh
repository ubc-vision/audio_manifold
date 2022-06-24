#!/bin/bash
#SBATCH --account=rrg-kyi
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=0-24:00
#SBATCH --gres=gpu:1 #1 gpu
#SBATCH --output=run_compute_depth_for_mavd.log  

/home/drydenw/projects/rrg-kyi/drydenw/monodepth2
python test_simple_mavd.py --model_name mono+stereo_1024x320 --rgb_f ../mavd/MAVD_dataset/mavd_rgb_256.h5 --depth_f ../mavd/MAVD_dataset/mavd_depth_256.h5
