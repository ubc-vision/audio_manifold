The OAP dataset can be downloaded by following the instructions [here](https://github.com/arunbalajeev/binaural-sound-perception). The MAVD dataset can be downloaded by following the instructions [here](https://github.com/robot-learning-freiburg/MM-DistillNet).

Our as in Vasudevan et al., method uses pretrained models to create pseudo-labels that our method uses as ground truth. The method to create the semantic segmentation labels is [DeepLavV3+](https://github.com/VainF/DeepLabV3Plus-Pytorch) and the method used to create depth maps is Monodepth2. We use pretrained models available from the code repositories.

After the ground truth images are created using the pretraiend models, we convert the images to [HDF5](https://www.h5py.org/) data format.

Download the [OAP](https://github.com/arunbalajeev/binaural-sound-perception))and [MAVD](https://github.com/robot-learning-freiburg/MM-DistillNet) datasets and clone the [DeepLabv3+](https://github.com/VainF/DeepLabV3Plus-Pytorch) and [Monodepth2](https://github.com/nianticlabs/monodepth2) repositories. For the DeepLabV3+ repository, download a pretrained model (we used best_deeplabv3plus_mobilenet_cityscapes_os16.pth) from the Google Drive link in the project description. 

For the OAP dataset, extract the data using the scripts provided in that repository, ie:

```
python extract_videosegments.py
python extract_spectrograms.py
```

**Note: we only use daytime scenes for the MAVD dataset. This is because our pseudo-ground truth pretrained models do not work on dark nighttime scenes.**

## OAP Depth

Replace ```YOUR_PATH_HERE``` with the path to your clone of binaural-sound-perception (and the dataset downloaded to it). Use the file test_simple.py from the monodepth2 repository.

```
compute_oap.sh
```

## OAP Segmentaiton

Replace ```OAP_DATASET_PATH``` to the root of the dataset downloaed from the binaural-sound-perception repository.

```
python predict_oap.py --dataset cityscapes --dataset_path OAP_DATASET_PATH --model deeplabv3plus_mobilenet --ckpt pretrained_weights/best_deeplabv3plus_mobilenet_cityscapes_os16.pth
```

## MAVD Depth

Move the script ```test_simple_mavd.py``` to your monodepth2 repo. Replace ```PATH_TO_MAVD``` with your path to the MAVD RGB HDF5 file and ```PATH_TO_DEPTH``` and with the location you want the depth HDF5 to be written to. Then run:

```
python test_simple_mavd.py --model_name mono+stereo_1024x320 --rgb_f ../PATH_TO_MAVD/mavd_rgb.h5 --depth_f ../PATH_TO_DEPTH/dpeth.h5
```

## MAVD Segmentation

Move the script ```predict_mavd.py``` to your cloned DeepLabV3+ repository. Replace ```PATH_TO_MAVD_SEG``` the path to where you want the segmentaiton map file to, replace ```PATH_TO_MAVD``` with your path to the MAVD RGB file and '''PATH_TO_PRETRAINED_WEIGHTS''' to where you downloaded the pretrained weights to. Then run:

```
python predict_mavd.py --mask_f PATH_TO_MAVD_SEG/mavd_seg.h5 --rgb_f PATH_TO_MAVD/mavd_rgb.h5 --dataset cityscapes --model deeplabv3plus_mobilenet --ckpt PATH_TO_PRETRAINED_WEIGHTS/best_deeplabv3plus_mobilenet_cityscapes_os16.pth
```

## OAP Depth to HDF5

Replase ```OAP_DATASET_PATH``` with the path to your OAP dataset that was processed according to the instructions in its repositoy, then run:

```
python mk_oap_disp_h5.py --oap_path OAP_DATASET_PATH
```

## OAP Segmentation to HDF5

The OAP segmentation instructions will give HDF5 files by default.

## OAP Audio to HDF5

Replase ```OAP_DATASET_PATH``` with the path to your OAP dataset that was processed according to the instructions in its repositoy, then run:

```
python mk_oap_audio_h5.py --oap_path OAP_DATASET_PATH
```

## MAVD RGB to HDF5

Our methods for creating Segmentation and Depth for the MAVD dataset requires the MAVD dataset to be in a HDF5 file. Replace ``PATH_TO_MAVD_DIR``` with the path to the directory that you downloaded the MAVD dataset to. This function creates two HDF5 files, one with all RGB data and another with only daytime scenes. **Our paper uses daytime scenes only.**

```
python mavd_jpg_to_hdf5.py --dir PATH_TO_MAVD_dir --ids ids.txt
```

## MAVD Audio to HDF5

Replace ``PATH_TO_MAVD_DIR``` with the path to the directory that you downloaded the MAVD dataset to. This function creates two HDF5 files, one with all audio (mp3) data and another with only daytime scenes. 

```
python mavd_mp3_to_hdf5.py --dir PATH_TO_MAVD_dir --ids ids.txt