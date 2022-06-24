# Estimating Visual Information from Sound through Manifold Learning
​
This is the code repository our paper [arXiv].
​
![](teaser.png)
​
## Setup
​
### Dependencies
​
We suggest to use `miniconda` to setup an environment for this project.
​
- pytorch
- torchvision
- torchaudio
- python-opencv
- h5py
- tensorboard


```conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch```
​
```conda install python-opencv h5py tensorboard -c conda-forge```
​
### Datasets
​
The OAP and MAVD datasets must be converted into our `hdf5` format. For audio and each one of the visual modality: RGB, Depth, and Semantic segmentation, we compile a specific h5 file. Inside these files data samples are simply indexed by their original names.
In our experiments we use images at lower resolution w.r.t the original dataset, for ease the loading of data we save resized images 256x256 and encode in jpeg for RGB, png for semantic segmentation and, float for depth.
Instead for audio we downsample the original audio to 16khz and save it asfloat normalized to [-1, +1] in the h5 file.
​
### Preparation scripts
​
The OAP dataset can be downloaded by following the instructions [here](https://github.com/arunbalajeev/binaural-sound-perception). The MAVD dataset can be downloaded by following the instructions [here](https://github.com/robot-learning-freiburg/MM-DistillNet).

Instructions for 
​
## Training
​
### 1. VQ-VAE
​
Our method works in two stage. The first stage consist in learning the quantized manifold of the interested visual modality. For doing this:
​
```python trian.py --mode manifold --manifold (vqvae|vae) --dataset (eth|mavd) --data (depth|seg) (--depth_f <h5 file> | --seg_f <h5 file>) --vq_emb_num <num> --vq_emb_dim <num> --in_size <input size> --out_size <manifold size> --batch_size <num> --ifr <num> --cpus <num> --log_dir <dir>```
​
### 2. Audio Manifold Transform
​
```python trian.py --mode transofrm --manifold (vqvae|vae) --dataset (eth|mavd) --data (depth|seg) --audio_f <h5 file> (--depth_f <h5 file> | --seg_f <h5 file>) --frm audio --to depth --vq_to <ckpt file> --vq_emb_num <num> --vq_emb_dim <num> --batch_size <num> --ifr <num> --in_size <input size> --out_size <manifold size> --cpus <num> --log_dir <dir> ```
​
## Testing
​
## Cite
​
[arXiv]# audio_manifold
