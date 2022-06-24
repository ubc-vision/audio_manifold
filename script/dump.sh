#!/bin/bash

if [ "$#" -ne 6 ]; then
    echo "usage: dump.sh <log> <manifold> <tran> <insize> <outsize> <modality>"
    exit -1
fi

log=$1
manifold=$2
tran=$3
insize=$4
r=$5
modality=$6
res=res"$r"x"$r"
db="mavd"

python train.py \
    --mode dump \
    --dataset $db \
    --frm audio \
    --to $modality \
    --audio_f $db/data/audio.h5 \
    --video_f $db/data/video.h5 \
    --seg_f $db/data/seg.h5 \
    --manifold $manifold \
    --transform $tran \
    --vq_to $log/"$manifold"_"$res"_num64_dim64/"$modality"/ckpt.pyth \
    --log_dir $log/mft_"$manifold"_"$tran"_"$res"_num64_dim64/audio-"$modality"/ \
    --in_size $insize \
    --out_size $r \
    --cpus 4


