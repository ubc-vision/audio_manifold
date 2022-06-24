#!/bin/bash

if [ "$#" -ne 5 ]; then
    echo "usage: test.sh <log> <manifold> <tran> <insize> <outsize>"
    exit -1
fi

log=$1
manifold=$2
tran=$3
insize=$4
r=$5
res=res"$r"x"$r"

python train.py --mode eval \
    --dataset eth \
    --frm audio \
    --to seg \
    --audio_f eth/data/audio.h5 \
    --seg_f eth/data/seg.h5 \
    --manifold $manifold \
    --transform $tran \
    --vq_to $log/"$manifold"_"$res"_num64_dim64/seg/ckpt.pyth \
    --log_dir $log/mft_"$manifold"_"$tran"_"$res"_num64_dim64/audio-seg/ \
    --in_size $insize \
    --out_size $r

