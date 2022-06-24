#!/bin/bash

if [[ $# -ne 6 ]]; then
        echo "Usage: ./video.sh <db> <tran> <insz> <outsz> <from> <to>"
        exit -1
fi

db=$1
tran=$2
insz=$3
r=$4
from=$5
to=$6

res=res"$r"x"$r"
logdir=$db/log/$insz/mft_"$tran"__"$res"_num64_dim64/"$from"-"$to"

cd ~/src/eth
python -W ignore train.py \
        --mode video \
        --dataset $db \
        --audio_f $db/data/"$from".h5 \
        --"$to"_f $db/data/"$to".h5 \
        --frm $from \
        --to $to \
        --in_size $insz \
        --out_size $r \
        --vq_frm $db/log/$insz/vq_"$res"_num64_dim64/$from/ckpt.pyth \
        --vq_to $db/log/$insz/vq_"$res"_num64_dim64/$to/ckpt.pyth \
        --log_dir $logdir \
        --tran $tran \
        --cpus 2

for dir in $(ls $logdir/out2); do
        echo $dir
        ffmpeg -y -framerate 1 -i $logdir/out2/$dir/%08d.jpg \
                -r 25 -c:v libx264 -pix_fmt yuv420p \
                $logdir/out2/$dir.mp4
done

