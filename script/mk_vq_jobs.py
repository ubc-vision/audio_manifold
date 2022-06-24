#!/usr/bin/env python
# -*- coding=utf-8 -*-

head = "#!/bin/bash \n"
head += "#SBATCH --time 03:00:00\n"
head += "#SBATCH --mem 16GB\n"
head += "#SBATCH --cpus-per-task 4\n"
head += "#SBATCH --gres=gpu:1\n"
head += "#SBATCH --account=rrg-kyi\n"
head += "cd ~/src/eth \n"


def mk_script(args, out_size, vq_emb_num, vq_emb_dim):
    dataset = args.dataset
    in_size = args.in_size
    manifold = args.manifold
    exp = "res{}x{}_num{}_dim{}".format(
        out_size, out_size, vq_emb_num, vq_emb_dim
    )
    for data in ["audio", "video", "depth", "seg", "depth-seg"]:
        if data in ["depth", "seg"] and dataset == "ytasmr":
            continue
        script = head
        script += "python -W ignore train.py "
        script += "--mode manifold "
        script += "--dataset {} ".format(dataset)
        if data == "audio":
            script += "--audio_f {}/data/audio.h5 ".format(dataset)
        elif data == "video":
            script += "--video_f {}/data/video.h5 ".format(dataset)
            if dataset == "ytasmr":
                script += "--mask_f {}/data/mask.h5 ".format(dataset)
        elif data == "seg":
            script += "--seg_f {}/data/seg.h5 ".format(dataset)
        elif data == "depth":
            script += "--depth_f {}/data/disp.h5 ".format(dataset)
        elif data == "depth-seg":
            script += "--depth_f {}/data/disp.h5 ".format(dataset)
            script += "--seg_f {}/data/seg.h5 ".format(dataset)
        else:
            pass
        script += "--data {} ".format(data)
        script += "--vq_emb_num {} ".format(vq_emb_num)
        script += "--vq_emb_dim {} ".format(vq_emb_dim)
        script += "--batch_size 64 "
        script += "--in_size {} --out_size {} ".format(in_size, out_size)
        script += "--manifold {} ".format(manifold)
        script += "--ifr 250 "
        # script += "--top_k 16 "
        script += "--log_dir {}/log/{}/{}_{}/{}/ \n".format(
            dataset, in_size, manifold, exp, data
        )
        with open(
            "{}/job/{}/{}_{}_{}.sh".format(
                dataset, in_size, manifold, exp, data
            ),
            "w",
        ) as f:
            f.write(script)


def main(args):
    for res in [8, 16, 32]:
        for vq_emb_num in [64, 128, 256, 512]:
            for vq_emb_dim in [64]:
                print(args.dataset, args.in_size, res, vq_emb_num, vq_emb_dim)
                mk_script(args, res, vq_emb_num, vq_emb_dim)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--in_size", type=int, required=True)
    parser.add_argument("--manifold", type=str, default="vqvae")
    args = parser.parse_args()

    main(args)
