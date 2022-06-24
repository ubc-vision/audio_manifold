#!/usr/bin/env python

head = "#!/bin/bash \n"
head += "#SBATCH --time 03:00:00\n"
head += "#SBATCH --mem 16GB\n"
head += "#SBATCH --cpus-per-task 4\n"
head += "#SBATCH --gres=gpu:1\n"
head += "#SBATCH --account=rrg-kyi\n"
head += "cd ~/src/eth \n"


def mk_script(args, out_size, vq_emb_num, vq_emb_dim):
    dataset = args.dataset
    manifold = args.manifold
    trans = args.transform
    in_size = args.in_size
    exp = "res{}x{}_num{}_dim{}".format(
        out_size, out_size, vq_emb_num, vq_emb_dim
    )
    files = {
        "audio": "--audio_f {}/data/audio.h5 ".format(dataset),
        "video": "--video_f {}/data/video.h5 ".format(dataset),
        "depth": "--depth_f {}/data/disp.h5 ".format(dataset),
        "seg": "--seg_f {}/data/seg.h5 ".format(dataset),
    }
    for frm, to in [("audio", "video"), ("audio", "depth"), ("audio", "seg")]:
        script = head
        script += "python -W ignore train.py "
        script += "--mode transform "
        script += "--dataset {} ".format(dataset)
        script += files[frm]
        script += files[to]
        if dataset == "ytasmr":
            script += "--mask_f {}/data/mask.h5 ".format(dataset)
        script += "--frm {} --to {} ".format(frm, to)
        script += "--vq_to {}/log/{}/{}_{}/{}/ckpt.pyth ".format(
            dataset, in_size, manifold, exp, to
        )
        script += "--vq_emb_num {} ".format(vq_emb_num)
        script += "--vq_emb_dim {} ".format(vq_emb_dim)
        script += "--batch_size 64 "
        script += "--ifr 100 "
        script += "--in_size {} --out_size {} ".format(in_size, out_size)
        script += "--transform {} ".format(trans)
        script += "--manifold {} ".format(manifold)
        # script += "--audio_aug "
        # script += "--top_k {} ".format(16)
        # script += "--restore "
        script += "--cpus 4 "
        script += "--log_dir {}/log/{}/mft_{}_{}_{}/{}-{} \n".format(
            dataset, in_size, manifold, trans, exp, frm, to
        )
        with open(
            "{}/job/{}/mft_{}_{}_{}_{}-{}.sh".format(
                dataset, in_size, manifold, trans, exp, frm, to
            ),
            "w",
        ) as f:
            f.write(script)


def main(args):
    for out_size in [8, 16, 32]:
        for vq_emb_num in [64]:
            for vq_emb_dim in [64]:
                print(args.dataset, out_size, vq_emb_num, vq_emb_dim)
                mk_script(args, out_size, vq_emb_num, vq_emb_dim)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--in_size", type=int, required=True)
    parser.add_argument("--transform", type=str, required="res")
    parser.add_argument("--manifold", type=str, default="vqvae")
    args = parser.parse_args()

    main(args)
