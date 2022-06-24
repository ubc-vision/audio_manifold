#!/usr/bin/env python

head = "#!/bin/bash \n"
head += "#SBATCH --time 03:00:00\n"
head += "#SBATCH --mem 16GB\n"
head += "#SBATCH --cpus-per-task 4\n"
head += "#SBATCH --gres=gpu:1\n"
head += "#SBATCH --account=rrg-kyi\n"
head += "cd ~/src/eth \n"

res = {5: "8x8", 4: "16x16", 3: "32x32"}


def mk_script(dataset, n_pool, vq_emb_num, vq_emb_dim):
    exp = "_res{}_num{}_dim{}".format(res[n_pool], 64, vq_emb_dim)

    script = head
    script += "python -W ignore train.py "
    script += "--mode vqvae "
    script += "--dataset {} ".format(dataset)
    script += "--audio_f {}/data/audio.h5 ".format(dataset)
    script += "--video_f {}/data/video_256.h5 ".format(dataset)
    script += "--frm audio --to video "
    script += "--vq_frm {}/log/vq{}/audio/ckpt.pth ".format(dataset, exp)
    script += "--vq_to {}/log/vq{}/video/ckpt.pth ".format(dataset, exp)
    script += "--vq_emb_num {} ".format(vq_emb_num)
    script += "--vq_emb_dim {} ".format(vq_emb_dim)
    script += "--batch_size 64 "
    script += "--ifr 100 "
    script += "--log_dir {}/log/mf{}/audio2video \n".format(dataset, exp)
    with open("{}/job/mf{}_audio2video.sh".format(dataset, exp), "w") as f:
        f.write(script)

    for data in ["depth"]:
        script = head
        script += "python -W ignore train.py "
        script += "--mode vq "
        script += "--dataset {} ".format(dataset)
        #script += "--audio_f {}/data/audio.h5 ".format(dataset)
        #script += "--video_f {}/data/video_256.h5 ".format(dataset)
        script += "--depth_f {}/data/depth_256.h5 ".format(dataset)
        script += "--data {} ".format(data)
        script += "--vq_emb_num {} ".format(vq_emb_num)
        script += "--vq_emb_dim {} ".format(vq_emb_dim)
        script += "--batch_size 64 "
        script += "--ifr 100 "
        script += "--n_pool {} ".format(n_pool)
        script += "--log_dir {}/log/vq{}/{}/ \n".format(dataset, exp, data)
        with open("{}/job/vq{}_{}.sh".format(dataset, exp, data), "w") as f:
            f.write(script)


def main():
    for dataset in ["eth"]:
        for n_pool in [3, 4, 5]:
            for vq_emb_num in [64]:
                for vq_emb_dim in [16, 32, 64]:
                    print(dataset, n_pool, vq_emb_num, vq_emb_dim)
                    mk_script(dataset, n_pool, vq_emb_num, vq_emb_dim)


if __name__ == "__main__":
    main()
