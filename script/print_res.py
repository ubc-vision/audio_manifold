#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import os
from glob import glob

import pandas as pd


def parse(fil):
    fil = fil.replace(".csv", "")
    f = os.path.basename(fil)
    f = f.replace("hm-", "hm")
    _, insz, exp, frm, to = f.split("-")
    _, net, res, _, _ = exp.replace("__", "_").split("_")
    res = res.split("x")[-1]
    return "{}-{}-{}".format(insz, net, res)


def main(args):
    glb = "*{}*.csv".format(args.exp)
    for csv_f in glob(os.path.join(args.csv_dir, glb)):
        exp = parse(csv_f)
        df = pd.read_csv(csv_f)
        tags = [t for t in df.keys().tolist() if "metric" in t]
        str = [exp]
        for t in tags:
            str += ["{}:{},{}".format(t, df[t].min(), df[t].max())]
        print(" ".join(str))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True)
    parser.add_argument("--exp", type=str, required=True)
    args = parser.parse_args()
    main(args)
