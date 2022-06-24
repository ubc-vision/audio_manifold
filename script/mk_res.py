#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import os
import shutil
from pathlib import Path

from pandas import DataFrame


def main(args):
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    res = []
    for fil in Path(args.log_dir).rglob("*test.jpg"):
        _, _, in_res, exp, outto, _ = str(fil).split(os.sep)
        _, net, _, out_res, _, _ = exp.split("_")
        _, out_res = out_res.replace("res", "").split("x")
        _, outto = outto.split("-")
        el = {
            "tran": net,
            "in_res": int(in_res),
            "out_res": int(out_res),
            "to": outto,
            "fil": str(fil),
        }
        res.append(el)

    df = DataFrame(res)

    html = "<html>\n<body>\n"
    for tran in ["old", "res", "shallow"]:
        sub = df[df["tran"] == tran]
        sub = sub.sort_values(by=["out_res"])
        html += "<p>\n"
        for _, asd in sub.iterrows():
            in_fil = asd["fil"]
            out_fil = in_fil.replace(os.sep, "-")
            shutil.copyfile(in_fil, os.path.join(args.out_dir, out_fil))
            html += "<img src={} width=128>\n".format(out_fil)
        html += "</p>\n"
    html += "</body>\n</html>"
    with open(os.path.join(args.out_dir, "index.html"), "w") as f:
        f.write(html)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
