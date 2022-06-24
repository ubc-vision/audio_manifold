#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import os
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)


def main(args):
    for path in Path(args.log_dir).rglob(args.exp):
        print(path)
        out_f = str(path).replace(os.sep, "-") + ".csv"
        out_f = out_f.replace("-log", "")
        out_f = os.path.join(args.out_dir, out_f)

        acc = EventAccumulator(str(path))
        acc.Reload()

        df = []
        for tag in acc.Tags()["scalars"]:
            events = acc.Scalars(tag)
            asd = [{"step": x.step, tag: x.value} for x in events]
            df.append(pd.DataFrame(asd))

        out = pd.concat(df, axis=1)
        out.to_csv(out_f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="res/csv")
    parser.add_argument("--exp", type=str, required=True)
    args = parser.parse_args()
    main(args)
