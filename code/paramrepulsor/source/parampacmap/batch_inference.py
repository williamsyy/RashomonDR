import argparse
import os
import re

import tqdm

from parampacmap.inference import main as infer

EXP_PATTERN = r"version_([0-9]*)"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchexpdir", type=str, required=True)
    return parser.parse_args()

def main(args):
    batch_exp_dir: str = args.batchexpdir
    exp_dirs = os.listdir(batch_exp_dir)
    for exp_dir in tqdm.tqdm(exp_dirs):
        if re.search(EXP_PATTERN, exp_dir) is not None:
            full_exp_dir = os.path.join(batch_exp_dir, exp_dir)
            args.expdir = full_exp_dir
            args.config = None
            try:
                infer(args)
            except TypeError as T:
                print(f"Failed to infer {exp_dir}")
                raise T


if __name__ == "__main__":
    args = get_args()
    main(args)
