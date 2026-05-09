import argparse
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["nanhai", "bohai"], default="nanhai")
    args = p.parse_args()
    root = Path(__file__).resolve().parent.parent.parent
    npz_path = root / "data" / f"{args.dataset}.npz"
    with np.load(npz_path) as data:
        print(npz_path, data["data"].shape)
