import argparse
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        choices=["nanhai", "bohai"],
        default="nanhai",
        help="List array keys in data/<dataset>.npz",
    )
    args = p.parse_args()
    root = Path(__file__).resolve().parent.parent.parent
    file_path = root / "data" / f"{args.dataset}.npz"
    data = np.load(file_path, allow_pickle=True)
    print(file_path)
    print(data.files)
    for item in data.files:
        print(f"Contents of {item}: {data[item]}")
