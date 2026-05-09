import argparse
from pathlib import Path

import numpy as np


def verify_dataset(dataset: str):
    root = Path(__file__).resolve().parent.parent.parent
    data_path = root / "data" / f"{dataset}.npz"

    if not data_path.is_file():
        print(f"Data file not found: {data_path}")
        return None

    data_npz = np.load(str(data_path))
    data = data_npz["data"]

    print(f"[{dataset}] data shape: {data.shape}")
    print(f"Time steps: {data.shape[0]}")
    print(f"Nodes: {data.shape[1]}")
    print(f"Features: {data.shape[2]}")
    print(f"Value range: [{data.min():.4f}, {data.max():.4f}]")
    print(f"Mean: {data.mean():.4f}")

    return data.shape


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Inspect data/<dataset>.npz")
    p.add_argument("--dataset", choices=["nanhai", "bohai"], default="nanhai")
    args = p.parse_args()
    verify_dataset(args.dataset)
