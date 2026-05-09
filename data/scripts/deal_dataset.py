"""
Build training NPZ files from grid CSVs for maritime datasets.

Output:
  data/<dataset>.npz  with array key ``data`` shaped [time_steps, num_nodes, num_features].

Default source CSV paths match the sample files shipped in this repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = {
    "nanhai": "data/nanhai_265.csv",
    "bohai": "data/bohai_300.csv",
}


def csv_to_spatiotemporal_array(csv_path: Path) -> np.ndarray:
    """Read numeric CSV; return float array [T, N] or [T, N, F]."""
    df = pd.read_csv(csv_path, low_memory=False)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    arr = df.to_numpy(dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D numeric table from {csv_path}, got shape {arr.shape}")
    # Single target channel per node: [T, N] -> [T, N, 1]
    arr = arr[:, :, np.newaxis]
    return arr


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    p = argparse.ArgumentParser(
        description="Convert maritime grid CSV to data/<dataset>.npz (key 'data')."
    )
    p.add_argument("--dataset", choices=["nanhai", "bohai"], required=True)
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to input CSV (default: bundled sample for the dataset).",
    )
    args = p.parse_args()

    csv_rel = args.csv or DEFAULT_CSV[args.dataset]
    csv_p = Path(csv_rel)
    if not csv_p.is_absolute():
        csv_p = root / csv_p

    if not csv_p.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_p}")

    data = csv_to_spatiotemporal_array(csv_p)
    out_npz = root / "data" / f"{args.dataset}.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, data=data)
    print(f"Wrote {out_npz} with data shape {data.shape}")


if __name__ == "__main__":
    main()
