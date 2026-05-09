import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def csv_to_npz(csv_file_path, npz_file_path):
    try:
        print(f"Reading CSV: {csv_file_path}")
        df = pd.read_csv(csv_file_path, low_memory=False)
        print(f"Read CSV with {df.shape[0]} rows and {df.shape[1]} columns.")

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        data_array = df.to_numpy()
        print("Converted DataFrame to NumPy array.")

        np.savez_compressed(npz_file_path, data=data_array)
        print(f"Saved compressed NPZ: {npz_file_path}")
    except FileNotFoundError:
        print(f"Error: CSV not found: {csv_file_path}.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent
    p = argparse.ArgumentParser(
        description="Flatten full CSV table to NPZ (generic; not the maritime grid script)."
    )
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument(
        "--out-npz",
        type=Path,
        default=None,
        help="Default: same basename as CSV in the same directory",
    )
    args = p.parse_args()
    csv_p = args.csv if args.csv.is_absolute() else root / args.csv
    out = args.out_npz
    if out is None:
        out = csv_p.with_suffix(".npz")
    elif not out.is_absolute():
        out = root / out
    csv_to_npz(str(csv_p), str(out))
