import os
import numpy as np

from lib.paths import rel_to_root

# nanhai / bohai: NPZ at repo root data/{dataset}.npz
MARITIME_DATASETS = frozenset({"nanhai", "bohai"})


def maritime_npz_path(dataset: str) -> str:
    if dataset not in MARITIME_DATASETS:
        raise ValueError(
            f"Unknown maritime dataset {dataset!r}; supported: {', '.join(sorted(MARITIME_DATASETS))}"
        )
    return rel_to_root("data", f"{dataset}.npz")


def load_st_dataset(dataset):
    """Load spatio-temporal array [time, node, feature]; maritime datasets only."""
    if dataset not in MARITIME_DATASETS:
        raise ValueError(
            f"Unsupported dataset {dataset!r}. This repo only includes bohai and nanhai: "
            f"{', '.join(sorted(MARITIME_DATASETS))}."
        )

    data_path = maritime_npz_path(dataset)
    data = np.load(data_path)["data"]
    print(
        f"[maritime] Loaded {dataset} from {os.path.abspath(data_path)}, shape {data.shape}"
    )

    if data.shape[0] < data.shape[1]:
        print("Detected possible need to transpose data; adjusting...")
        data = data.transpose(1, 0, 2)
        print(f"Shape after transpose: {data.shape}")

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print(
        "Load %s Dataset shaped: "
        % dataset,
        data.shape,
        data.max(),
        data.min(),
        data.mean(),
        np.median(data),
    )
    return data


if __name__ == "__main__":
    print("--- load_dataset self-test ---")
    dataset_name = "nanhai"
    print(f"\nLoading dataset: {dataset_name!r}")
    arr = load_st_dataset(dataset_name)
    print(f"\nOK - shape: {arr.shape}")
    print("\n--- end self-test ---")
