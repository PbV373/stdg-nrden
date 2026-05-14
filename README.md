# STDG-NRDEN

**Paper:** *Spatio-Temporal Dynamic Graph Neural Rough Differential Equation Network for Chlorophyll-a Concentration Prediction.*

**STDG-NRDEN** is the official short name for this method. The implementation combines neural controlled / rough differential equations with a graph-structured spatial module to predict **chlorophyll-a concentration** in maritime grids.

## License

This project is released under the [MIT License](LICENSE). Third-party code (e.g. under `controldiffeq/`) retains its original terms where noted in source files.

## Security and distribution

For environments that **reject single opaque archives** (e.g. `.zip`, `.rar`, `.7z`) as the only deliverable: use a **git clone** or an **unpacked directory tree** with individual files (`requirements.txt`, CSV/NPZ data, source). Do not rely on one compressed bundle as the sole redistribution format.

## What is included

| Path | Purpose |
|------|---------|
| `model/` | Training entry `Run_cde.py`, configs `*_NRDEN.conf`, model and trainer code |
| `lib/` | Data loading, sliding windows, metrics, normalization |
| `data/` | Maritime CSV/NPZ assets in a **flat** `data/` root (`data/nanhai.npz`, `data/bohai.npz`, …) |
| `data/scripts/` | `deal_dataset.py` (CSV → NPZ), helpers `verify.py`, `change.py`, etc. |
| `controldiffeq/` | Neural CDE integration utilities |

Dataset identifiers are **`nanhai`** (South China Sea grid) and **`bohai`** (Bohai Sea grid). Configs are named `model/{dataset}_NRDEN.conf`. Logs go under `runs/{dataset}/`. Optional test weights: `pre-trained/{dataset}.pth`.

**Repository:** [github.com/PbV373/stdg-nrden](https://github.com/PbV373/stdg-nrden) (short name `stdg-nrden`; method acronym **STDG-NRDEN**).

## Installation

From the repository root:

```bash
pip install -r requirements.txt
```

Install a **CUDA-enabled PyTorch** build that matches your system from [pytorch.org](https://pytorch.org).

`requirements.txt` installs `iisignature` as the default log-signature backend. Advanced users may replace it with a PyTorch-compatible `signatory` build if preferred.

See **[DEPENDENCIES_AND_COMPUTE.md](DEPENDENCIES_AND_COMPUTE.md)** for package roles, optional TensorBoard, and hardware notes.

## Data layout and preparation

Training expects **`data/{dataset}.npz`** with a single array **`data`** of shape **`[time_steps, num_nodes, num_features]`** (typically one channel per node). `lib/load_dataset.py` will transpose if it detects `[nodes, time, ...]`.

### Build NPZ from CSV (recommended path)

1. Place a numeric grid CSV under `data/` (rows = time, columns = nodes), or pass `--csv` with an absolute or repo-relative path.
2. From the **repository root**:

```bash
python data/scripts/deal_dataset.py --dataset nanhai
python data/scripts/deal_dataset.py --dataset bohai --csv data/bohai_300.csv
```

This writes `data/nanhai.npz` or `data/bohai.npz`. Defaults point at the sample CSVs shipped in this repo (`nanhai_265.csv`, `bohai_300.csv`).

### Sanity-check NPZ

```bash
python data/scripts/verify.py --dataset nanhai
```

## Tutorial: typical workflows

### 1) Train

From the **repository root** (required for imports and relative paths):

```bash
python model/Run_cde.py --dataset nanhai --mode train
python model/Run_cde.py --dataset bohai --mode train
```

Hyperparameters are read from `model/{dataset}_NRDEN.conf`. Ensure `num_nodes` matches the second dimension of `data['data']` after any transpose.

### 2) Evaluate with a saved checkpoint

Place weights at `pre-trained/{dataset}.pth`, then:

```bash
python model/Run_cde.py --dataset nanhai --mode test
```

### 3) Optional TensorBoard

```bash
python model/Run_cde.py --dataset nanhai --mode train --tensorboard
```

If import fails (often due to old TensorFlow/TensorBoard vs NumPy 2.x), training still runs; see `DEPENDENCIES_AND_COMPUTE.md`.

## User guide: main CLI options and behaviour

| Option | Role |
|--------|------|
| `--dataset` | `nanhai` or `bohai`; selects config `model/{dataset}_{model}.conf`, NPZ path, log subfolder |
| `--model` | Config stem suffix (default `NRDEN`) -> reads `model/{dataset}_{model}.conf` |
| `--mode` | `train` or `test` |
| `--device` | GPU index (CUDA) |
| `--deterministic` | More reproducible, slower (disables `cudnn.benchmark`) |
| `--tensorboard` | Enable `SummaryWriter` when dependencies allow |

Config file sections (`[data]`, `[model]`, `[train]`, `[test]`, `[log]`) map to `Run_cde.py` arguments. Important data keys:

- `lag` — input window length  
- `horizon` — prediction horizon  
- `num_nodes` — must match data  
- `normalizer`, `val_ratio`, `test_ratio` — preprocessing and splits  

**Training outputs:** a timestamped directory under `runs/{dataset}/` with logs; metrics are printed via the trainer logger.

**Test behaviour:** loads `pre-trained/{dataset}.pth` and runs the test loop; ensure the checkpoint matches the model configuration.

## Reproducibility and limitations

- **Included data:** sample maritime grids are provided so you can run the pipeline end-to-end. They may be subsampled or processed relative to the exact tables used in the camera-ready paper.
- **Exact paper numbers:** bit-exact reproduction may require the **same random seeds**, **deterministic mode**, **solver settings**, **log-signature backend**, and **full proprietary grids** if those were not redistributed.
- **Synthetic fallback:** if you cannot share real observations, you can still validate the implementation by generating synthetic spatio-temporal tensors of shape `[T, N, 1]`, saving them with `deal_dataset.py`-compatible NPZ layout, and tuning `num_nodes` in the config accordingly. Document any domain shift relative to the paper.

## Citation

If you use this code, please cite:

> Spatio-Temporal Dynamic Graph Neural Rough Differential Equation Network for Chlorophyll-a Concentration Prediction.

After publication, add the official BibTeX entry below.
