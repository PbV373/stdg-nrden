# Dependencies and computational requirements

This document complements `README.md` and `requirements.txt` with explicit software stack notes and rough hardware guidance for **STDG-NRDEN** (*Spatio-Temporal Dynamic Graph Neural Rough Differential Equation Network for Chlorophyll-a Concentration Prediction*) training and evaluation.

## Python packages

Core pins are listed in `requirements.txt`:

| Package | Role |
|---------|------|
| PyTorch (`torch`, `torchvision`, `torchaudio`) | Model, training, CUDA |
| NumPy, SciPy | Arrays, splines / numerical helpers |
| pandas | CSV ingestion in data scripts |
| scikit-learn | Metrics / utilities used by the codebase |
| matplotlib | Optional plotting when enabled in config |
| tqdm | Progress display |
| torchdiffeq | ODE / CDE solvers (used by `controldiffeq`) |

**Log-signature backend (required):** install **exactly one** of:

- `iisignature` — recommended on Windows; used with prefix cumulants when `stream=True` is unavailable.
- `signatory` — common on Linux/macOS; must be compatible with your PyTorch build.

Optional:

- `tensorboard` — only if you pass `--tensorboard` to `model/Run_cde.py`. Old `tensorboard` stacks may conflict with NumPy 2.x; see README troubleshooting.

## System requirements

| Resource | Minimum (smoke test) | Typical paper-scale run |
|----------|----------------------|-------------------------|
| GPU | Not strictly required (CPU runs are very slow) | NVIDIA GPU with ≥ 8 GB VRAM (e.g. RTX 3060 / 4060 class) |
| RAM | ≥ 8 GB | ≥ 16 GB |
| Disk | A few GB for code, data, logs | More if you keep many `runs/` checkpoints |

Exact throughput depends on `batch_size`, `num_nodes`, `lag`, `horizon`, `embed_dim`, `hid_dim`, and the log-signature backend. Larger graphs (e.g. 300 nodes) increase memory and per-step cost.

## Reproducibility vs speed

- Default training enables `cudnn.benchmark` for speed when input shapes are fixed.
- Pass `--deterministic` for more reproducible (but slower) runs; see `lib/TrainInits.py`.

## Distribution policy (archives)

For security review environments: this repository is intended to be used **without** distributing a single opaque archive (e.g. `.zip` / `.rar` / `.7z`) as the only artifact. Prefer a git checkout or unpacked directories plus documented individual files (`requirements.txt`, data files, etc.).
