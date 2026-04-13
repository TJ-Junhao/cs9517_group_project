# COMP9517 Group Project — Plant / Soil Segmentation

Semantic segmentation of plants from top-down agricultural imagery on the EWS
dataset. The repo bundles three families of methods under a shared
`ImagePipeline` abstraction and a uniform CLI:

- **Classical CV** — k-means, HSV thresholding, Canny edges, Excessive Green,
  watershed, GrabCut, and a DenseCRF post-processor.
- **Machine Learning** — pixel-wise Logistic Regression and Random Forest on
  hand-crafted features.
- **Deep Learning** — U-Net, ResU-Net, ASPP-ResU-Net, and Attention-Gate
  ASPP-ResU-Net.

All methods share the same evaluation pipeline (normal + robustness under
Gaussian noise / blur / brightness shift / rotation / JPEG compression),
emit identical JSON performance reports, and can be plotted against each
other with the comparison tool.

**Maintainer:** Junhao Bai

## Repo Layout

```text
cs9517_group_project/
├── configs/                 # JSON configs consumed by the CLIs
│   ├── traditional_cv.json
│   ├── networks.json
│   ├── network_exg.json
│   └── resunet_parameter.json
├── datasets/EWS-Dataset/    # train / val / test splits (not committed)
├── scripts/                 # CLI entry points and thin shell wrappers
│   ├── classic_cv.py        # classical CV methods
│   ├── train.sh / eval.sh   # wrappers around project.deep_learning.*
│   └── compare.sh           # wrapper around project.visualization.compare
├── src/project/
│   ├── config/              # argparse + config merging
│   ├── data/                # image IO, JSON IO
│   ├── processing/          # ImagePipeline + classical CV ops
│   ├── machine_learning/    # LR / RF train + evaluate
│   ├── deep_learning/       # U-Net family train + evaluate
│   ├── models/              # network and ML model definitions
│   ├── evaluation/          # metrics (confusion, IoU, report)
│   ├── visualization/       # confusion plots, cross-method comparison
│   └── utils/               # paths, registry, seeding, file helpers
└── runs/                    # per-run outputs (checkpoints, metrics, plots)
```

---

## Prerequisites

- **Python 3.12**
- **`uv`** (recommended) — <https://github.com/astral-sh/uv>
- macOS / Linux. Tested on macOS 25.x.
- ~4 GB disk (dataset + checkpoints). GPU optional; all training falls back
  to CPU but a CUDA / MPS device is strongly recommended for the U-Net
  variants.

---

## Installation

### 1. Clone and enter the repo

```bash
git clone <your-fork-url> cs9517_group_project
cd cs9517_group_project
```

### 2. Create the virtual environment and install dependencies

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

The `-e .` installs the `project` package in editable mode, so
`python -m project.deep_learning.train` and friends resolve to `src/project`.

### 3. Install `pydensecrf` (for `crf_method` only)

The upstream `pydensecrf` on PyPI fails to build against Cython ≥ 3.
Install from the git master instead:

```bash
uv pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

If the build still fails, drop `cython` to the 0.29 line before installing:

```bash
uv pip install "cython<3"
uv pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

You can skip this step if you do not plan to use `crf_method`; all other
methods work without `pydensecrf`.

### 4. Place the dataset

```text
datasets/EWS-Dataset/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Ground-truth masks follow the convention **plant = 0 (black), soil = 255
(white)**. This is enforced by the data loader and by every method's output
convention.

---

## CLI Usage

All CLIs accept `-R <run_name>` to tag outputs and `-C <config.json>` to load
hyper-parameters from `configs/`. Outputs always land in
`runs/<run_name>/`.

### Classical CV

```bash
./scripts/classic_cv.py -R <run_name> -C traditional_cv.json -m <mode>
```

- `-R` — run name (also the key into the config file)
- `-C` — config file in `configs/` (method + kwargs)
- `-m` — one of `train`, `validation`, `test`

Registered methods (see `src/project/utils/registry.py`):

| Name                     | Description                                                      |
|--------------------------|------------------------------------------------------------------|
| `kmeans_method`          | 2-class k-means in RGB with ExG plant-class selection            |
| `edge_method`            | Gaussian blur → Canny → closing → small-object removal           |
| `excessive_green_method` | Threshold on the Excessive Green (2G − R − B) index              |
| `hsv_segmentation_method`| HSV range mask (defaults to green in `LOWER/UPPER_GREEN`)        |
| `watershed_method`       | ExG-seeded watershed                                             |
| `grabcut_method`         | ExG-initialised GrabCut (slow; see below)                        |
| `crf_method`             | DenseCRF post-processing of an ExG unary (requires `pydensecrf`) |

Example:

```bash
./scripts/classic_cv.py -R watershed_method -C traditional_cv.json -m test
./scripts/classic_cv.py -R crf_method       -C traditional_cv.json -m test
```

Each run produces normal evaluation outputs plus a robustness sweep over
five corruption families. Expect `grabcut_method` to take >30 min on the
full robustness matrix; `iters` is already lowered to 2 in the registered
method.

### Deep Learning — Training

Direct invocation:

```bash
python -m project.deep_learning.train_neural_network -R <run_name> -C <config.json>
```

Convenience wrapper:

```bash
./scripts/train.sh <run_name> <config.json> [resume_checkpoint]
```

Common flags (also valid as config keys):

| Flag                   | Default | Purpose                                                             |
|------------------------|---------|---------------------------------------------------------------------|
| `-M / --model`         | `unet`  | `unet` / `resunet` / `aspp_resunet` / `attention_gate_aspp_resunet` |
| `-lr / --learning_rate`| `1e-4`  | Adam learning rate                                                  |
| `-ep / --epoch`        | `40`    | Max epochs                                                          |
| `-p / --patience`      | `5`     | Early-stopping patience on val loss                                 |
| `-md / --min_delta`    | `1e-4`  | Min val-loss improvement to reset patience                          |
| `-c / --criteria`      | `0.6`   | Sigmoid threshold used for metrics                                  |
| `-b / --batch_size`    | `3`     | —                                                                   |
| `--features`           | `[]`    | Extra channels: any of `exg`, `hsv`                                 |
| `-r / --resume`        | —       | Path to a checkpoint                                                |

Example:

```bash
./scripts/train.sh UNet_Baseline networks.json
./scripts/train.sh ResUnet_exg   network_exg.json
```

Checkpoints go to `runs/<run_name>/checkpoints/model.pt`, training logs to
`runs/<run_name>/logs/`.

### Deep Learning — Evaluation

```bash
python -m project.deep_learning.evaluate -R <run_name> -C <config.json> -m <mode>
```

Convenience wrapper (runs test / train / validation in sequence):

```bash
./scripts/eval.sh <run_name> <config.json>
```

Produces, under `runs/<run_name>/`:

- `performance/performance_<mode>.json` — full sklearn report + IoU + inference time
- `plots/` — confusion matrices
- `outputs/<mode>/` — predicted masks
- `failures/<mode>/` — 10 worst mIoU images per corruption level
- `performance/robustness/<corruption>/level_<i>/` — per-level reports
- `plots/robustness/<corruption>/level_<i>/` — per-level confusion matrices

### Machine Learning — Random Forest / Logistic Regression

Random Forest:

```bash
python -m project.machine_learning.train_rf --run <run_name> \
    [--n-estimators 100] [--max-depth 15] [--samples-per-class 2000] \
    [--feature-mode rgb_hsv_exg]

python -m project.machine_learning.evaluate_rf --run <run_name> -m <mode>
```

Logistic Regression:

```bash
python -m project.machine_learning.train_lr --run <run_name>
python -m project.machine_learning.evaluate_lr --run <run_name> -m <mode>
```

Both use the same `runs/<run_name>/` layout and emit the same JSON report
shape as the deep-learning pipeline, so they drop straight into the
comparison tool.

### Cross-method Comparison

Renders side-by-side metric tables and robustness curves from the JSON
reports across runs:

```bash
./scripts/compare.sh <mode> <datatype> [run_name]
# mode     ∈ { cross_model, robustness_level }
# datatype ∈ { train, validation, test }
```

Equivalent direct invocation:

```bash
python -m project.visualization.compare -m cross_model -D test
python -m project.visualization.compare -m robustness_level -D test -R UNet_Baseline
```

Outputs go to `comparisons/`.

---

## Config Files

Every CLI that takes `-C <file>.json` expects a top-level object keyed by
run name. Example (`configs/traditional_cv.json`):

```json
{
  "watershed_method": {
    "method": "watershed_method",
    "kwargs": { "exg_low": -20, "exg_high": 20 }
  },
  "crf_method": {
    "method": "crf_method",
    "kwargs": {
      "exg_threshold": 10,
      "gt_prob": 0.7,
      "iters": 5,
      "sxy_bilateral": 60,
      "srgb_bilateral": 13,
      "compat_bilateral": 10,
      "remove_object_size": 100
    }
  }
}
```

`-R <key>` selects the entry. CLI flags that match a config key override
the config value.

For deep learning (`configs/networks.json`), the kwargs are flattened onto
the top level of the entry:

```json
{
  "UNet_Baseline": {
    "model": "unet",
    "learning_rate": 1e-4,
    "epoch": 100,
    "patience": 10,
    "criteria": 0.6,
    "features": [],
    "batch_size": 3
  }
}
```

---

## Output Conventions

- **Pixel convention:** `plant = 0`, `soil = 255` in every saved mask (both
  ground truth and prediction) so they can be diffed visually.
- **Class-label convention for metrics:** `label 0 = soil`, `label 1 = plant`
  (`class_names = ["soil", "plant"]` in `evaluation/metrics.py`). The
  `ImagePipeline.flatten` helper returns arrays in this convention
  regardless of method, so reports are directly comparable across classical
  CV, ML, and DL.
- **Reproducibility:** every entry point calls `set_seed(SEED)` (numpy /
  random / torch / torch.cuda), and robustness evaluation reseeds before
  the sweep so stochastic corruptions (`gaussian_noise`, warps) are
  deterministic.

---

## Extending

Adding a new classical method:

1. Add the op to `src/project/processing/pipeline.py` (as a method on
   `ImagePipeline` returning a new pipeline).
2. Compose it into a method function in
   `src/project/processing/classic_cv.py` — end with `.invert()` so the
   output follows the `plant = 0` pixel convention.
3. Register it under `TRADITIONAL_CV_METHODS` in
   `src/project/utils/registry.py` and add a config entry in
   `configs/traditional_cv.json`.

Adding a new network:

1. Define the module in `src/project/models/cnn.py`.
2. Register it under `MODELS` in `src/project/utils/registry.py`.
3. Reference it by name in a `configs/*.json` entry.
