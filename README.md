# MoonBoard Grade Predictor

This repository trains and evaluates MoonBoard grade prediction models (classification + ordinal) based on SetTransformer / DeepSet families, using split-based experiments.

This README focuses on the following assets:

- `data/`
- `models/`
- `result/`
- `main.py`
- `split_data.py`
- `train_models.py`
- `run_inference.py`
- `confusion_matrix.py`

## 1. Execution Environment

### Python

- Recommended: Python 3.10+
- Minimum: Python 3.9+

### Required Packages

- torch
- numpy
- pandas
- scikit-learn
- openpyxl
- matplotlib
- seaborn

### Optional Packages (for some ensemble types)

- xgboost
- lightgbm

Install example:

```bash
pip install torch numpy pandas scikit-learn openpyxl matplotlib seaborn xgboost lightgbm
```

### Device

All scripts automatically select device in this order:

1. CUDA
2. MPS
3. CPU

---

## 2. Project Structure and Meaning

### `data/`

Input datasets and metadata.

Typical files:

- `data/cleaned_moonboard2024_grouped.json`
- `data/cleaned_moonboard2024_filtered_*.json`
- `data/hold_difficulty.txt`

Used by `main.py` dataset loaders and downstream scripts.

### `models/`

Model definitions and ensemble modules.

Typical files:

- `models/classifier.py` (classification architectures)
- `models/ordinal.py` (ordinal architectures)
- `models/ensemble.py` (ensemble implementations: voting, stacking, tree-based, adaboost style)
- `models/modules.py`, `models/modules_modified.py` (SetTransformer blocks etc.)
- `models/utils_ordinal.py` (ordinal loss / utilities)

### `result/`

All generated artifacts.

Main outputs include:

- split index definitions (`split_indices.json` or `splits/split_indices.json` depending on command)
- trained model checkpoints under `result/split_models/`
- inference reports (`.csv`, `.xlsx`)
- confusion matrix images


---

## 3. Key Scripts

### `main.py`

Central module that provides:

- dataset loading
- mappings (`hold_to_idx`, `grade_to_label`, etc.)
- collate functions (`make_collate_fn`)
- model training/evaluation helpers
- model/ensemble classes imported from `models/`

Other scripts (`split_data.py`, `train_models.py`, `run_inference.py`) use this as the backend.

### `split_data.py`

Creates reproducible train/test index splits and saves them to JSON.

- default split count: 25
- stratified by grade labels
- default output: `result/split_indices.json`

### `train_models.py`

Trains base models and ensembles for each saved split.

- loads split records from `--splits-path`
- saves checkpoints to `--save-root` (default `result/split_models`)
- supports parallel base-model training (`--parallel-workers`)
- supports checkpoint reuse (skip if checkpoint exists)
- supports single split execution (`--split-id`)
- can skip ensembles (`--skip-ensembles`)

### `run_inference.py`

Evaluates all checkpoints in each split directory against each split test set.

Outputs:

- per-split per-model metrics (iterations)
- summary statistics (mean/std)
- class-wise iterations and summaries (classification models only)
- ordinal threshold-wise iterations and summaries (ordinal models only)

Important behavior:

- primary metric is model-family dependent:
  - classification: strict accuracy
  - ordinal: cumulative threshold accuracy (mean over thresholds)

Main arguments:

- `--splits-path` (default: `result/splits/split_indices.json`)
- `--model-root` (default: `result/split_models`)
- `--batch-size` (default: `16`)
- `--eval-target` (choices: `all`, `classification`, `ordinal`; default: `all`)
- `--decision-threshold` (default: `0.5`)
- `--output-csv` (default: `result/inference_split_iterations.csv`)
- `--output-excel` (default: `result/inference_split_summary.xlsx`)

### `confusion_matrix.py`

Generates row-normalized confusion matrix images per model and optionally inserts them into an Excel workbook.

Current default standalone target directories:

- `result/classification`
- `result/ordinal`

If your actual checkpoints are split-based (`result/split_models/...`), either:

- adapt `checkpoint_dirs` argument when calling the function, or
- run this script after collecting desired checkpoints into flat folders.

---

## 4. Standard Workflow (Recommended)

### Step 1: Generate split indices

```bash
python split_data.py --num-splits 25 --test-size 0.2 --base-seed 42 --output-path result/split_indices.json
```

Output:

- `result/split_indices.json`

### Step 2: Train models for all splits

```bash
python train_models.py --splits-path result/split_indices.json --save-root result/split_models --parallel-workers 4
```

Single split example:

```bash
python train_models.py --splits-path result/split_indices.json --save-root result/split_models --split-id 5 --parallel-workers 4
```

Skip ensemble example:

```bash
python train_models.py --splits-path result/split_indices.json --save-root result/split_models --skip-ensembles
```

Main output tree:

- `result/split_models/split_001/classification/*.pt`
- `result/split_models/split_001/ordinal/*.pt`
- ...

### Step 3: Run split-based inference summary

```bash
python run_inference.py --splits-path result/split_indices.json --model-root result/split_models --batch-size 16 --eval-target all --decision-threshold 0.5 --output-csv result/inference_split_iterations.csv --output-excel result/inference_split_summary.xlsx
```

Ordinal-only example:

```bash
python run_inference.py --splits-path result/split_indices.json --model-root result/split_models --eval-target ordinal --decision-threshold 0.5
```

Outputs:

- `result/inference_split_iterations.csv`
- `result/inference_split_summary.xlsx`
- `result/inference_split_iterations_classwise.csv`
- `result/inference_split_iterations_classwise_summary.csv`
- `result/inference_split_iterations_ordinal_thresholds.csv`
- `result/inference_split_iterations_ordinal_thresholds_summary.csv`

### Step 4 (optional): Confusion matrices

```bash
python confusion_matrix.py
```

Typical outputs:

- `result/confusion_mean_<model>.png`
- inserted sheets in `result/model_comparison_results.xlsx` (if that workbook exists)

---

## 5. Output File Reference

### Split Definition

- `result/split_indices.json`
  - split metadata and train/test indices

### Trained Checkpoints

- `result/split_models/split_<ID>/classification/<model>.pt`
- `result/split_models/split_<ID>/ordinal/<model>.pt`

Checkpoint payload keys:

- `model_name`
- `is_ordinal`
- `class_name`
- `model_object`

### Inference Reports

- `result/inference_split_iterations.csv`
  - one row per split-model
  - includes `Primary Metric` and `Primary Accuracy (%)`
  - classification primary metric: strict accuracy
  - ordinal primary metric: cumulative threshold accuracy
  - also includes strict / ±1 metrics for reference
- `result/inference_split_summary.xlsx`
  - `iterations`
  - `summary`
  - `classwise_iterations` (classification only)
  - `classwise_summary` (classification only)
  - `ordinal_threshold_iterations` (ordinal only)
  - `ordinal_threshold_summary` (ordinal only)

Additional CSV reports:

- `result/inference_split_iterations_classwise.csv` (classification only)
- `result/inference_split_iterations_classwise_summary.csv` (classification only)
- `result/inference_split_iterations_ordinal_thresholds.csv` (ordinal only)
- `result/inference_split_iterations_ordinal_thresholds_summary.csv` (ordinal only)

### Confusion Matrices

- `result/confusion_mean_<model>.png`
- optional Excel insertion to `result/model_comparison_results.xlsx`

---

## 6. Notes

- Use `run_inference.py` as the primary evaluation script for split-based training artifacts.
- `confusion_matrix.py` default checkpoint directories are flat (`result/classification`, `result/ordinal`); adjust when using split-based outputs.
