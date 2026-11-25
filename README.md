# Equal Tuning, Equal Size: a Fair Comparison of Quantum and Classical Models for Plant Disease Detection

Code and all results for the paper of the same title (submitted to ESANN 2027).

Each quantum model is paired with a classical model that has the same input, the same
head, the same number of trainable parameters and the same tuning budget. We compare them
on seven binary PlantVillage tasks (healthy vs diseased: Apple, Cherry, Grape, Peach,
Pepper, Potato, Tomato).

- **QCNN vs CNN (images).** A two-qubit filter (12 parameters) runs on every 2×2 patch of
  a 32×32 grayscale image. A classical filter replaces it, either Conv2x2 (10 parameters)
  or PatchMLP, an exact 12-parameter twin of the circuit. Both feed the same 258-parameter
  head.
- **QNN vs ANN (features).** An 8-qubit variational classifier is compared with a small
  neural network. Both take 8 principal components of HOG, colour and LBP features.
- **Ensembles.** An image model and a feature model, combined by averaging their
  probabilities.

## Main result

Test accuracy (%), mean over 7 crops and 10 seeds:

| Comparison | Quantum | Classical | Difference (pp) | 95% interval over crops | McNemar p |
|---|---|---|---|---|---|
| QCNN vs Conv2x2 | 86.18 | 86.88 | −0.70 | [−3.15, +1.75] | 0.09 |
| QCNN vs PatchMLP (exact twin) | 86.18 | 87.41 | −1.23 | [−4.64, +2.18] | 0.008 |
| QNN vs ANN | 84.04 | 84.78 | −0.74 | [−2.02, +0.53] | 0.83 |
| QCNN+QNN vs Conv2x2+ANN | 87.97 | 88.60 | −0.62 | [−2.62, +1.37] | 0.70 |
| QCNN+QNN vs PatchMLP+ANN | 87.97 | 88.83 | −0.86 | [−3.09, +1.37] | 0.43 |
| QCNN vs PatchMLP, same encoded input | 86.18 | 86.18 | 0.00 | [−1.90, +1.90] | 0.055 |

The quantum models are slightly behind in every comparison, and no gap is significant
across crops. When the classical filter gets the same amplitude-encoded (normalised)
patches as the circuit, the two tie.

## Repository layout

```
qbench/            the package: data, models (circuits simulated exactly in PyTorch),
                   training, scans, test runs, statistics, HTML reports
run_local.py       one command per experiment (list below)
paper_numbers.py   rebuilds Table 1, Fig. 2 and Fig. 3 from the saved results
revision_numbers.py  equivalence tests, encoding control, last-five-epoch check, SDs
shortcut_check.py  near-black-pixel shortcut check (the reason Corn is excluded)
tests/             smoke test, reference implementations, PennyLane reference script
paper_figures/     output of paper_numbers.py (LaTeX sources of Table 1, Fig. 2, Fig. 3)
work/results/      every training run of the project (CSV), frozen configs (JSON),
                   per-image test predictions (preds/, preds_trace/), data splits
work/reports/      HTML summaries of each experiment
work/pl_check/     inputs and outputs of the PennyLane cross-check
```

## Installation

Python 3.14. Most runs used the CPU; a GPU is optional.

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt          # Windows: .venv\Scripts\pip ...
```

The PennyLane cross-check runs in a separate environment (Python 3.12) named `.venv_pl`:

```
python3.12 -m venv .venv_pl
.venv_pl/bin/pip install -r requirements-pennylane.txt
```

## Reproduce the paper's numbers from the saved results

These scripts train nothing and do not need the images:

```
python paper_numbers.py            # Table 1, Fig. 2, Fig. 3 -> paper_figures/
python revision_numbers.py         # -> work/results/revision_a1_tost.csv, revision_a2_a3.csv, revision_a6_sd.csv
python run_local.py robust_check   # the 490 re-runs with per-epoch test tracing match the original test runs
```

## Re-run the experiments

All commands are resumable: rows that already exist in `work/results/` are skipped. To
run everything from scratch, point `QBENCH_WORK` to an empty folder. The order used in
the paper:

| Step | Command | What it does | Main output |
|---|---|---|---|
| 1 | `python run_local.py filter_scan` | 22 two-qubit filters and 2 untrained controls, 12 recipes each; validation only | `qscan1_*`, `qscan2_*`, `filter_scan.csv` |
| 2 | `python run_local.py classical_scan` | 10 classical filters through the same scan | `classical_scan.csv` |
| 3 | `python run_local.py filter_test` | B6, Conv2x2, PatchMLP and the identity control with frozen recipes; test seeds 0–9 | `filter_test_frozen.json`, `ftest_*`, `preds/` |
| 4 | `python run_local.py qnn_tune` | QNN and ANN, Stage 1 (Cherry and Grape) | `qnn_tune_*` |
| 5 | `python run_local.py qnn_scan` | QNN and ANN, Stage 2 (7 crops, validation only); freezes the winners | `qnnscan2_*`, `qnn_test_frozen.json` |
| 6 | `python run_local.py qnn_test` | QNN, ANN and the encoding-only control on test | `qtest_*` |
| 7 | `python run_local.py ensemble` | probability-averaged ensembles (trains nothing) | `ensemble_rows.csv`, `ensemble_test.csv` |
| 8 | `python run_local.py robust` | re-runs every test model with the test accuracy recorded each epoch, plus the classical filters on encoded input | `robust_frozen.json`, `trace_*`, `preds_trace/` |
| 9 | `python run_local.py lr_ext` | learning rates 0.002 and 0.02 for the image models of both sides (validation only) | `revision_a4_lr_ext.csv` |
| 10 | `python run_local.py pl_check` | every circuit against PennyLane | `pennylane_check.json`, `work/pl_check/` |
| 11 | `python shortcut_check.py` | accuracy of the near-black-pixel rule per crop | `revision_a6_shortcut.csv` |

The preliminary study of filter size and of ancilla and fully quantum variants is in
`run_local.py arch`, `arch100` and `fqc_test`.

Results on another machine may differ slightly from the saved ones. The file list of
each crop is read in directory order before a fixed shuffle, and directory order
depends on the file system. Floating-point results also depend on the hardware. The
splits themselves are fixed by the manifests (next section).

## Data

The images come from PlantVillage (D. P. Hughes and M. Salathé, arXiv:1511.08060,
2015), segmented-leaf version. We use a balanced subset with fixed
train/validation/test splits (978/210/202–212 images per crop). The exact files of every
split are listed in `work/results/split_manifest_<crop>.csv`. Some files in the subset
are augmented copies (`aug__*`) that are not in the original dataset.

The images are not included in this repository. To re-run the experiments, place them as

```
data/PlantVillage_Processed_700 images/<Crop>/<Class>/<train|validation|test>/<file>
```

or set `QBENCH_DATA` to their location.

Corn is excluded. Its healthy images have no black background, so the share of
near-black pixels alone separates its two classes (99.5% on validation). Its splits
and earlier runs stay in `work/results/`, but no result in the paper uses them.

## Result files

Each row of a run CSV is one training run, with the columns `crop, model, recipe, seed,
phase, filter_params, head_params, total_params, val_acc, best_epoch, val_last5,
test_acc, seconds, split`.

- `phase = tune` means validation only (`test_acc` is empty); `phase = final` means a
  test run.
- `split` is a hash of the local file list.
- `*_curves.csv` files hold the per-epoch training loss and validation accuracy.

| Paper | Files |
|---|---|
| Fig. 2 (validation scan) | `filter_scan.csv`, `classical_scan.csv`, `qscan1_*`, `qscan2_*` |
| Table 1 (test accuracy) | `ftest_*`, `qtest_*`, `ensemble_rows.csv`, `trace_c_*_amp_*` (encoded input) |
| Fig. 3 (paired differences, McNemar) | `paper_figures/pairs.csv`, `preds/` |
| Frozen recipes, written before each test run | `filter_test_frozen.json`, `qnn_test_frozen.json`, `robust_frozen.json` |
| Equivalence test, encoding control, last-five-epoch check | `revision_a1_tost.csv`, `revision_a2_a3.csv` |
| SD over seeds and crops, shortcut check, learning-rate extension | `revision_a6_sd.csv`, `revision_a6_shortcut.csv`, `revision_a4_lr_ext.csv` |
| Simulator vs PennyLane | `pennylane_check.json`, `work/pl_check/` |
| Preliminary architecture study | `arch_*`, `arch100_*`, `architecture100_*`, `fqc_test*` |

Files from an earlier version of the protocol (before Corn was excluded) are kept for
transparency. No result in the paper uses them: `tune_*`, `confirm_*`, `final*`,
`frozen_configs.json`, `best_configs*.json`, `qnn_final*`, `frozen_qnn.json`, `router*`,
`endtoend.csv`, `repro*`.

## Licence

Code under the MIT licence (see `LICENSE`).
