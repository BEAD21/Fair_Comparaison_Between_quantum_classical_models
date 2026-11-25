r"""A6: near-black-pixel shortcut check (the one that excluded Corn), re-run on every crop.

Feature: share of near-black pixels per 32x32 grayscale image (pixel < 0.1 of the
image range). A one-feature threshold classifier is fitted on TRAIN (best threshold
and direction) and scored on VALIDATION. Test is not used.
    .venv\Scripts\python shortcut_check.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('QBENCH_DATA', os.path.join(HERE, 'data', 'PlantVillage_Processed_700 images'))
os.environ.setdefault('QBENCH_WORK', os.path.join(HERE, 'work'))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from qbench import config
from qbench.data import CropData


def frac_black(X, t=0.1):
    X = X.reshape(len(X), -1)
    lo, hi = X.min(), X.max()
    return ((X - lo) / (hi - lo + 1e-12) < t).mean(1)


def stump(f, y):
    best = (0, None, None)
    for th in np.unique(f):
        for sgn in (1, -1):
            acc = ((sgn * (f - th) > 0).astype(int) == y).mean()
            best = max(best, (acc, th, sgn), key=lambda b: b[0])
    return best


rows = []
for crop in config.CROPS + config.EXCLUDED_CROPS:
    d = CropData(crop)
    ftr, fva = frac_black(d.X_train), frac_black(d.X_val)
    acc_tr, th, sgn = stump(ftr, d.y_train)
    acc_va = ((sgn * (fva - th) > 0).astype(int) == d.y_val).mean()
    rows.append({'crop': crop, 'excluded': crop in config.EXCLUDED_CROPS,
                 'train_acc': 100 * acc_tr, 'val_acc': 100 * acc_va,
                 **{f'mean_black_{k}': ftr[d.y_train == v].mean() for k, v in d.label_map.items()}})
df = pd.DataFrame(rows)
df.to_csv(config.RESULTS_DIR / 'revision_a6_shortcut.csv', index=False)
print(df.round(3).to_string(index=False))
