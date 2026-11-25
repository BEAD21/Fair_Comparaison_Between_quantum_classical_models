"""The whole pipeline: ResNet-50 router -> per-crop binary classifier.

An image counts as correct only if the router picks its crop AND that crop's
model gets the disease right. A misrouted image is always wrong: the other
crop's model can only answer with that crop's labels. So the score needs just
two per-image files, and no model ever runs on another crop's images:
    results/preds/ALL8/...         router predictions on the pooled test set
    results/preds/<crop>/...       stage-2 predictions on that crop's test set

Router seed s is paired with stage-2 seed s, so each model gets 10 end-to-end
scores. All models share the same router, so differences between them come
from stage 2 only. Nothing here trains or selects: it only reads predictions
of the final runs.

QCNN plan: the frozen configs (frozen_configs.json), and each family's controls
with that family's recipe. QNN / ANN come from qnn_final (frozen_qnn.json).
"""
import json

import numpy as np
import pandas as pd

from . import config
from .router import RECIPE as ROUTER_RECIPE
from .train import DEFAULT_RECIPE, predictions_path, recipe_id

CONTROLS = {'quantum': ['q/identity', 'q/random'],
            'classical_raw': ['c/Pool2/raw'],
            'classical_amp': ['c/Pool2/amp']}
ORDER = ['quantum', 'classical_raw', 'classical_amp', 'qnn', 'qnn_matched', 'ann']


def qcnn_plan():
    """[(model, recipe)] for the final QCNN / CNN run."""
    fz = json.loads((config.RESULTS_DIR / 'frozen_configs.json').read_text())['families']
    plan = []
    for fam, ctrl in CONTROLS.items():
        r = fz[fam]['recipe']
        plan.append((fz[fam]['model'], r))
        plan += [(c, r) for c in ctrl]
    return plan


def stage2_models():
    """{family: (model, recipe, store)}: the frozen models that have a final store."""
    out = {}
    fz = json.loads((config.RESULTS_DIR / 'frozen_configs.json').read_text())['families']
    for f in ('quantum', 'classical_raw', 'classical_amp'):
        out[f] = (fz[f]['model'], fz[f]['recipe'], 'final')
    p = config.RESULTS_DIR / 'frozen_qnn.json'
    if p.exists():
        q = json.loads(p.read_text())
        for f in ('qnn', 'ann'):
            out[f] = (q[f]['model'], q[f]['recipe'], 'qnn_final')
    p = config.RESULTS_DIR / 'frozen_qnn_matched.json'
    if p.exists():
        q = json.loads(p.read_text())
        out['qnn_matched'] = (q['model'], q['recipe'], 'qnn_final')
    return out


def _load(crop, model, recipe, seed):
    rid = recipe_id({**DEFAULT_RECIPE, **recipe})
    path = predictions_path(crop, model, rid, seed)
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    return z['pred'], z['y'], str(z['split'])


def score(family, model, recipe, store, seeds=config.SEEDS):
    """One row per crop and seed + crop='ALL'. test_acc = end-to-end accuracy,
    oracle_acc = the same model with a perfect router."""
    df = pd.read_csv(config.RESULTS_DIR / f'{store}.csv')
    rid = recipe_id({**DEFAULT_RECIPE, **recipe})
    df = df[(df['model'] == model) & (df['recipe'] == rid)]
    rows = []
    for s in seeds:
        router = _load('ALL8', 'rn/head', ROUTER_RECIPE, s)
        if router is None:
            raise FileNotFoundError(f'router predictions missing for seed {s}: run `router`')
        rp, ry, _ = router
        start, ok_all, oracle_all, per = 0, [], [], []
        for i, crop in enumerate(config.CROPS):
            st2 = _load(crop, model, recipe, s)
            if st2 is None:
                raise FileNotFoundError(f'{model} {crop} seed {s}: no predictions, run its final')
            pred, y, _ = st2
            end = start + len(y)
            if end > len(rp) or not (ry[start:end] == i).all():
                raise ValueError(f'router and {crop} test images do not line up')
            row = df[(df['crop'] == crop) & (df['seed'] == s)]
            if len(row) != 1:
                raise ValueError(f'{model} {crop} seed {s}: {len(row)} rows in {store}.csv')
            row = row.iloc[0]
            right = pred == y
            if abs(100 * right.mean() - row['test_acc']) > 1e-6:
                raise ValueError(f'{model} {crop} seed {s}: predictions differ from {store}.csv')
            ok = right & (rp[start:end] == i)
            ok_all.append(ok)
            oracle_all.append(right)
            per.append(row)
            rows.append({'crop': crop, 'model': model, 'recipe': rid, 'seed': s, 'family': family,
                         'phase': 'final', 'filter_params': row['filter_params'],
                         'total_params': row['total_params'], 'val_acc': row['val_acc'],
                         'best_epoch': row['best_epoch'], 'test_acc': 100 * ok.mean(),
                         'oracle_acc': row['test_acc'],
                         'routed': 100 * (rp[start:end] == i).mean()})
            start = end
        if start != len(rp):
            raise ValueError(f'router has {len(rp)} test images, stage 2 has {start}')
        ok, right = np.concatenate(ok_all), np.concatenate(oracle_all)
        rows.append({'crop': 'ALL', 'model': model, 'recipe': rid, 'seed': s, 'family': family,
                     'phase': 'final', 'filter_params': per[0]['filter_params'],
                     'total_params': per[0]['total_params'],
                     'val_acc': float(np.mean([r['val_acc'] for r in per])),
                     'best_epoch': int(np.mean([r['best_epoch'] for r in per])),
                     'test_acc': 100 * ok.mean(), 'oracle_acc': 100 * right.mean(),
                     'routed': 100 * (rp == ry).mean()})
    return pd.DataFrame(rows)


def build_table():
    """Writes results/endtoend.csv; returns (rows, families that were skipped)."""
    parts, skipped = [], {}
    for fam, (m, r, store) in stage2_models().items():
        try:
            parts.append(score(fam, m, r, store))
        except (FileNotFoundError, ValueError) as e:
            skipped[fam] = str(e)
    if not parts:
        return pd.DataFrame(), skipped
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(config.RESULTS_DIR / 'endtoend.csv', index=False)
    return df, skipped
