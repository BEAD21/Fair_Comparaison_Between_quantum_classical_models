"""Robustness re-runs for the revision (written 2026-11-19).

Two checks asked for by the review panel, both on the TEST seeds 0-9, 7 crops:

A3  checkpoint rule.  Recipes were chosen on the mean of the last five
    validation epochs, but test numbers used the best validation epoch.
    Every frozen test model is re-trained with the test set evaluated after
    every epoch (train.fit trace_loader; random states restored, so training
    is unchanged).  Each row then has
        test_acc            best-validation-epoch number (must equal the original)
        test_last5          mean test accuracy over the last five epochs
        test_last5_probavg  accuracy of the softmax averaged over the last five epochs
A2  amplitude-encoding control.  Conv2x2 and PatchMLP on the SAME per-patch
    abs + L2 normalisation the circuit receives (c/<filter>/amp).  Their recipes
    are their best of the 12-recipe grid in the validation scan, chosen exactly
    like the raw ones (filterscan.best_recipe); written once to FROZEN.

Nothing here selects a model on test data.  Original prediction files are not
touched: traced outputs go to results/preds_trace/.
"""
import json

import numpy as np
import pandas as pd

from . import config
from .train import DEFAULT_RECIPE, recipe_id, run_one, trace_path

FROZEN = 'robust_frozen.json'
AMP = ['c/Conv2x2/amp', 'c/PatchMLP/amp']
SEEDS = config.SEEDS


def freeze():
    p = config.RESULTS_DIR / FROZEN
    if p.exists():
        return json.loads(p.read_text())
    from . import filterscan as F
    ft = json.loads((config.RESULTS_DIR / 'filter_test_frozen.json').read_text())['models']
    qt = json.loads((config.RESULTS_DIR / 'qnn_test_frozen.json').read_text())['models']
    models = {**ft, **qt}
    for m in AMP:
        br = F.best_recipe(m)
        if br is None:
            raise RuntimeError(f'no scan results for {m}')
        models[m] = br[1]
    fz = {'written': '2026-11-19', 'crops': config.CROPS, 'seeds': SEEDS, 'models': models,
          'amp_rule': 'best of the 12-recipe grid on Cherry+Grape, seeds 100-102 (filterscan.best_recipe), '
                      'same rule as the raw classical filters'}
    p.write_text(json.dumps(fz, indent=2))
    return fz


def _store(model, crop):
    return f"trace_{model.replace('/', '_')}_{crop}"


def _orig_store(model, crop):
    if model.startswith(('qn/', 'cn/')):
        return f"qtest_{model.replace('/', '_')}_{crop}"
    return f"ftest_{model.replace('/', '_')}_{crop}"


def _data(model, crop, cache={}):
    key = ('feat' if model.startswith(('qn/', 'cn/')) else 'img', crop)
    if key not in cache:
        if key[0] == 'feat':
            from .qnn import FeatureData
            cache[key] = FeatureData(crop)
        else:
            from .data import CropData
            cache[key] = CropData(crop)
    return cache[key]


def jobs():
    return [(m, c) for m in freeze()['models'] for c in config.CROPS]


def run_job(model, crop, tag='', seeds=None):
    from .results import ResultStore
    r = freeze()['models'][model]
    rid = recipe_id({**DEFAULT_RECIPE, **r})
    store = ResultStore(_store(model, crop))
    for s in (SEEDS if seeds is None else seeds):
        if store.done(crop, model, rid, s, 'final'):
            continue
        row, hist = run_one(_data(model, crop), model, s, r, 'final', trace_test=True)
        store.add(row, hist)
        print(f"{tag}{model:<15} {crop:<7} seed {s} | best-ep TEST {row['test_acc']:6.2f} "
              f"| last5 {row['test_last5']:6.2f} | {row['seconds']:5.1f}s", flush=True)


def rows():
    fz = freeze()
    out = []
    for m, r in fz['models'].items():
        rid = recipe_id({**DEFAULT_RECIPE, **r})
        for c in config.CROPS:
            p = config.RESULTS_DIR / f'{_store(m, c)}.csv'
            if p.exists():
                d = pd.read_csv(p)
                out.append(d[(d['recipe'] == rid) & (d['phase'] == 'final')])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def check_reproduction(df=None):
    """Traced re-runs must reproduce the original best-epoch runs exactly."""
    df = rows() if df is None else df
    bad, n = [], 0
    for (m, c), g in df.groupby(['model', 'crop']):
        p = config.RESULTS_DIR / f'{_orig_store(m, c)}.csv'
        if not p.exists():
            continue
        o = pd.read_csv(p)
        o = o[(o['recipe'] == g['recipe'].iloc[0]) & (o['phase'] == 'final')].set_index('seed')
        for _, row in g.iterrows():
            if row['seed'] not in o.index:
                continue
            n += 1
            ref = o.loc[row['seed']]
            if ref['best_epoch'] != row['best_epoch'] or abs(ref['test_acc'] - row['test_acc']) > 1e-6:
                bad.append((m, c, row['seed'], ref['best_epoch'], row['best_epoch'],
                            ref['test_acc'], row['test_acc']))
    return n, bad


def load_probs(model, crop, seed, kind='prob_last5'):
    r = freeze()['models'][model]
    z = np.load(trace_path(crop, model, recipe_id({**DEFAULT_RECIPE, **r}), seed))
    return z[kind], z['y']


# ---- A4 follow-up: learning-rate axis extended for both sides (validation only) ----
# 18 of 22 quantum filters chose the lower edge of the 2-value lr axis {0.005, 0.01}.
# The test models of BOTH sides get lr 0.002 and 0.02 with the same act x BN levels,
# Stage 1 protocol (Cherry + Grape, seeds 100-102, val_last5). Rows go to the same
# qscan1 stores; filterscan.best_recipe ignores them (it reads the 12-recipe grid only).
LR_EXT = [0.002, 0.02]
LR_MODELS = ['q/ry_cnot_B6', 'c/Conv2x2/raw', 'c/PatchMLP/raw', 'c/Conv2x2/amp', 'c/PatchMLP/amp']


def lr_ext_recipes():
    from .filterscan import FIXED, GRID
    return [{'act': a, 'norm': n, 'lr': lr, **FIXED} for a in GRID['act'] for n in GRID['norm'] for lr in LR_EXT]


def lr_ext_job(model, tag=''):
    from . import filterscan as F
    from .data import CropData
    from .tune import DEV_CROPS, TUNE_SEEDS
    cache = {}

    def data_of(c):
        if c not in cache:
            cache[c] = CropData(c)
        return cache[c]
    for r in lr_ext_recipes():
        F._run(F._store(model, 1), data_of, model, r, DEV_CROPS, TUNE_SEEDS, tag)


def lr_ext_board():
    """Per model: best recipe on the original 12-recipe grid vs on the 24-recipe grid."""
    from . import filterscan as F
    from .tune import DEV_CROPS, TUNE_SEEDS
    ids12 = F.RECIPE_IDS
    ids24 = ids12 | {recipe_id({**DEFAULT_RECIPE, **r}) for r in lr_ext_recipes()}
    out = []
    for m in LR_MODELS:
        d = pd.read_csv(config.RESULTS_DIR / f'{F._store(m, 1)}.csv')
        d = d[d['crop'].isin(DEV_CROPS) & d['seed'].isin(TUNE_SEEDS)]
        g = d.groupby(['recipe', 'crop'])[F.METRIC].mean().unstack('crop').reindex(columns=DEV_CROPS).dropna()
        s = g.mean(axis=1)
        b12, b24 = s[s.index.isin(ids12)], s[s.index.isin(ids24)]
        out.append({'model': m, 'n_recipes': int(s.index.isin(ids24).sum()),
                    'best12': b12.idxmax(), 'score12': b12.max(),
                    'best24': b24.idxmax(), 'score24': b24.max(),
                    'changed': b12.idxmax() != b24.idxmax()})
    return pd.DataFrame(out)
