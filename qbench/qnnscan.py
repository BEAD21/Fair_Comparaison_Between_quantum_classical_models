"""QNN vs ANN under the filter protocol (written 2026-11-13, before any run of it).

Replaces the old qnn_tune selection, which used Corn (removed 2026-11-09).

Stage 1  the existing qnn_tune grid: every model x lr {0.001, 0.003, 0.01, 0.03}
         x epochs {25, 50, 100} = 12 configs, identical for both families, on
         the dev crops Cherry + Grape, seeds 100-102. No new training: Corn rows
         are just ignored. (The image models' act x BN grid does not apply here:
         the feature models have no BN and the ANN activation is part of the
         model name, dense_tanh / dense_relu / mlp5_tanh / mlp5_relu.)
Stage 2  each model's best Stage 1 config, 7 crops x seeds 100-109, VALIDATION
         only (phase='tune').
Rule     fixed now: per family, the model with the highest Stage 2 val_last5
         (mean over 7 crops x 10 seeds) is frozen to FROZEN with its config.
Test     QNN winner, ANN winner, and qn/identity with the QNN winner's config
         (control: encoding only, nothing to train in the circuit), seeds 0-9,
         phase='final', once. Predictions + softmax probabilities saved.
         Stats as the filter test (filtertest.report).
"""
import json

import numpy as np
import pandas as pd

from . import config
from . import qnnstudy as Q
from .qnn import QNN_FAMILIES
from .train import DEFAULT_RECIPE, recipe_id, run_one
from .tune import CONFIRM_SEEDS

MODELS = [m for ms in QNN_FAMILIES.values() for m in ms]
FROZEN = 'qnn_test_frozen.json'
SEEDS = config.SEEDS
METRIC = 'val_last5'


def best_recipe(model):
    """Best Stage 1 config of one model on the dev crops (complete grids only)."""
    fam = next(f for f, ms in QNN_FAMILIES.items() if model in ms)
    b = Q.board(fam)
    b = b[b['model'] == model]
    if not len(b):
        raise RuntimeError(f'no complete Stage 1 grid for {model}')
    return json.loads(b.iloc[0]['recipe'])


def _store2(model):
    return f"qnnscan2_{model.replace('/', '_')}"


def run_stage2(models, tag=''):
    from .results import ResultStore
    for m in models:
        r = best_recipe(m)
        rid = recipe_id({**DEFAULT_RECIPE, **r})
        store = ResultStore(_store2(m))
        for crop in config.CROPS:
            for s in CONFIRM_SEEDS:
                if store.done(crop, m, rid, s, 'tune'):
                    continue
                row, hist = run_one(Q._data(crop), m, s, r, 'tune')
                store.add(row, hist)
                print(f"{tag}{m:<14} {crop:<7} seed {s} | last5 {row[METRIC]:6.2f} | {row['seconds']:5.1f}s",
                      flush=True)


def _stage2_grid(m):
    p = config.RESULTS_DIR / f'{_store2(m)}.csv'
    if not p.exists():
        return None
    rid = recipe_id({**DEFAULT_RECIPE, **best_recipe(m)})
    d = pd.read_csv(p)
    d = d[(d['recipe'] == rid) & (d['phase'] == 'tune') & d['crop'].isin(config.CROPS)
          & d['seed'].isin(CONFIRM_SEEDS)]
    return d.pivot_table(index='seed', columns='crop', values=METRIC).reindex(
        index=CONFIRM_SEEDS, columns=config.CROPS), d


def table():
    from .tune import DEV_CROPS
    crops = config.CROPS
    held = [c for c in crops if c not in DEV_CROPS]
    rows = []
    for fam, ms in QNN_FAMILIES.items():
        for m in ms:
            got = _stage2_grid(m)
            if got is None or not len(got[1]):
                continue
            g, d = got
            rows.append({'family': fam, 'model': m, 'params': int(d['total_params'].iloc[0]),
                         'block': int(d['filter_params'].iloc[0]), **g.mean().round(2).to_dict(),
                         'dev': round(np.nanmean(g[DEV_CROPS].to_numpy()), 2),
                         'held_out': round(np.nanmean(g[held].to_numpy()), 2),
                         'all': round(np.nanmean(g.to_numpy()), 2),
                         'seed_sd': round(g.mean(axis=1).std(ddof=1), 2),
                         'runs': int(g.notna().to_numpy().sum()),
                         'complete': int(g.notna().to_numpy().sum()) == len(crops) * len(CONFIRM_SEEDS),
                         'recipe': json.dumps(best_recipe(m))})
    t = pd.DataFrame(rows)
    return t.sort_values(['family', 'all'], ascending=[False, False]).reset_index(drop=True) if len(t) else t


def freeze():
    """Apply the rule once (Stage 2 must be complete); later calls read the file."""
    p = config.RESULTS_DIR / FROZEN
    if p.exists():
        return json.loads(p.read_text())
    t = table()
    if len(t) < len(MODELS) or not t['complete'].all():
        raise RuntimeError('Stage 2 not complete for every model -- run qnn_scan first')
    win = {f: t[t['family'] == f].sort_values('all', ascending=False).iloc[0] for f in QNN_FAMILIES}
    models = {win['qnn']['model']: json.loads(win['qnn']['recipe']),
              win['ann']['model']: json.loads(win['ann']['recipe']),
              'qn/identity': json.loads(win['qnn']['recipe'])}
    fz = {'written': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
          'rule': 'per family, highest Stage 2 val_last5 (7 crops x seeds 100-109)',
          'qnn': win['qnn']['model'], 'ann': win['ann']['model'],
          'stage2_val': {f: float(win[f]['all']) for f in QNN_FAMILIES},
          'crops': config.CROPS, 'seeds': SEEDS, 'models': models,
          'checkpoint': 'best validation epoch'}
    p.write_text(json.dumps(fz, indent=2))
    return fz


def _store_t(model, crop):
    return f"qtest_{model.replace('/', '_')}_{crop}"


def jobs():
    return [(m, c) for c in config.CROPS for m in freeze()['models']]


def run_job(model, crop, tag=''):
    from .results import ResultStore
    r = freeze()['models'][model]
    rid = recipe_id({**DEFAULT_RECIPE, **r})
    store = ResultStore(_store_t(model, crop))
    for s in SEEDS:
        if store.done(crop, model, rid, s, 'final'):
            continue
        row, hist = run_one(Q._data(crop), model, s, r, 'final')
        store.add(row, hist)
        print(f"{tag}{model:<14} {crop:<7} seed {s} | TEST {row['test_acc']:6.2f} | {row['seconds']:5.1f}s",
              flush=True)


def test_rows():
    fz = freeze()
    out = []
    for m, r in fz['models'].items():
        rid = recipe_id({**DEFAULT_RECIPE, **r})
        for c in config.CROPS:
            p = config.RESULTS_DIR / f'{_store_t(m, c)}.csv'
            if p.exists():
                d = pd.read_csv(p)
                out.append(d[(d['recipe'] == rid) & (d['phase'] == 'final')])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def report(path):
    from . import filtertest as FT
    fz = freeze()
    q, a = fz['qnn'], fz['ann']
    return FT.report(path, df=test_rows(), models=[q, a, 'qn/identity'],
                     pairs=[(q, a), (q, 'qn/identity')], recipes=fz['models'], frozen=FROZEN,
                     title='QNN vs ANN', csv='qnn_test.csv')
