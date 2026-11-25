"""QNN vs parameter-matched ANN (qbench/qnn.py), same protocol as the QCNN.

Tuning (validation only, test never evaluated)
    families    qnn: qn/sel_L1..L4 (24-96 circuit params)
                ann: cn/dense_{tanh,relu} (72), cn/mlp5_{tanh,relu} (93)
                4 models each, same head Linear(8, 2).
    grid        lr in LRS x epochs {25, 50, 100} -> 48 configs per family,
                identical for both. 25 and 50 are read from the 100-epoch run
                (constant lr: exact prefix, see archstudy.check_prefix).
    data        dev crops Cherry, Corn, Grape; seeds 100-102.
    score       val_last5, mean over crops x seeds.
    freeze()    best config per family -> results/frozen_qnn.json. Written
                before any final run; used unchanged on every crop.

Final (the only phase that touches test)
    the two frozen configs + 0-parameter-block controls with the same recipe
    as their family (qn/identity, qn/random <- qnn; cn/linear <- ann) + the
    notebook's own settings as reference rows (qn/sel_L4 lr 0.01 / cn/ann_nb
    lr 0.001, 50 epochs). All 8 crops, seeds 0-9, store 'qnn_final'.
"""
import json

import pandas as pd

from . import config
from .archstudy import _prefix_row
from .qnn import QNN_FAMILIES
from .train import DEFAULT_RECIPE, predictions_path, recipe_id, run_one
from .tune import DEV_CROPS, TUNE_SEEDS, _paired_ci

LRS = [0.001, 0.003, 0.01, 0.03]
MAX_EPOCHS, SHORTER = 100, [25, 50]
NOTEBOOK = [('qn/sel_L4', {'lr': 0.01, 'epochs': 50}),
            ('cn/ann_nb', {'lr': 0.001, 'epochs': 50})]
CONTROLS = {'qnn': ['qn/identity', 'qn/random'], 'ann': ['cn/linear']}
FROZEN = 'frozen_qnn.json'
# Exact size match (requested in an earlier review). Rule, fixed before its test run: the QNN with
# the ANN winner's exact counts (72 block + 18 head = L=3), best config by the
# same validation score. Its validation winner within the size class is the ANN
# already frozen, so the ANN does not change. Written once, never reselected.
MATCHED = 'frozen_qnn_matched.json'
MATCHED_MODEL = 'qn/sel_L3'


def store_name(model):
    return 'qnn_tune_' + model.replace('/', '_')


def _data(crop, cache={}):
    from .qnn import FeatureData
    if crop not in cache:
        cache[crop] = FeatureData(crop)
    return cache[crop]


def run_models(models, crops=DEV_CROPS, seeds=TUNE_SEEDS, lrs=LRS, tag=''):
    from .results import ResultStore
    for model in models:
        store = ResultStore(store_name(model))
        for crop in crops:
            for lr in lrs:
                for s in seeds:
                    rid = recipe_id({**DEFAULT_RECIPE, 'lr': lr, 'epochs': MAX_EPOCHS})
                    if store.done(crop, model, rid, s, 'tune'):
                        continue
                    row, hist = run_one(_data(crop), model, s, {'lr': lr, 'epochs': MAX_EPOCHS}, 'tune')
                    row['derived_from'] = 0
                    store.add(row, hist)
                    short = [_prefix_row(row, hist, n, lr) for n in SHORTER]
                    for r, h in short:
                        store.add(r, h)
                    print(f"{tag}{model:<14} {crop:<7} lr {lr:<6} seed {s} | last5 "
                          + '  '.join(f"@{n} {r['val_last5']:6.2f}" for n, (r, _) in zip(SHORTER, short))
                          + f"  @{MAX_EPOCHS} {row['val_last5']:6.2f} | {row['seconds']:5.1f}s", flush=True)


def rows(models, crops=DEV_CROPS, seeds=TUNE_SEEDS):
    out = [pd.read_csv(p) for m in models
           if (p := config.RESULTS_DIR / f'{store_name(m)}.csv').exists()]
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    return df[df['crop'].isin(crops) & df['seed'].isin(seeds) & (df['phase'] == 'tune')]


def board(family, crops=DEV_CROPS, seeds=TUNE_SEEDS):
    """Every config of a family, complete grids only, best first."""
    df = rows(QNN_FAMILIES[family], crops, seeds)
    if not len(df):
        return pd.DataFrame()
    g = (df.groupby(['model', 'recipe'])
           .agg(score=('val_last5', 'mean'), sd=('val_last5', 'std'), n=('val_last5', 'size'),
                best_val=('val_acc', 'mean'), filter_params=('filter_params', 'first'),
                total_params=('total_params', 'first'), sec=('seconds', 'mean'))
           .join(df.pivot_table(index=['model', 'recipe'], columns='crop',
                                values='val_last5', aggfunc='mean'))
           .reset_index())
    g = g[g['n'] == len(crops) * len(seeds)]
    return g.sort_values('score', ascending=False).reset_index(drop=True)


def freeze():
    """Best config per family by validation; writes results/frozen_qnn.json."""
    out = {}
    for fam in QNN_FAMILIES:
        b = board(fam)
        need = len(QNN_FAMILIES[fam]) * len(LRS) * (1 + len(SHORTER))
        if len(b) < need:
            raise RuntimeError(f"{fam}: only {len(b)}/{need} configs complete -- finish tuning first")
        top = b.iloc[0]
        out[fam] = {'model': top['model'], 'recipe': json.loads(top['recipe']),
                    'val_last5': round(float(top['score']), 3),
                    'worst_config': round(float(b['score'].min()), 3)}
    df = rows([m for ms in QNN_FAMILIES.values() for m in ms])
    key = lambda f: (out[f]['model'], recipe_id({**DEFAULT_RECIPE, **out[f]['recipe']}))
    out['val_qnn_minus_ann'] = _paired_ci(df, key('qnn'), key('ann'))
    p = config.RESULTS_DIR / FROZEN
    p.write_text(json.dumps(out, indent=2))
    return out, p


def freeze_matched():
    """Best L=3 config by validation; refuses to change an existing file."""
    from .models import build_model, count_params
    p = config.RESULTS_DIR / MATCHED
    if p.exists():
        return json.loads(p.read_text())
    fz = json.loads((config.RESULTS_DIR / FROZEN).read_text())
    a, q = (count_params(build_model(m)) for m in (fz['ann']['model'], MATCHED_MODEL))
    if a != q:
        raise RuntimeError(f'not an exact match: ANN {a}, QNN {q}')
    b = board('qnn')
    top = b[b['model'] == MATCHED_MODEL].iloc[0]
    out = {'model': MATCHED_MODEL, 'recipe': json.loads(top['recipe']),
           'val_last5': round(float(top['score']), 3), 'params': q,
           'ann_model': fz['ann']['model'], 'ann_recipe': fz['ann']['recipe'],
           'rule': 'exact parameter match to the frozen ANN; best config by validation'}
    p.write_text(json.dumps(out, indent=2))
    return out


def final_jobs():
    """[(model, recipe)] for the final run, from frozen_qnn.json."""
    fz = json.loads((config.RESULTS_DIR / FROZEN).read_text())
    jobs = []
    for fam in QNN_FAMILIES:
        r = fz[fam]['recipe']
        jobs.append((fz[fam]['model'], r))
        jobs += [(c, r) for c in CONTROLS[fam]]
    for m, r in NOTEBOOK:
        if (m, r) not in jobs:
            jobs.append((m, r))
    p = config.RESULTS_DIR / MATCHED
    if p.exists():
        q = json.loads(p.read_text())
        if (q['model'], q['recipe']) not in jobs:
            jobs.append((q['model'], q['recipe']))
    return jobs


def _job_store(model, rid):
    return f"qnn_job_{predictions_path('x', model, rid, 0).stem.rsplit('__', 1)[0]}"


def run_final(jobs, crops=config.CROPS, seeds=config.SEEDS, tag=''):
    """Parallel workers: each job has its own store, rebuild_final() merges them.
    A run whose predictions exist is never repeated, so test stays once."""
    from .results import ResultStore
    for crop in crops:
        for m, r in jobs:
            rid = recipe_id({**DEFAULT_RECIPE, **r})
            store = ResultStore(_job_store(m, rid))
            for s in seeds:
                if predictions_path(crop, m, rid, s).exists():
                    continue
                row, hist = run_one(_data(crop), m, s, r, 'final')
                store.add(row, hist)
                print(f"{tag}{m:<14} {crop:<7} seed {s} | val {row['val_acc']:6.2f} "
                      f"| TEST {row['test_acc']:6.2f} | {row['seconds']:5.1f}s", flush=True)


def rebuild_final(write=True):
    """qnn_final.csv from the curves (validation) and saved predictions (test).
    Trains nothing and evaluates nothing."""
    import glob
    import numpy as np
    from .models import count_params
    from .qnn import build_feat_model
    R = config.RESULTS_DIR
    cur = pd.concat([pd.read_csv(f) for f in glob.glob(str(R / 'qnn_final_curves.csv'))
                     + glob.glob(str(R / 'qnn_job_*_curves.csv'))], ignore_index=True)
    cur = cur.drop_duplicates(subset=['crop', 'model', 'recipe', 'seed', 'phase', 'epoch'], keep='last')
    rows = []
    for m, r in final_jobs():
        rid = recipe_id({**DEFAULT_RECIPE, **r})
        c = cur[(cur['model'] == m) & (cur['recipe'] == rid)]
        counts = count_params(build_feat_model(m))
        for crop in config.CROPS:
            for s in config.SEEDS:
                p = predictions_path(crop, m, rid, s)
                h = c[(c['crop'] == crop) & (c['seed'] == s)].sort_values('epoch')
                if not p.exists() or len(h) != r['epochs']:
                    continue
                z = np.load(p, allow_pickle=False)
                v = h['val_acc'].to_numpy()
                rows.append({'crop': crop, 'model': m, 'recipe': rid, 'seed': s, 'phase': 'final',
                             **counts, 'val_acc': float(v.max()), 'best_epoch': int(v.argmax()),
                             'val_last5': float(v[-5:].mean()),
                             'test_acc': 100.0 * float((z['pred'] == z['y']).mean()),
                             'seconds': float('nan'), 'split': str(z['split'])})
    df = pd.DataFrame(rows)
    if write:
        df.to_csv(R / 'qnn_final.csv', index=False)
        cur.to_csv(R / 'qnn_final_curves.csv', index=False)
    return df
