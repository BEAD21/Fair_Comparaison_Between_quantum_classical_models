"""Final TEST of the filter comparison, run once (written 2026-11-10).

Everything is fixed before any test number is seen:
  models   B6 (quantum), Conv2x2 raw (best classical in the validation scan),
           PatchMLP raw (strict 12-parameter twin on the same 2x2 patches),
           identity (quantum pipeline, nothing to train in the circuit)
  recipes  each model's best recipe of the 12-recipe grid (filterscan.best_recipe,
           Cherry + Grape, seeds 100-102), written to FROZEN on the first call
           and never recomputed afterwards
  data     7 crops (config.CROPS), seeds 0-9, phase='final'; the checkpoint is
           the best validation epoch, as in every earlier final run
  stats    paired by (crop, seed): mean difference with 95% t-intervals over
           seeds and over crops; exact McNemar on each model's MAJORITY VOTE
           over the 10 seeds (5-5 ties -> mean softmax probability), pooled over
           crops and per crop with Holm correction
Predictions (class + softmax probability) are saved per image for the ensembles.
"""
import html
import json

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .train import DEFAULT_RECIPE, predictions_path, recipe_id, run_one

MODELS = ['q/ry_cnot_B6', 'c/Conv2x2/raw', 'c/PatchMLP/raw', 'q/identity']
REF = 'q/ry_cnot_B6'
PAIRS = [(REF, 'c/Conv2x2/raw'), (REF, 'c/PatchMLP/raw'), (REF, 'q/identity')]
FROZEN = 'filter_test_frozen.json'
SEEDS = config.SEEDS


def freeze():
    """Write the frozen recipes once; later calls only read the file."""
    p = config.RESULTS_DIR / FROZEN
    if p.exists():
        return json.loads(p.read_text())
    from . import filterscan as F
    fz = {'written': '2026-11-10', 'crops': config.CROPS, 'seeds': SEEDS, 'models': {},
          'checkpoint': 'best validation epoch', 'mcnemar': 'majority vote over seeds, ties by mean probability',
          'ensembles': 'plain average of softmax probabilities, no weights tuned'}
    for m in MODELS:
        br = F.best_recipe(m)
        if br is None:
            raise RuntimeError(f'no scan results for {m}')
        fz['models'][m] = br[1]
    p.write_text(json.dumps(fz, indent=2))
    return fz


def _store(model, crop):
    return f"ftest_{model.replace('/', '_')}_{crop}"


def jobs():
    return [(m, c) for c in config.CROPS for m in MODELS]


def run_job(model, crop, tag=''):
    from .data import CropData
    from .results import ResultStore
    recipe = freeze()['models'][model]
    store = ResultStore(_store(model, crop))
    rid = recipe_id({**DEFAULT_RECIPE, **recipe})
    data = None
    for s in SEEDS:
        if store.done(crop, model, rid, s, 'final'):
            continue
        if data is None:
            data = CropData(crop)
        row, hist = run_one(data, model, s, recipe, 'final')
        store.add(row, hist)
        print(f"{tag}{model:<16} {crop:<7} seed {s} | TEST {row['test_acc']:6.2f} | {row['seconds']:5.1f}s", flush=True)


def rows():
    fz = freeze()
    out = []
    for m in MODELS:
        rid = recipe_id({**DEFAULT_RECIPE, **fz['models'][m]})
        for c in config.CROPS:
            p = config.RESULTS_DIR / f'{_store(m, c)}.csv'
            if p.exists():
                d = pd.read_csv(p)
                out.append(d[(d['recipe'] == rid) & (d['phase'] == 'final')])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _votes(model, crop, recipe=None):
    """Majority vote over the seeds for every test image, and the labels."""
    rid = recipe_id({**DEFAULT_RECIPE, **(recipe if recipe is not None else freeze()['models'][model])})
    zs = [np.load(predictions_path(crop, model, rid, s)) for s in SEEDS]
    y = zs[0]['y']
    assert all((z['y'] == y).all() for z in zs), 'test order differs between seeds'
    ones = np.sum([z['pred'] == 1 for z in zs], axis=0)
    p1 = np.mean([z['prob'][:, 1] for z in zs], axis=0)
    vote = np.where(ones * 2 > len(zs), 1, np.where(ones * 2 < len(zs), 0, (p1 > 0.5).astype(int)))
    return vote, y


def mcnemar(a, b, recipes=None):
    """Exact McNemar per crop + pooled, majority votes; Holm over crops.
    recipes: {model: recipe}, default the frozen filter-test recipes."""
    recipes = recipes or freeze()['models']
    res, B, C = [], 0, 0
    for c in config.CROPS:
        va, y = _votes(a, c, recipes[a])
        vb, y2 = _votes(b, c, recipes[b])
        assert (y == y2).all(), 'test labels differ between models'
        b_, c_ = int(((va == y) & (vb != y)).sum()), int(((va != y) & (vb == y)).sum())
        B, C = B + b_, C + c_
        p = stats.binomtest(min(b_, c_), b_ + c_, 0.5).pvalue if b_ + c_ else 1.0
        res.append({'crop': c, 'n': len(y), 'acc_a': 100 * (va == y).mean(), 'acc_b': 100 * (vb == y).mean(),
                    'a_only': b_, 'b_only': c_, 'p': p})
    d = pd.DataFrame(res)
    order = d['p'].argsort().to_numpy()                       # Holm step-down
    adj, run = np.empty(len(d)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(d) - k) * d['p'].iloc[i]))
        adj[i] = run
    d['p_holm'] = adj
    pooled = {'a_only': B, 'b_only': C, 'p': stats.binomtest(min(B, C), B + C, 0.5).pvalue if B + C else 1.0}
    return d, pooled


def _ci(v):
    h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return f'{v.mean():+.2f} [{v.mean() - h:+.2f}, {v.mean() + h:+.2f}]'


def report(path, df=None, models=MODELS, pairs=PAIRS, recipes=None, frozen=FROZEN,
           title='Filter comparison', csv='filter_test.csv'):
    """Table + paired stats. Defaults: the filter test; qnnscan passes its own."""
    df = rows() if df is None else df
    recipes = recipes or freeze()['models']
    MODELS, PAIRS, FROZEN = models, pairs, frozen
    crops = config.CROPS
    piv = {m: df[df['model'] == m].pivot_table(index='seed', columns='crop', values='test_acc')
              .reindex(index=SEEDS, columns=crops) for m in MODELS}
    done = {m: int(piv[m].notna().to_numpy().sum()) for m in MODELS}
    t = pd.DataFrame({m: piv[m].mean() for m in MODELS}).T
    t['mean'] = [np.nanmean(piv[m].to_numpy()) for m in MODELS]
    t['seed SD'] = [piv[m].mean(axis=1).std(ddof=1) for m in MODELS]
    t['params'] = [int(df[df['model'] == m]['filter_params'].iloc[0]) if done[m] else np.nan for m in MODELS]
    t['runs'] = [done[m] for m in MODELS]
    # mean +- SD over the 10 seeds; for 'mean' the SD is of the 7-crop average per seed
    pm = pd.DataFrame({c: [f'{piv[m][c].mean():.2f} ± {piv[m][c].std(ddof=1):.2f}' for m in MODELS] for c in crops},
                      index=MODELS)
    pm['mean (7 crops)'] = [f'{piv[m].mean(axis=1).mean():.2f} ± {piv[m].mean(axis=1).std(ddof=1):.2f}'
                            for m in MODELS]
    pm['params'] = t['params'].astype('Int64').astype(str)
    full = len(crops) * len(SEEDS)
    pair_rows = []
    for a, b in PAIRS:
        if done[a] < full or done[b] < full:
            continue
        dd = piv[a] - piv[b]
        mc, pooled = mcnemar(a, b, recipes)
        pair_rows.append({'pair': f'{a} - {b}', 'diff [CI seeds]': _ci(dd.mean(axis=1)),
                          'diff [CI crops]': _ci(dd.mean(axis=0)),
                          f'{a} only right': pooled['a_only'], f'other only right': pooled['b_only'],
                          'McNemar p (pooled)': f"{pooled['p']:.3g}",
                          'crops p_holm<0.05': ', '.join(mc.loc[mc['p_holm'] < .05, 'crop']) or 'none',
                          '_mc': mc})
    print('\n== test accuracy, mean ± SD over seeds 0-9')
    print(pm.T.to_string())
    for r in pair_rows:
        print(f"\n{r['pair']}: seeds {r['diff [CI seeds]']} | crops {r['diff [CI crops]']} | "
              f"McNemar pooled p={r['McNemar p (pooled)']} | per crop (Holm) significant: {r['crops p_holm<0.05']}")
        print(r['_mc'].round(4).to_string(index=False))

    def tab(d, idx=True):
        return d.to_html(float_format=lambda v: f'{v:.2f}', index=idx, border=0, na_rep='')
    out = ['<!DOCTYPE html><html><head><meta charset="utf-8"><title>Filter test</title><style>'
           'body{font:14px system-ui,sans-serif;margin:24px;color:#1a1a1a}table{border-collapse:collapse;margin-bottom:24px}'
           'th,td{border:1px solid #ccc;padding:4px 8px;text-align:right;white-space:nowrap}th{background:#f2f2f2}'
           f'</style></head><body><h1>{html.escape(title)}: TEST (run once, configs frozen before)</h1>',
           f'<p>Test accuracy (%), mean ± SD over seeds 0-9, one row per crop. Last row: the 7-crop average, '
           f'mean ± SD over seeds. Recipes frozen in {FROZEN}. '
           'McNemar: exact, on each model\'s majority vote over the 10 seeds; Holm over crops.</p>',
           tab(pm.T)]
    for r in pair_rows:
        out.append(f"<h2>{html.escape(r['pair'])}</h2><p>difference, CI over seeds: {r['diff [CI seeds]']}; "
                   f"CI over crops: {r['diff [CI crops]']}; pooled McNemar p = {r['McNemar p (pooled)']}</p>")
        out.append(tab(r['_mc'].rename(columns={'a_only': 'only first right', 'b_only': 'only second right'}), False))
    out.append('</body></html>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(out), encoding='utf-8')
    t.to_csv(config.RESULTS_DIR / csv)
    return path
