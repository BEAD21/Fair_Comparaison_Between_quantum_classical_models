"""Quantum filter scan, VALIDATION ONLY (phase='tune': test is never evaluated).

Stage 1  every filter x the same 12 recipes (RECIPE_LIST), dev crops, seeds 100-102.
Stage 2  each filter's best recipe (Stage 1), all 8 crops, seeds 100-109.

Filters: 10-20 parameters in every family (IN_RANGE), plus the smaller filters of
the earlier search as information rows, plus two controls with nothing to train
in the circuit (identity, random). All are on the same 2x2 patches. Every
filter gets the same recipes, so the tuning effort is equal per filter.

Crops follow config.CROPS and tune.DEV_CROPS (Corn removed 2026-11-09). Stage 1
rows of other crops stay in the CSVs but are ignored; when the best recipe
changes, Stage 2 runs again with the new recipe and the table only reads rows of
the current best recipe.

Nothing is selected here: the report only shows the table.
"""
import html

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .tune import CONFIRM_SEEDS, DEV_CROPS, TUNE_SEEDS, recipe_from_id
from .train import DEFAULT_RECIPE, recipe_id, run_one

# Recipe grid, fixed 2026-11-09 before any run with it (replaces the random draw,
# which had no relu and 7-8 of 10 recipes with BN). Full factorial of the three
# settings that mattered in the earlier validation search on the dev crops:
# activation x BN x learning rate. Epochs and warmup moved val by ~0.3 pp, so
# they are fixed (50 epochs, warmup 2). The classical match uses the same grid.
GRID = {'act': ['tanh', 'relu', 'leaky'], 'norm': ['none', 'bn'], 'lr': [0.005, 0.01]}
FIXED = {'epochs': 50, 'warmup_epochs': 2}
RECIPE_LIST = [{'act': a, 'norm': n, 'lr': lr, **FIXED}
               for a in GRID['act'] for n in GRID['norm'] for lr in GRID['lr']]
RECIPES = len(RECIPE_LIST)                                # 12
RECIPE_IDS = {recipe_id({**DEFAULT_RECIPE, **r}) for r in RECIPE_LIST}


def draw_recipes(n=None):
    return list(RECIPE_LIST)
IN_RANGE = ([f'q/ry_cnot_B{b}' for b in range(5, 11)]           # 10, 12, ... 20
            + [f'q/ry_only_L{l}' for l in range(5, 11)]         # 10, 12, ... 20
            + ['q/hea_L3', 'q/hea_L4', 'q/hea_L5']              # 12, 16, 20
            + ['q/wgate_P2', 'q/wgate_P3']                      # 12, 18: the notebooks' w_gate_pl
            + ['q/su4'])                                        # 15: general 2-qubit gate (KAK)
SMALLER = ['q/ry_cnot_B3', 'q/ry_only_L2', 'q/ry_only_L3', 'q/hea_L2']
FILTERS = IN_RANGE + SMALLER
CONTROLS = ['q/identity', 'q/random']             # 0 trainable circuit parameters
# su4 last: it is the only untested new filter, so if it fails the rest still run
ALL = [m for m in FILTERS if m != 'q/su4'] + CONTROLS + ['q/su4']
REF = 'q/ry_cnot_B6'
# Classical counterparts (2026-11-10, after B6 was kept): same head (258), same
# grid, crops, seeds and selection as the quantum scan. PatchMLP is the strict
# match (12 params on the same 2x2 patches); Conv2x2 has the patches but 10
# params; the 3x3 models have 12 params but see overlapping 3x3 windows.
CLASSICAL = [f'c/{f}/{e}' for f in ('PatchMLP', 'Conv2x2', 'Separable3x3', 'FixedBasis6', 'DWSep3x3')
             for e in ('raw', 'amp')]
METRIC = 'val_last5'


def _store(model, stage):
    return f"qscan{stage}_{model.replace('/', '_')}"


def _run(store_name, data_of, model, recipe, crops, seeds, tag):
    from .results import ResultStore
    store = ResultStore(store_name)
    rid = recipe_id({**DEFAULT_RECIPE, **recipe})
    for crop in crops:
        for s in seeds:
            if store.done(crop, model, rid, s, 'tune'):
                continue
            row, hist = run_one(data_of(crop), model, s, recipe, 'tune')
            store.add(row, hist)
            print(f"{tag}{model:<16} {crop:<7} seed {s} | last5 {row[METRIC]:6.2f} | {row['seconds']:5.1f}s", flush=True)


def best_recipe(model):
    """Best complete recipe of Stage 1 on the dev crops, or None."""
    p = config.RESULTS_DIR / f'{_store(model, 1)}.csv'
    if not p.exists():
        return None
    d = pd.read_csv(p)
    # only recipes of the current grid: older recipes stay in the CSV but never compete
    d = d[d['crop'].isin(DEV_CROPS) & d['seed'].isin(TUNE_SEEDS) & d['recipe'].isin(RECIPE_IDS)]
    g = d.groupby(['recipe', 'crop'])[METRIC].mean().unstack('crop').reindex(columns=DEV_CROPS).dropna()
    if not len(g):
        return None
    return g, recipe_from_id(g.mean(axis=1).idxmax())


def run_model(models, tag='', crops=None, dev=None, tune_seeds=TUNE_SEEDS, seeds=CONFIRM_SEEDS,
              recipe_list=None):
    from .data import CropData
    cache = {}

    def data_of(c):
        if c not in cache:
            cache[c] = CropData(c)
        return cache[c]

    recipe_list = recipe_list if recipe_list is not None else draw_recipes(RECIPES)
    for m in models:
        for r in recipe_list:
            _run(_store(m, 1), data_of, m, r, dev or DEV_CROPS, tune_seeds, f'{tag}[1] ')
        _, r = best_recipe(m)
        _run(_store(m, 2), data_of, m, r, crops or config.CROPS, seeds, f'{tag}[2] ')


def table(models=ALL, crops=None, seeds=CONFIRM_SEEDS, ref=REF):
    """One row per filter with Stage 2 results of its CURRENT best recipe, on
    `crops` only; differences are to the reference."""
    crops = list(crops or config.CROPS)
    dev = [c for c in crops if c in DEV_CROPS]
    held = [c for c in crops if c not in DEV_CROPS]
    rows, by = [], {}
    for m in models:
        p = config.RESULTS_DIR / f'{_store(m, 2)}.csv'
        br = best_recipe(m)
        if not p.exists() or br is None:
            continue
        s1, rid = br[0].mean(axis=1), recipe_id({**DEFAULT_RECIPE, **br[1]})
        d = pd.read_csv(p)
        d = d[(d['recipe'] == rid) & d['crop'].isin(crops) & d['seed'].isin(seeds)]
        if not len(d):
            continue
        g = d.pivot_table(index='seed', columns='crop', values=METRIC).reindex(index=seeds, columns=crops)
        by[m] = g
        group = ('in range' if m in IN_RANGE else 'control' if m in CONTROLS
                 else 'classical' if m.startswith('c/') else 'smaller (info)')
        rows.append({'filter': m, 'group': group,
                     'params': int(d['filter_params'].iloc[0]),
                     **g.mean().round(2).to_dict(),
                     'dev': round(np.nanmean(g[dev].to_numpy()), 2),
                     'held_out': round(np.nanmean(g[held].to_numpy()), 2) if held else np.nan,
                     'all': round(np.nanmean(g.to_numpy()), 2),
                     'seed_sd': round(g.mean(axis=1).std(ddof=1), 2) if len(g) > 1 else np.nan,
                     'worst_crop': round(g.mean().min(), 2),
                     'recipe_worst': round(s1.min(), 2), 'recipe_median': round(s1.median(), 2),
                     'recipes': len(s1), 'runs': int(g.notna().to_numpy().sum()),
                     'complete': int(g.notna().to_numpy().sum()) == len(crops) * len(seeds),
                     'recipe': rid})
    t = pd.DataFrame(rows)
    if not len(t):
        return t
    if ref in by and len(seeds) > 1:
        for i, m in enumerate(t['filter']):
            dd = by[m] - by[ref]
            if m == ref or dd.isna().any().any():
                continue
            for unit, v, col in (('seeds', dd.mean(axis=1), 'ci_seeds'), ('crops', dd.mean(axis=0), 'ci_crops')):
                h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
                t.loc[i, 'minus_ref'] = round(v.mean(), 2)
                t.loc[i, col] = f'[{v.mean() - h:+.2f}, {v.mean() + h:+.2f}]'
    return t.sort_values(['group', 'all'], ascending=[True, False]).reset_index(drop=True)


def write_html(t, path, crops=None, title='Quantum filter scan: validation only'):
    crops = list(crops or config.CROPS)
    cols = ['filter', 'params', *crops, 'dev', 'held_out', 'all', 'seed_sd', 'worst_crop', 'minus_ref', 'ci_seeds',
            'ci_crops', 'recipe_worst', 'recipe_median', 'recipes', 'recipe', 'complete']
    cols = [c for c in cols if c in t.columns]
    names = {'held_out': 'held-out', 'seed_sd': 'seed SD', 'worst_crop': 'worst crop', 'minus_ref': f'minus {REF}',
             'ci_seeds': '95% CI (seeds)', 'ci_crops': '95% CI (crops)', 'recipe_worst': 'worst recipe (dev)',
             'recipe_median': 'median recipe (dev)'}

    def cell(v):
        return '' if (isinstance(v, float) and np.isnan(v)) else html.escape(str(v))
    out = ['<!DOCTYPE html><html><head><meta charset="utf-8"><title>Quantum filter scan</title><style>'
           'body{font:14px system-ui,sans-serif;margin:24px;color:#1a1a1a}table{border-collapse:collapse;margin-bottom:28px}'
           'th,td{border:1px solid #ccc;padding:4px 8px;text-align:right;white-space:nowrap}th{background:#f2f2f2}'
           'td:first-child,th:first-child{text-align:left}</style></head><body>',
           f'<h1>{html.escape(title)}</h1>',
           f'<p>Mean last-5-epoch validation accuracy, seeds 100-109, each filter with its best of {RECIPES} recipes '
           f'(chosen on {", ".join(DEV_CROPS)}, seeds 100-102). Test data was not used. No filter has been selected.</p>']
    for grp, sec in (('in range', '10 to 20 parameters'), ('smaller (info)', 'Smaller filters, information only'),
                     ('control', 'Controls: nothing to train in the circuit'),
                     ('classical', 'Classical counterparts (same head, grid, crops and seeds)')):
        s = t[t['group'] == grp]
        if len(s):
            out.append(f'<h2>{sec}</h2><table><tr>' + ''.join(f'<th>{names.get(c, c)}</th>' for c in cols) + '</tr>')
            out += ['<tr>' + ''.join(f'<td>{cell(r[c])}</td>' for c in cols) + '</tr>' for _, r in s.iterrows()]
            out.append('</table>')
    out.append('</body></html>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(out), encoding='utf-8')
    return path
