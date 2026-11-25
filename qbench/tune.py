"""Phase 2: equal-budget hyper-parameter search, on validation only.

Fairness rules
  * Every family gets the same number of trials (N_TRIALS).
  * Trial k uses the SAME training recipe in every family (one shared list
    of draws), so quantum and classical see identical recipes; only the
    filter differs.
  * Each family searches its own filters within the <=12-param budget.
  * phase='tune' -> the test set is never evaluated.
  * Tuning seeds (100-102) are disjoint from the final seeds (0-9).

Selection: highest val_last5 -- validation accuracy averaged over the last
5 epochs, then over the tuning seeds. This rewards the level a recipe
reliably reaches; ranking by the single best epoch instead rewards recipes
whose validation curve is noisy (one lucky spike wins). Ties broken by the
best-epoch val accuracy, then by fewer filter parameters.
"""
import json
import random

import numpy as np
import pandas as pd

from . import config
from .train import DEFAULT_RECIPE, recipe_id, run

N_TRIALS = 40
TUNE_SEEDS = [100, 101, 102]
SAMPLER_SEED = 2026

# the shared recipe space (identical for every family)
SPACE = {
    'act': ['none', 'relu', 'leaky', 'tanh'],
    'norm': ['none', 'bn'],
    'lr': [0.001, 0.002, 0.005, 0.01, 0.02],
    'epochs': [20, 40],
    'warmup_epochs': [0, 2],
}

# candidate filters per family: all <= 12 trainable filter params
FAMILIES = {
    'quantum': ['q/ry_cnot_B3', 'q/ry_cnot_B6', 'q/hea_L2', 'q/hea_L3',
                'q/ry_only_L2', 'q/ry_only_L3'],
    'classical_raw': ['c/Conv2x2/raw', 'c/Separable3x3/raw',
                      'c/FixedBasis6/raw', 'c/DWSep3x3/raw'],
    'classical_amp': ['c/Conv2x2/amp', 'c/Separable3x3/amp',
                      'c/FixedBasis6/amp', 'c/DWSep3x3/amp'],
}


def recipes(n=N_TRIALS, seed=SAMPLER_SEED):
    """The shared list of n recipe draws (same for every family and crop)."""
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        r = {k: rng.choice(v) for k, v in SPACE.items()}
        key = tuple(sorted(r.items()))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def trials(family, n=N_TRIALS):
    """[(model, recipe)] for a family: trial k = filters[k % F] + recipes[k]."""
    models = FAMILIES[family]
    return [(models[k % len(models)], r) for k, r in enumerate(recipes(n))]


def tune(family, crops, n=N_TRIALS, seeds=TUNE_SEEDS):
    """Run (or resume) every trial of `family` on `crops`."""
    store = f'tune_{family}'
    for k, (model, r) in enumerate(trials(family, n)):
        print(f"\n=== {family} trial {k + 1}/{n}: {model}  {recipe_id({**DEFAULT_RECIPE, **r})}")
        df = run(store, crops, [model], seeds=seeds, recipe=r, phase='tune', verbose=False)
        rid = recipe_id({**DEFAULT_RECIPE, **r})
        for crop in crops:
            v = df[(df['crop'] == crop) & (df['model'] == model) & (df['recipe'] == rid)
                   & df['seed'].isin(seeds)]['val_acc']
            sel = df[(df['crop'] == crop) & (df['model'] == model) & (df['recipe'] == rid)
                     & df['seed'].isin(seeds)]
            print(f"    {crop:<8} last5 {sel['val_last5'].mean():6.2f} | best-epoch val "
                  f"{v.mean():6.2f} +/- {v.std(ddof=1):4.2f}  ({' '.join(f'{x:.1f}' for x in v)})")


def leaderboard(family, crop):
    """One row per trial: mean/sd of val accuracy over the tuning seeds."""
    path = config.RESULTS_DIR / f'tune_{family}.csv'
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[(df['crop'] == crop) & (df['phase'] == 'tune') & df['seed'].isin(TUNE_SEEDS)]
    g = (df.groupby(['model', 'recipe'])
           .agg(val_last5=('val_last5', 'mean'), val_last5_sd=('val_last5', 'std'),
                val_mean=('val_acc', 'mean'), val_sd=('val_acc', 'std'),
                n=('val_acc', 'size'), filter_params=('filter_params', 'first'),
                best_epoch=('best_epoch', 'mean'), seconds=('seconds', 'mean'))
           .reset_index())
    g = g[g['n'] == len(TUNE_SEEDS)]
    return g.sort_values(['val_last5', 'val_mean', 'filter_params'],
                         ascending=[False, False, True]).reset_index(drop=True)


def dev_leaderboard(family, crops):
    """Phase-2 leaderboard averaged over crops (configs finished on every crop)."""
    boards = [leaderboard(family, c).assign(crop=c) for c in crops]
    df = pd.concat([b for b in boards if len(b)])
    g = (df.groupby(['model', 'recipe'])
           .agg(val_last5=('val_last5', 'mean'), n_crops=('crop', 'nunique'),
                filter_params=('filter_params', 'first'))
           .reset_index())
    g = g[g['n_crops'] == len(crops)]
    return g.sort_values(['val_last5', 'filter_params'],
                         ascending=[False, True]).reset_index(drop=True)


# ============================================================================
# Phase 2b: confirmation round (fixed BEFORE any of its results existed)
#
#   candidates : per family, the CONFIRM_TOP best configs of Phase 2 (mean
#                val_last5 over the dev crops); for quantum also the best
#                B6_RECIPES configs of q/ry_cnot_B6 if not already in
#   seeds      : CONFIRM_SEEDS (100-109; 100-102 reused from Phase 2)
#   score      : val_last5, mean over dev crops x seeds (validation only)
#   winner     : highest score per family
#   B6 rule    : q/ry_cnot_B6 is the main quantum model if its best config
#                is not significantly worse than the quantum winner: paired
#                by (crop, seed), the upper end of the 95% CI of
#                (B6 - winner) is >= 0. Declared in advance because the
#                paper studies the RY-CNOT entangling filter.
#                Otherwise the winner leads and B6 stays a named row.
# ============================================================================
DEV_CROPS = ['Cherry', 'Grape']          # Corn removed 2026-11-09 (config.EXCLUDED_CROPS)
CONFIRM_SEEDS = list(range(100, 110))
CONFIRM_TOP = 3
B6, B6_RECIPES = 'q/ry_cnot_B6', 2


def candidates(family, crops=DEV_CROPS):
    """[(model, recipe_id)] to confirm, chosen from Phase-2 results only."""
    lb = dev_leaderboard(family, crops)
    out = list(zip(lb['model'], lb['recipe']))[:CONFIRM_TOP]
    if family == 'quantum':
        b6 = lb[lb['model'] == B6]
        out += [c for c in zip(b6['model'], b6['recipe']) if c not in out][:B6_RECIPES]
    return out


def confirm(family, crops=DEV_CROPS, seeds=CONFIRM_SEEDS):
    """Run (or resume) the candidates on all dev crops x seeds. Same CSV as
    Phase 2, so seeds 100-102 are not re-run."""
    for k, (model, rid) in enumerate(candidates(family, crops)):
        print(f"\n=== confirm {family} {k + 1}: {model}  {rid}")
        run(f'tune_{family}', crops, [model], seeds=seeds,
            recipe=recipe_from_id(rid), phase='tune', verbose=False)
    print(confirm_board(family, crops, seeds).round(2).to_string(index=False))


def confirm_rows(family, crops=DEV_CROPS, seeds=CONFIRM_SEEDS):
    """All confirmation rows of the candidates (one per crop x seed)."""
    df = pd.read_csv(config.RESULTS_DIR / f'tune_{family}.csv')
    cand = set(candidates(family, crops))
    keep = pd.Series([(m, r) in cand for m, r in zip(df['model'], df['recipe'])], index=df.index)
    return df[keep & df['crop'].isin(crops) & df['seed'].isin(seeds)
              & (df['phase'] == 'tune')]


def confirm_board(family, crops=DEV_CROPS, seeds=CONFIRM_SEEDS):
    df = confirm_rows(family, crops, seeds)
    per_crop = df.pivot_table(index=['model', 'recipe'], columns='crop',
                              values='val_last5', aggfunc='mean')
    g = (df.groupby(['model', 'recipe'])
           .agg(score=('val_last5', 'mean'), sd=('val_last5', 'std'),
                n=('val_last5', 'size'), filter_params=('filter_params', 'first'))
           .join(per_crop).reset_index())
    g['complete'] = g['n'] == len(crops) * len(seeds)
    return g.sort_values(['complete', 'score', 'filter_params'],
                         ascending=[False, False, True]).reset_index(drop=True)


def _paired_ci(df, a, b):
    """(a - b) paired by (crop, seed): mean, 95% CI, paired t p-value."""
    from scipy import stats
    key = lambda c: df[(df['model'] == c[0]) & (df['recipe'] == c[1])].set_index(['crop', 'seed'])['val_last5']
    xa, xb = key(a), key(b)
    d = (xa - xb.reindex(xa.index)).dropna()
    m, h = d.mean(), stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    p = stats.ttest_rel(xa.reindex(d.index), xb.reindex(d.index)).pvalue
    return {'diff': float(m), 'lo': float(m - h), 'hi': float(m + h),
            'p_t': float(p), 'n_pairs': int(len(d))}


def freeze(crops=DEV_CROPS, seeds=CONFIRM_SEEDS):
    """Apply the decision rule -> results/frozen_configs.json."""
    out = {'dev_crops': crops, 'seeds': seeds, 'metric': 'val_last5', 'families': {}}
    for fam in FAMILIES:
        b = confirm_board(fam, crops, seeds)
        if not b['complete'].all():
            raise RuntimeError(f"{fam}: confirmation incomplete -- run confirm first")
        top = b.iloc[0]
        entry = {'model': top['model'], 'recipe': recipe_from_id(top['recipe']),
                 'score': float(top['score']), 'filter_params': int(top['filter_params']),
                 'per_crop': {c: float(top[c]) for c in crops}}
        if fam == 'quantum' and top['model'] != B6:
            b6 = b[b['model'] == B6].iloc[0]
            test = _paired_ci(confirm_rows(fam, crops, seeds),
                              (B6, b6['recipe']), (top['model'], top['recipe']))
            entry['validation_winner'] = dict(entry)
            entry['b6_vs_winner'] = test
            entry['b6_rule'] = 'B6 main if upper 95% CI of (B6 - winner) >= 0'
            if test['hi'] >= 0:
                entry.update({'model': B6, 'recipe': recipe_from_id(b6['recipe']),
                              'score': float(b6['score']),
                              'filter_params': int(b6['filter_params']),
                              'per_crop': {c: float(b6[c]) for c in crops}})
            entry['chosen_by'] = 'B6 rule' if test['hi'] >= 0 else 'validation winner'
        out['families'][fam] = entry
    path = config.RESULTS_DIR / 'frozen_configs.json'
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path}")
    return out


def recipe_from_id(rid):
    return {} if rid == 'default' else json.loads(rid)


def select(crops, families=tuple(FAMILIES), save=True, name='best_configs'):
    """Best (model, recipe) per family and crop -> results/<name>.json."""
    best = {}
    for crop in crops:
        best[crop] = {}
        for fam in families:
            lb = leaderboard(fam, crop)
            if not len(lb):
                continue
            top = lb.iloc[0]
            best[crop][fam] = {'model': top['model'], 'recipe': recipe_from_id(top['recipe']),
                               'val_last5': float(top['val_last5']),
                               'val_mean': float(top['val_mean']), 'val_sd': float(top['val_sd']),
                               'filter_params': int(top['filter_params']),
                               'trials_done': int(len(lb))}
    if save:
        path = config.RESULTS_DIR / f'{name}.json'
        path.write_text(json.dumps(best, indent=2))
        print(f"saved -> {path}")
    return best
