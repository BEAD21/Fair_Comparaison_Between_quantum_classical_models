"""Architecture studies, validation only.

study 'arch' (40 epochs, all with the B6 filter)
    A. filter size   q/ry_cnot_B6 on 2x2 / 4x4 / 8x8 / 16x16 patches
                     (2 / 4 / 6 / 8 qubits), Baseline Hybrid skeleton
    B. architecture  Baseline Hybrid vs HC, Ancilla, FQC (first rebuild)

study 'arch100' (100 epochs, the notebooks' own models)
    Baseline Hybrid vs qa/hc_nb, qa/ancilla_nb, qa/fqc_nb exactly as their
    notebooks (filters, init, heads), + qa/fqc_nb_scale (1 classical output
    scale) as a diagnostic row. 100 epochs because the whole-image circuits
    were still improving at 40 (fqc_test on Grape).

Equal budget for every model: learning rate in LRS, seeds 100-102, the 3
dev crops; no act/norm options, no warmup.

Shorter runs for free: with a constant learning rate, an n-epoch run is
exactly the first n epochs of the longer run with the same seed (same init,
same batch order, same Adam state). So each run also yields the rows for
the shorter epoch counts (column derived_from). check_prefix() verifies it.

Score: val_last5, mean over crops x seeds. Test is never evaluated.
"""
import numpy as np
import pandas as pd

from . import config
from .train import DEFAULT_RECIPE, recipe_id, run_one
from .tune import DEV_CROPS, TUNE_SEEDS

LRS = [0.001, 0.002, 0.005, 0.01, 0.02]

STUDIES = {
    'arch': {
        'prefix': 'arch_', 'max_epochs': 40, 'shorter': [20],
        'groups': {
            'filter_size': ['q/ry_cnot_B6', 'q/ry_cnot_B6_k4', 'q/ry_cnot_B6_k8',
                            'q/ry_cnot_B6_k16'],
            'architecture': ['q/ry_cnot_B6', 'qa/hc', 'qa/ancilla', 'qa/fqc'],
        },
        # measured ms/batch: small-qubit models are as fast or faster on CPU,
        # the 10-11 qubit circuits faster on GPU -> both devices at once
        'workers': [('cpu', ['q/ry_cnot_B6', 'q/ry_cnot_B6_k4', 'q/ry_cnot_B6_k8']),
                    ('cuda', ['q/ry_cnot_B6_k16', 'qa/hc', 'qa/ancilla', 'qa/fqc'])],
    },
    'arch100': {
        'prefix': 'arch100_', 'max_epochs': 100, 'shorter': [20, 40],
        'groups': {
            'architecture100': ['q/ry_cnot_B6', 'qa/hc_nb', 'qa/ancilla_nb', 'qa/fqc_nb',
                                'qa/fqc_nb_scale'],
        },
        # ~ms/batch GPU / CPU: hc 12/20, ancilla 17/19, fqc 15/20, B6 10/5
        'workers': [('cuda', ['qa/hc_nb', 'qa/fqc_nb']),
                    ('cpu', ['q/ry_cnot_B6', 'qa/ancilla_nb']),
                    ('cpu', ['qa/fqc_nb_scale'])],
    },
}


def models_of(study):
    return list(dict.fromkeys(m for ms in STUDIES[study]['groups'].values() for m in ms))


def store_name(model, study='arch'):
    return STUDIES[study]['prefix'] + model.replace('/', '_')


def _prefix_row(row, hist, n, lr):
    """The n-epoch row contained in a longer run's history."""
    h = hist[:n]
    vals = [e['val_acc'] for e in h]
    best = int(np.argmax(vals))                 # first max, as fit() does
    return {**row, 'recipe': recipe_id({**DEFAULT_RECIPE, 'lr': lr, 'epochs': n}),
            'val_acc': vals[best], 'best_epoch': best,
            'val_last5': float(np.mean(vals[-5:])),
            'seconds': round(row['seconds'] * n / len(hist), 1),
            'derived_from': len(hist)}, h


def run_models(models, study='arch', crops=DEV_CROPS, seeds=TUNE_SEEDS, tag='', lrs=None):
    """Run (or resume) every model x crop x lr x seed at the study's max epochs."""
    from .data import CropData
    from .results import ResultStore
    S = STUDIES[study]
    E = S['max_epochs']
    for model in models:
        store = ResultStore(store_name(model, study))
        for crop in crops:
            todo = [(lr, s) for lr in (lrs or LRS) for s in seeds
                    if not store.done(crop, model,
                                      recipe_id({**DEFAULT_RECIPE, 'lr': lr, 'epochs': E}),
                                      s, 'tune')]
            if not todo:
                continue
            data = CropData(crop)
            for lr, s in todo:
                row, hist = run_one(data, model, s, {'lr': lr, 'epochs': E}, 'tune')
                row['derived_from'] = 0
                store.add(row, hist)
                short = [_prefix_row(row, hist, n, lr) for n in S['shorter']]
                for r, h in short:
                    store.add(r, h)
                extra = f" s={row['scale_s']:.1f}" if 'scale_s' in row else ''
                print(f"{tag}{model:<18} {crop:<7} lr {lr:<6} seed {s} | last5 "
                      + '  '.join(f"@{n} {r['val_last5']:6.2f}" for n, (r, _) in
                                  zip(S['shorter'], short))
                      + f"  @{E} {row['val_last5']:6.2f} | {row['seconds']:5.1f}s{extra}",
                      flush=True)


def rows(models, study='arch', crops=DEV_CROPS, seeds=TUNE_SEEDS):
    out = []
    for m in models:
        p = config.RESULTS_DIR / f'{store_name(m, study)}.csv'
        if p.exists():
            out.append(pd.read_csv(p))
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    return df[df['crop'].isin(crops) & df['seed'].isin(seeds) & (df['phase'] == 'tune')]


def board(group, study='arch', crops=DEV_CROPS, seeds=TUNE_SEEDS, df=None):
    """Best recipe per model of a group (+ per-crop scores), complete grids only.
    df: use these rows instead of reading the study's CSVs (e.g. one lr only)."""
    models = STUDIES[study]['groups'][group]
    df = rows(models, study, crops, seeds) if df is None else df
    if not len(df):
        return pd.DataFrame()
    need = len(crops) * len(seeds)
    agg = dict(score=('val_last5', 'mean'), n=('val_last5', 'size'),
               filter_params=('filter_params', 'first'), head_params=('head_params', 'first'),
               total_params=('total_params', 'first'), sec=('seconds', 'mean'))
    if 'scale_s' in df:
        agg['scale_s'] = ('scale_s', 'mean')
    g = (df.groupby(['model', 'recipe']).agg(**agg)
           .join(df.pivot_table(index=['model', 'recipe'], columns='crop',
                                values='val_last5', aggfunc='mean'))
           .reset_index())
    g = g[g['n'] == need]
    g = g.sort_values('score', ascending=False).groupby('model').head(1)
    worst = (df.groupby(['model', 'recipe'])['val_last5'].mean()
               .groupby('model').min().rename('worst_recipe'))
    order = {m: i for i, m in enumerate(models)}
    return (g.merge(worst, on='model')
             .sort_values('model', key=lambda s: s.map(order)).reset_index(drop=True))


def check_prefix(model='q/ry_cnot_B6', crop='Grape', seed=100, lr=0.005, n=20, longer=40):
    """A real n-epoch run must equal the first n epochs of a longer run."""
    from .data import CropData
    d = CropData(crop)
    rn, hn = run_one(d, model, seed, {'lr': lr, 'epochs': n}, 'tune')
    rl, hl = run_one(d, model, seed, {'lr': lr, 'epochs': longer}, 'tune')
    p, _ = _prefix_row(rl, hl, n, lr)
    diff = max(abs(a['val_acc'] - b['val_acc']) for a, b in zip(hn, hl[:n]))
    print(f"prefix check {model} {crop}: real {n}-ep last5 {rn['val_last5']:.3f}, "
          f"derived {p['val_last5']:.3f}, max per-epoch val diff {diff:.3f}")
    return diff
