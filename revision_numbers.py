r"""Numbers for the revision (2026-11-19). Trains nothing.

A1  equivalence (TOST, margin +-1 pp, stated before computing) on the paired
    crop-level differences (n = 7) of the five quantum - classical comparisons;
    seed-level (n = 10) shown as a secondary column.  Minimum detectable effect
    at 80 % power (two-sided paired t, alpha 0.05) from the observed crop SD.
A6  crop-level SD of every test model.
A2/A3 (when the robust runs exist): amp-encoded classical twins, and every
    comparison re-scored with the last-five-epoch test mean.

    .venv\Scripts\python revision_numbers.py
Writes work/results/revision_*.csv
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('QBENCH_DATA', os.path.join(HERE, 'data', 'PlantVillage_Processed_700 images'))
os.environ.setdefault('QBENCH_WORK', os.path.join(HERE, 'work'))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
from scipy import optimize, stats

from qbench import config, filtertest as FT, qnnscan as S

R = config.RESULTS_DIR
CROPS = config.CROPS
MARGIN = 1.0
PAIRS = [('q/ry_cnot_B6', 'c/Conv2x2/raw', 'QCNN vs Conv2x2'),
         ('q/ry_cnot_B6', 'c/PatchMLP/raw', 'QCNN vs PatchMLP'),
         ('qn/sel_L4', 'cn/mlp5_tanh', 'QNN vs ANN'),
         ('ens/quantum', 'ens/classical', 'Q-ens vs Conv2x2+ANN'),
         ('ens/quantum', 'ens/twin', 'Q-ens vs PatchMLP+ANN')]


def pivots(rows, col='test_acc'):
    out = {}
    for m, g in rows.groupby('model'):
        p = g.pivot_table(index='seed', columns='crop', values=col).reindex(index=config.SEEDS, columns=CROPS)
        if p.notna().all().all():
            out[m] = p
    return out


def tost(d, margin=MARGIN):
    """Paired TOST: 90 % t-interval inside (-margin, +margin) <=> both one-sided p < .05."""
    d = np.asarray(d, float)
    n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    h = stats.t.ppf(.95, n - 1) * se
    p_lo = 1 - stats.t.cdf((m + margin) / se, n - 1)     # H0: diff <= -margin
    p_hi = stats.t.cdf((m - margin) / se, n - 1)         # H0: diff >= +margin
    return m, m - h, m + h, max(p_lo, p_hi)


def mde(sd, n, power=.8, alpha=.05):
    """Smallest true mean difference a two-sided paired t-test detects with `power`."""
    tc = stats.t.ppf(1 - alpha / 2, n - 1)

    def pw(delta):
        nc = delta / (sd / np.sqrt(n))
        return stats.nct.sf(tc, n - 1, nc) + np.nan_to_num(stats.nct.cdf(-tc, n - 1, nc)) - power
    return optimize.brentq(pw, 1e-6, 4 * sd)


def compare(piv, pairs=PAIRS, tag=''):
    res = []
    for a, b, lab in pairs:
        if a not in piv or b not in piv:
            continue
        d = piv[a] - piv[b]
        dc, ds = d.mean(axis=0), d.mean(axis=1)
        m, lo, hi, p = tost(dc)
        _, slo, shi, sp = tost(ds)
        res.append({'set': tag, 'comparison': lab, 'diff': m, 'crop_ci90_lo': lo, 'crop_ci90_hi': hi,
                    'tost_p_crops': p, 'equivalent_1pp_crops': bool(lo > -MARGIN and hi < MARGIN),
                    'crop_sd': dc.std(ddof=1), 'mde80_crops': mde(dc.std(ddof=1), len(dc)),
                    'seed_ci90_lo': slo, 'seed_ci90_hi': shi, 'tost_p_seeds': sp,
                    'equivalent_1pp_seeds': bool(slo > -MARGIN and shi < MARGIN),
                    'p_two_sided_crops': stats.ttest_1samp(dc, 0).pvalue})
    return pd.DataFrame(res)


def main():
    rows = pd.concat([FT.rows(), S.test_rows(), pd.read_csv(R / 'ensemble_rows.csv')], ignore_index=True)
    piv = pivots(rows)

    a1 = compare(piv, tag='best-epoch (paper)')
    a1.to_csv(R / 'revision_a1_tost.csv', index=False)
    pd.set_option('display.width', 250)
    print('A1  TOST, margin +-1 pp\n', a1.round(3).to_string(index=False))

    sd = pd.DataFrame({m: {'mean': p.values.mean(), 'sd_seeds': p.mean(axis=1).std(ddof=1),
                           'sd_crops': p.mean(axis=0).std(ddof=1)} for m, p in piv.items()}).T
    sd.to_csv(R / 'revision_a6_sd.csv')
    print('\nA6  SD over seeds vs over crops\n', sd.round(2).to_string())

    # A2 / A3 from the traced re-runs, if present
    try:
        from qbench import robust as RB
        tr = RB.rows()
    except Exception as e:                      # noqa: BLE001
        print('\nrobust runs not available:', e)
        return
    if not len(tr):
        print('\nrobust runs not available yet')
        return
    n, bad = RB.check_reproduction(tr)
    print(f'\nreproduction: {n} traced runs compared with the originals, {len(bad)} differ')
    out = []
    for col, tag in [('test_acc', 'best-epoch (traced)'), ('test_last5', 'last-5 mean'),
                     ('test_last5_probavg', 'last-5 prob. average')]:
        pv = pivots(tr, col)
        pv.update(ensembles(tr, col))
        pairs = PAIRS + [('q/ry_cnot_B6', 'c/Conv2x2/amp', 'QCNN vs Conv2x2 (amp)'),
                         ('q/ry_cnot_B6', 'c/PatchMLP/amp', 'QCNN vs PatchMLP (amp)'),
                         ('c/PatchMLP/raw', 'c/PatchMLP/amp', 'PatchMLP raw vs amp'),
                         ('c/Conv2x2/raw', 'c/Conv2x2/amp', 'Conv2x2 raw vs amp')]
        out.append(compare(pv, pairs, tag))
        means = {m: round(p.values.mean(), 2) for m, p in pv.items()}
        print(f'\n{tag}: means', means)
    out = pd.concat(out, ignore_index=True)
    out.to_csv(R / 'revision_a2_a3.csv', index=False)
    print(out[['set', 'comparison', 'diff', 'crop_ci90_lo', 'crop_ci90_hi', 'seed_ci90_lo', 'seed_ci90_hi',
               'p_two_sided_crops', 'equivalent_1pp_crops']].round(2).to_string(index=False))


def ensembles(tr, col):
    """Ensemble accuracies from the traced probabilities (best-epoch or last-5 averaged)."""
    from qbench import robust as RB
    from qbench.ensemble import ENSEMBLES
    kind = 'prob' if col == 'test_acc' else 'prob_last5'
    if col == 'test_last5':                     # a mean of per-epoch accuracies has no ensemble analogue
        return {}
    out = {}
    have = set(tr['model'])
    for name, (a, b) in ENSEMBLES.items():
        if a not in have or b not in have:
            continue
        p = pd.DataFrame(index=config.SEEDS, columns=CROPS, dtype=float)
        for c in CROPS:
            for s in config.SEEDS:
                pa, y = RB.load_probs(a, c, s, kind)
                pb, y2 = RB.load_probs(b, c, s, kind)
                assert (y == y2).all()
                p.loc[s, c] = 100 * ((pa + pb) / 2).argmax(1).__eq__(y).mean()
        out[name] = p
    return out


if __name__ == '__main__':
    main()
