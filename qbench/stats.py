"""Summaries and paired comparisons over seeds."""
import numpy as np
import pandas as pd
from scipy import stats


def ci95(v):
    """(mean, half-width of the t-based 95% CI)."""
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return (v.mean() if len(v) else float('nan')), float('nan')
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def summarize(df, metric='test_acc', by=('crop', 'model', 'recipe')):
    """mean +/- 95% CI, sd, n, per group."""
    rows = []
    for key, g in df.groupby(list(by)):
        m, h = ci95(g[metric])
        rows.append({**dict(zip(by, key)), 'filter_params': g['filter_params'].iloc[0],
                     'mean': m, 'ci95': h, 'sd': g[metric].std(ddof=1), 'n': len(g)})
    return pd.DataFrame(rows)


def paired(df, a, b, crop, metric='test_acc', recipe_a=None, recipe_b=None):
    """a - b paired by seed: mean diff, 95% CI, paired t p-value, Wilcoxon p."""
    def pick(model, recipe):
        s = df[(df['crop'] == crop) & (df['model'] == model)]
        if recipe is not None:
            s = s[s['recipe'] == recipe]
        if s['recipe'].nunique() > 1:
            raise ValueError(f"{model} has several recipes on {crop}; pass recipe_*")
        return s.set_index('seed')[metric]
    x, y = pick(a, recipe_a), pick(b, recipe_b)
    seeds = x.index.intersection(y.index)
    d = (x.loc[seeds] - y.loc[seeds]).values
    m, h = ci95(d)
    p_t = stats.ttest_rel(x.loc[seeds], y.loc[seeds]).pvalue if len(d) > 1 else float('nan')
    p_w = (stats.wilcoxon(d).pvalue if len(d) > 1 and np.any(d != 0) else float('nan'))
    return {'crop': crop, 'a': a, 'b': b, 'n': len(d), 'diff': m,
            'lo': m - h, 'hi': m + h, 'p_t': p_t, 'p_wilcoxon': p_w}
