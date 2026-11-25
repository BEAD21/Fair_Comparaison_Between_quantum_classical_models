r"""Every number, table and data-driven figure of the ESANN paper, from the saved
results (trains nothing, evaluates nothing). Writes paper_figures/*.tex
(Table 1, Fig. 2 and Fig. 3 of the paper) and paper_figures/pairs.csv.

    python paper_numbers.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('QBENCH_DATA', os.path.join(HERE, 'data', 'PlantVillage_Processed_700 images'))
os.environ.setdefault('QBENCH_WORK', os.path.join(HERE, 'work'))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
from scipy import stats

from qbench import config, ensemble as E, filtertest as FT, qnnscan as S, robust as RB

OUT = os.path.join(HERE, 'paper_figures')
os.makedirs(OUT, exist_ok=True)
R = config.RESULTS_DIR
CROPS = config.CROPS


def fp(p):
    return f'{p:.2f}' if p >= 0.01 else f'{p:.3f}'


def ci(v):
    v = np.asarray(v, float)
    h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - h, v.mean() + h


# ---------------------------------------------------------------- test rows
# encoded-input classical twins (revision A2): best-epoch rows of the traced re-runs
_rb = RB.rows()
rows = pd.concat([FT.rows(), S.test_rows(), pd.read_csv(R / 'ensemble_rows.csv'),
                  _rb[_rb['model'].isin(RB.AMP)]], ignore_index=True)
piv = {m: g.pivot_table(index='seed', columns='crop', values='test_acc').reindex(index=config.SEEDS, columns=CROPS)
       for m, g in rows.groupby('model')}
for m, p in piv.items():
    assert p.notna().all().all(), f'{m}: incomplete'
params = rows.groupby('model')['total_params'].first().astype(int).to_dict()

recipes = {**FT.freeze()['models'], **S.freeze()['models'], **{m: {} for m in E.ENSEMBLES},
           **{m: RB.freeze()['models'][m] for m in RB.AMP}}
qnn_fz = S.freeze()

# ------------------------------------------------------------ main table
TABLE = [('QCNN', [('q/ry_cnot_B6', 'QCNN, B6 filter (quantum)'), ('c/Conv2x2/raw', 'CNN, Conv2x2 (best on val.)'),
                   ('c/PatchMLP/raw', 'CNN, PatchMLP (exact twin)'),
                   ('c/Conv2x2/amp', 'Conv2x2, encoded input'), ('c/PatchMLP/amp', 'PatchMLP, encoded input'),
                   ('q/identity', 'Identity (no trained circuit)')]),
         ('QNN', [('qn/sel_L4', 'QNN, 4 layers (quantum)'), ('cn/mlp5_tanh', 'ANN, 8--5--8 (classical)'),
                  ('qn/identity', 'Encoding only (no trained circuit)')]),
         ('ENS', [('ens/quantum', 'QCNN + QNN'), ('ens/classical', 'Conv2x2 + ANN'),
                  ('ens/twin', 'PatchMLP + ANN')])]
QUANTUM = {'q/ry_cnot_B6', 'qn/sel_L4', 'ens/quantum'}
lines = [r'\begin{tabular}{@{}l@{\hspace{4pt}}r' + 'r' * len(CROPS) + r'@{\hspace{5pt}}c@{}}', r'\toprule',
         'Model & Par. & ' + ' & '.join(c[:3] for c in CROPS) + r' & Mean $\pm$ SD \\']
heads = {'QCNN': 'Image models (32$\\times$32 pixels)', 'QNN': 'Feature models (8 PCA features)',
         'ENS': 'Ensembles (50/50 average of probabilities)'}
for key, models in TABLE:
    lines += [r'\midrule', rf'\multicolumn{{{len(CROPS) + 3}}}{{@{{}}l}}{{\emph{{{heads[key]}}}}} \\']
    # best per crop among the non-control rows of the block
    real = [m for m, _ in models if 'identity' not in m and '/amp' not in m]
    best = {c: max(piv[m][c].mean() for m in real) for c in CROPS}
    for m, label in models:
        p = piv[m]
        cells = []
        for c in CROPS:
            v = p[c].mean()
            s = f'{v:.1f}'
            cells.append(rf'\textbf{{{s}}}' if m in real and abs(v - best[c]) < 1e-9 else s)
        per_seed = p.mean(axis=1)
        mean = f'{per_seed.mean():.2f} $\\pm$ {per_seed.std(ddof=1):.2f}'
        lab = rf'\textcolor{{qcol}}{{{label}}}' if m in QUANTUM else label
        lines.append(f'{lab} & {params[m]} & ' + ' & '.join(cells) + f' & {mean} \\\\')
lines += [r'\bottomrule', r'\end{tabular}']
open(os.path.join(OUT, 'table_test.tex'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

# ------------------------------------------------------ paired comparisons
PAIRS = [('q/ry_cnot_B6', 'c/Conv2x2/raw', 'QCNN vs Conv2x2'),
         ('q/ry_cnot_B6', 'c/PatchMLP/raw', 'QCNN vs PatchMLP'),
         ('qn/sel_L4', 'cn/mlp5_tanh', 'QNN vs ANN'),
         ('ens/quantum', 'ens/classical', 'Q-ens.\\ vs Conv2x2+ANN'),
         ('ens/quantum', 'ens/twin', 'Q-ens.\\ vs PatchMLP+ANN'),
         ('q/ry_cnot_B6', 'c/PatchMLP/amp', 'QCNN vs PatchMLP, enc.')]
CONTROLS = [('q/ry_cnot_B6', 'q/identity'), ('qn/sel_L4', 'qn/identity')]
res = []
for a, b, lab in PAIRS + [(a, b, '') for a, b in CONTROLS]:
    d = piv[a] - piv[b]
    m, slo, shi = ci(d.mean(axis=1))
    _, clo, chi = ci(d.mean(axis=0))
    mc, pooled = FT.mcnemar(a, b, recipes)
    res.append({'a': a, 'b': b, 'label': lab, 'diff': m, 'seed_lo': slo, 'seed_hi': shi,
                'crop_lo': clo, 'crop_hi': chi, 'mcnemar_p': pooled['p'],
                'a_only': pooled['a_only'], 'b_only': pooled['b_only'],
                'holm_sig': ', '.join(mc.loc[mc['p_holm'] < .05, 'crop']) or 'none',
                'crops_q_wins': int((d.mean(axis=0) > 0).sum())})
res = pd.DataFrame(res)
res.to_csv(os.path.join(OUT, 'pairs.csv'), index=False)

# forest plot (TikZ): x in percentage points, 1 unit = XS cm
XS, LO, HI = 0.74, -5, 3
fl = [r'\begin{tikzpicture}[x=%.2fcm,y=0.46cm,font=\footnotesize]' % XS]
n = len(PAIRS)
fl.append(rf'\fill[black!4] ({LO},0.4) rectangle (0,{n + 0.6});')
fl.append(rf'\node[font=\scriptsize,black!55,anchor=south west] at ({LO},{n + 0.55}) {{classical better}};')
fl.append(rf'\node[font=\scriptsize,black!55,anchor=south east] at ({HI},{n + 0.55}) {{quantum better}};')
fl.append(rf'\draw[black!50,dashed] (0,0.4) -- (0,{n + 0.6});')
for i, r in res.head(n).iterrows():
    y = n - i
    fl.append(rf'\node[anchor=east] at ({LO - 0.1},{y}) {{{r["label"]}}};')
    fl.append(rf'\draw[qcol!45,line width=0.6pt] ({r.crop_lo:.3f},{y}) -- ({r.crop_hi:.3f},{y});')
    fl.append(rf'\draw[qcol!45] ({r.crop_lo:.3f},{y - 0.15}) -- ++(0,0.3) ({r.crop_hi:.3f},{y - 0.15}) -- ++(0,0.3);')
    fl.append(rf'\draw[qcol,line width=2.2pt] ({r.seed_lo:.3f},{y}) -- ({r.seed_hi:.3f},{y});')
    fl.append(rf'\fill[white,draw=qcol,line width=0.8pt] ({r["diff"]:.3f},{y}) circle (2.6pt);')
    fl.append(rf'\node[anchor=west,font=\scriptsize] at ({HI + 0.05},{y}) {{${r["diff"]:+.2f}$, $p={fp(r.mcnemar_p)}$}};')
fl.append(rf'\draw ({LO},0.4) -- ({HI},0.4);')
for t in range(LO, HI + 1):
    fl.append(rf'\draw ({t},0.4) -- ({t},0.3) node[below,font=\scriptsize] {{${t:+d}$}};'.replace('$+0$', '$0$'))
fl.append(rf'\node[below,font=\scriptsize] at ({(LO + HI) / 2},-0.25) {{test accuracy difference, quantum $-$ classical (percentage points)}};')
fl.append(r'\end{tikzpicture}')
open(os.path.join(OUT, 'forest.tex'), 'w', encoding='utf-8').write('\n'.join(fl) + '\n')

# ------------------------------------------------------- validation scans
qs = pd.read_csv(R / 'filter_scan.csv')
cs = pd.read_csv(R / 'classical_scan.csv')
cs = cs[cs['filter'].str.startswith('c/')]
qs_f = qs[~qs['filter'].isin(['q/identity', 'q/random'])]
ctrl = qs.set_index('filter')['all']


def coords(d, dx):
    return ' '.join(f'({int(p) + dx},{a:.2f})' for p, a in zip(d['params'], d['all']))


def name(f):
    return f.split('/')[1]


sl = [r'\begin{tikzpicture}', r'\begin{axis}[width=7.4cm,height=3.4cm,font=\footnotesize,',
      r'  xlabel={trainable filter parameters},ylabel={validation acc.\ (\%)},xmin=1,xmax=21,',
      r'  ymin=%.1f,ymax=%.1f,xtick={2,4,...,20},grid=major,grid style={black!8},' % (
          min(qs_f['all'].min(), cs['all'].min(), ctrl.min()) - 0.4, max(qs_f['all'].max(), cs['all'].max()) + 0.8),
      r'  legend style={font=\scriptsize,at={(1.03,0.5)},anchor=west,draw=none},legend cell align=left]',
      r'\addplot[only marks,mark=*,mark size=1.8pt,qcol,fill opacity=0.7] coordinates {%s};' % coords(qs_f, -0.2),
      r'\addlegendentry{quantum filters (%d)}' % len(qs_f),
      r'\addplot[only marks,mark=square*,mark size=1.8pt,ccol] coordinates {%s};' % coords(cs, 0.2),
      r'\addlegendentry{classical filters (%d)}' % len(cs),
      r'\addplot[black!60,dashed,domain=1:21] {%.2f};' % ctrl['q/identity'],
      r'\addlegendentry{identity control}']
b6 = qs.set_index('filter').loc['q/ry_cnot_B6']
cv = cs.set_index('filter').loc['c/Conv2x2/raw']
pm = cs.set_index('filter').loc['c/PatchMLP/raw']
# labels in empty regions, with leader lines to the (shifted) points
lab = [(b6, -0.2, 'qcol', 'B6 (chosen)', (6.0, 87.85)), (cv, 0.2, 'ccol', 'Conv2x2', (15.5, 87.85)),
       (pm, 0.2, 'ccol', 'PatchMLP', (14.0, 82.75))]
for row, dx, col, text, (lx, ly) in lab:
    x, y = row['params'] + dx, row['all']
    sl.append(r'\draw[%s,thin] (axis cs:%.2f,%.2f) -- (axis cs:%.2f,%.2f) node[font=\scriptsize,fill=white,'
              r'inner sep=1pt,anchor=%s] {%s};' % (col, x, y, lx, ly, 'east' if lx < x else 'west', text))
sl += [r'\end{axis}', r'\end{tikzpicture}']
open(os.path.join(OUT, 'scan.tex'), 'w', encoding='utf-8').write('\n'.join(sl) + '\n')

# --------------------------------------------------------------- printout
pd.set_option('display.width', 220)
print('== pairs (test)')
print(res.round(3).to_string(index=False))
print('\n== scan (validation, all crops)')
print(qs[['filter', 'params', 'all', 'dev', 'held_out']].sort_values('all', ascending=False).head(8).to_string(index=False))
print(qs[qs['filter'].isin(['q/identity', 'q/random'])][['filter', 'all']].to_string(index=False))
print(cs[['filter', 'params', 'all']].sort_values('all', ascending=False).to_string(index=False))
print('in-range quantum spread:', qs_f['all'].min(), qs_f['all'].max(), 'n=', len(qs_f))
print('\nQNN stage 2:', qnn_fz['stage2_val'])
print('params:', params)
print('written ->', os.path.abspath(OUT))
