"""QCNN + QNN vs CNN + ANN ensembles on TEST (written 2026-11-13). Trains nothing.

Rule (fixed before any ensemble number was computed): for each crop and seed,
average the two members' softmax probabilities 50/50 and take the argmax.
Members are the frozen test runs:
    image models   filter_test_frozen.json  (B6, Conv2x2 raw, PatchMLP raw)
    feature models qnn_test_frozen.json     (QNN L4, ANN mlp5_tanh)
The same seed of both members is combined (seed s with seed s); both see the
same test images in the same order (same create_dataframes split, unshuffled
test loader; checked again here per file).
Ensembles
    ens/quantum   B6 + QNN
    ens/classical Conv2x2 raw + ANN   (best classical members)
    ens/twin      PatchMLP raw + ANN  (strict parameter twin of the image model)
Each ensemble's per-seed predictions are written next to the members'
(model 'ens/...', recipe 'default'), so filtertest.report gives the same
statistics as the other tests.
"""
import json

import numpy as np
import pandas as pd

from . import config
from .train import DEFAULT_RECIPE, predictions_path, recipe_id

ENSEMBLES = {'ens/quantum': ('q/ry_cnot_B6', 'qn/sel_L4'),
             'ens/classical': ('c/Conv2x2/raw', 'cn/mlp5_tanh'),
             'ens/twin': ('c/PatchMLP/raw', 'cn/mlp5_tanh')}
PAIRS = [('ens/quantum', 'ens/classical'), ('ens/quantum', 'ens/twin')]
RID = recipe_id(DEFAULT_RECIPE)          # 'default'


def _recipes():
    R = config.RESULTS_DIR
    r = dict(json.loads((R / 'filter_test_frozen.json').read_text())['models'])
    q = json.loads((R / 'qnn_test_frozen.json').read_text())['models']
    r.update(q)
    return r


def _params():
    from . import filtertest as FT, qnnscan as S
    df = pd.concat([FT.rows(), S.test_rows()], ignore_index=True)
    return df.groupby('model')['total_params'].first().to_dict(), df


def build():
    """Write ensemble predictions; return rows like a ResultStore (one per crop x seed)."""
    rec = _recipes()
    params, members = _params()
    rows = []
    for name, (a, b) in ENSEMBLES.items():
        for c in config.CROPS:
            for s in config.SEEDS:
                za = np.load(predictions_path(c, a, recipe_id({**DEFAULT_RECIPE, **rec[a]}), s))
                zb = np.load(predictions_path(c, b, recipe_id({**DEFAULT_RECIPE, **rec[b]}), s))
                assert (za['y'] == zb['y']).all() and str(za['split']) == str(zb['split']), (name, c, s)
                p = (za['prob'] + zb['prob']) / 2
                pred = p.argmax(1)
                out = predictions_path(c, name, RID, s)
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez(out, pred=pred.astype(np.int16), y=za['y'], split=za['split'], prob=p.astype(np.float32))
                rows.append({'crop': c, 'model': name, 'recipe': RID, 'seed': s, 'phase': 'final',
                             'filter_params': params[a] + params[b], 'total_params': params[a] + params[b],
                             'test_acc': 100.0 * float((pred == za['y']).mean()),
                             'members': f'{a} + {b}'})
    return pd.DataFrame(rows), members


def report(path):
    from . import filtertest as FT
    df, members = build()
    df.to_csv(config.RESULTS_DIR / 'ensemble_rows.csv', index=False)
    # ensemble vs each of its members, paired by (crop, seed)
    lines = []
    for name, ms in ENSEMBLES.items():
        e = df[df['model'] == name].set_index(['crop', 'seed'])['test_acc']
        for m in ms:
            x = members[members['model'] == m].set_index(['crop', 'seed'])['test_acc'].reindex(e.index)
            d = (e - x).groupby('seed').mean()
            lines.append(f'{name} - {m}: {FT._ci(d)} (CI over seeds)')
    print('\n== ensemble minus its members, test accuracy')
    print(*lines, sep='\n')
    p = FT.report(path, df=df, models=list(ENSEMBLES), pairs=PAIRS,
                  recipes={m: {} for m in ENSEMBLES}, frozen='filter_test_frozen.json + qnn_test_frozen.json',
                  title='Ensembles (50/50 probability average)', csv='ensemble_test.csv')
    txt = p.read_text(encoding='utf-8').replace(
        '</body>', '<h2>Ensemble minus its members</h2><pre>' + '\n'.join(lines) + '</pre>'
        '<p>params row = total trainable parameters of both members.</p></body>')
    p.write_text(txt, encoding='utf-8')
    return p
