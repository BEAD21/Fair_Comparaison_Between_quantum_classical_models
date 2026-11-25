"""Checks every quantum circuit of the pipeline against PennyLane.

Random parameters and inputs (random numbers + a real training image or its
features; never test data) go through the pipeline's simulator here and through
PennyLane default.qubit in a separate environment (tests/pl_reference.py). The
two outputs must agree to float32 precision.

The negative controls give PennyLane slightly different parameters on purpose:
they must FAIL, which shows the comparison can detect a mismatch.
"""
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from . import config
from .arch import expz, probs
from .data import CropData
from .models import build_model
from .qnn import FeatureData, N_FEAT

TOL = 1e-5
DRAWS = 2                              # parameter draws per circuit
N_PATCHES = 32                         # patches compared per filter draw
SHIFT = 0.1                            # parameter change in the negative controls

FILTERS = {'q/identity': ('identity', 2), 'q/random': ('random', 2),
           'q/ry_cnot_B3': ('ry_cnot', 2), 'q/ry_cnot_B6': ('ry_cnot', 2),
           'q/hea_L2': ('hea', 2), 'q/hea_L3': ('hea', 2),
           'q/ry_only_L2': ('ry_only', 2), 'q/ry_only_L3': ('ry_only', 2),
           'q/ry_cnot_B10': ('ry_cnot', 2), 'q/ry_only_L6': ('ry_only', 2), 'q/ry_only_L10': ('ry_only', 2),
           'q/hea_L5': ('hea', 2), 'q/wgate_P2': ('wgate', 2), 'q/wgate_P3': ('wgate', 2),
           'q/qaoa_p6': ('qaoa', 2), 'q/qaoa_p10': ('qaoa', 2),
           'q/ry_cnot_B6_k4': ('ry_cnot', 4), 'q/ry_cnot_B6_k8': ('ry_cnot', 6),
           'q/ry_cnot_B6_k16': ('ry_cnot', 8)}
_z79 = lambda m, x: expz(m.circuit(x), [7, 9])
_p9 = lambda m, x: torch.exp(m(x))
ARCH = {  # tag: (filter, ancilla, classifier, output, feature function)
    'qa/hc': ('b6', False, None, 'z79', _z79),
    'qa/hc_nb': ('nb', False, None, 'z79', _z79),
    'qa/ancilla': ('b6', True, None, 'p379', lambda m, x: probs(m.circuit(x), [3, 7, 9])),
    'qa/ancilla_nb': ('nb', True, None, 'p379', lambda m, x: probs(m.circuit(x), [3, 7, 9])),
    'qa/fqc': ('b6', False, 'ry', 'p9', _p9),
    'qa/fqc_sel': ('b6', False, 'sel', 'p9', _p9),
    'qa/fqc_nb': ('nb', False, 'sel', 'p9', _p9),
    'qa/fqc_nb_scale': ('nb', False, 'sel', 'z9', lambda m, x: 2 * m(x)[:, :1] / m.fc.s),
}


def _params(module):
    """Non-trivial angles everywhere; dict of the circuit's arrays."""
    out = {}
    with torch.no_grad():
        for name, p in module.named_parameters():
            p.uniform_(-3, 3)
            out[name] = p.numpy().copy()
    return out


def export(folder):
    """Runs the pipeline's simulator; writes cases.json / cases.npz."""
    grape = CropData('Grape')
    feats = FeatureData('Grape')
    imgs = torch.cat([torch.rand(1, 32, 32), torch.as_tensor(grape.X_train[:1])]).float()
    cases, arrays, fast = [], {}, {}

    def add(meta, arr, y):
        i = len(cases)
        cases.append(meta)
        for k, v in arr.items():
            arrays[f'{i}/{k}'] = v
        fast[str(i)] = y.detach().numpy().astype(np.float64)

    with torch.no_grad():
        for d in range(DRAWS):
            torch.manual_seed(d)
            for name, (ansatz, n) in FILTERS.items():
                f = build_model(name).filt
                a = _params(f)
                if ansatz == 'random':
                    a['U'] = f.U.numpy()
                k = f.kernel_size
                x = imgs[:, None]
                p = F.unfold(x, kernel_size=k, stride=f.stride).permute(0, 2, 1).reshape(-1, k * k)
                y = f(x).reshape(x.shape[0], n, -1).permute(0, 2, 1).reshape(-1, n)
                keep = (p.norm(dim=1) > 0).nonzero()[:, 0]
                keep = keep[torch.randperm(len(keep))[:N_PATCHES]]
                add({'name': name, 'kind': 'filter', 'ansatz': ansatz, 'n': n, 'draw': d},
                    {**a, 'x': p[keep].numpy()}, y[keep])

            xf = torch.cat([torch.randn(2, N_FEAT) * 2, torch.as_tensor(feats.X_train[:2])]).float()
            for L in (1, 2, 3, 4):
                m = build_model(f'qn/sel_L{L}')
                w = _params(m.filt)['w']
                add({'name': f'qn/sel_L{L}', 'kind': 'qnn', 'draw': d}, {'w': w, 'x': xf.numpy()}, m.filt(xf))

            xi = imgs.reshape(2, -1)
            for tag, (filt, anc, vqc, out, fn) in ARCH.items():
                m = build_model(tag)
                a = {k.split('.')[-1]: v for k, v in _params(m.filt).items()}
                add({'name': tag, 'kind': 'arch', 'filt': filt, 'ancilla': anc, 'vqc': vqc, 'out': out, 'draw': d},
                    {**a, 'x': xi.numpy()}, fn(m, imgs))

        # negative controls: PennyLane receives one slightly different angle
        for src, key in (('qa/hc_nb', 'pool'), ('qa/ancilla_nb', 'theta'), ('qn/sel_L4', 'w'), ('q/ry_cnot_B6', 'theta')):
            i = next(j for j, c in enumerate(cases) if c['name'] == src)
            a = {k.split('/', 1)[1]: v for k, v in arrays.items() if k.startswith(f'{i}/')}
            a[key] = a[key].copy()
            a[key].flat[0] += SHIFT
            add({**cases[i], 'name': f'{src} (shifted on purpose)', 'negative': True}, a, torch.as_tensor(fast[str(i)]))

    np.savez(f'{folder}/cases.npz', **arrays)
    np.savez(f'{folder}/fast.npz', **fast)
    json.dump(cases, open(f'{folder}/cases.json', 'w'))
    return cases


def compare(folder):
    """-> (table, all passed). Positive cases must agree, negative ones must not."""
    cases = json.load(open(f'{folder}/cases.json'))
    a, b = np.load(f'{folder}/fast.npz'), np.load(f'{folder}/reference.npz')
    rows = []
    for i, c in enumerate(cases):
        rows.append({'name': c['name'], 'negative': bool(c.get('negative')),
                     'err': float(np.abs(a[str(i)] - b[str(i)]).max()), 'values': int(a[str(i)].size)})
    return rows


def run(python):
    folder = config.WORK_DIR / 'pl_check'
    folder.mkdir(parents=True, exist_ok=True)
    cases = export(folder)
    print(f"{len(cases)} cases exported, running PennyLane ...", flush=True)
    subprocess.run([python, str(Path(__file__).resolve().parent.parent / 'tests' / 'pl_reference.py'), str(folder)], check=True)
    rows = compare(folder)
    versions = json.load(open(folder / 'reference_versions.json'))
    worst = {}
    for r in rows:
        w = worst.setdefault(r['name'], {'negative': r['negative'], 'err': 0.0, 'values': 0})
        w['err'] = max(w['err'], r['err'])
        w['values'] += r['values']
    print(f"\nPennyLane {versions['pennylane']} ({versions['device']}), tolerance {TOL:g}")
    print(f"{'circuit':<42}{'values':>7}{'max |diff|':>13}   result")
    ok = True
    for name, w in worst.items():
        good = w['err'] > 10 * TOL if w['negative'] else w['err'] < TOL
        ok &= good
        print(f"{name:<42}{w['values']:>7}{w['err']:>13.2e}   "
              + ('caught' if w['negative'] and good else 'OK' if good else 'MISMATCH'))
    print('\npl_check:', 'ALL OK' if ok else 'FAILED')
    (config.RESULTS_DIR / 'pennylane_check.json').write_text(
        json.dumps({'versions': versions, 'tol': TOL, 'ok': bool(ok), 'circuits': worst}, indent=2))
    return ok
