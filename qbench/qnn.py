"""QNN vs parameter-matched classical network, on the notebook's features.

Ported from fork-of-svm-qsvm-ann-qnn-2-finished-all-crops.ipynb (the code
behind the paper's ANN/QNN columns):

features   128x128 colour image -> HOG (16x16 cells, 2x2 blocks) + HSV colour
           histogram (8x8x8) + uniform LBP (P=24, R=3) = 2302 numbers
           -> PCA(8) -> StandardScaler, both fitted on TRAIN only.
           PCA uses svd_solver='full' (the notebook's default picked the
           randomized solver without a seed, so its features were not exactly
           repeatable).
QNN        AngleEmbedding (RX(x_i) on wire i) + StronglyEntanglingLayers (L
           layers, 24 params each) + <Z> on all 8 wires -> Linear(8, 2).
           Notebook: L = 4 -> 96 circuit + 18 head = 114 parameters.
           Weights init uniform(0, 2 pi), as qml.qnn.TorchLayer.

Model names (all share the head Linear(8, 2) = 18 parameters, except ann_nb):
    qn/sel_L<L>      QNN with L layers                  24 L block params
    qn/identity      encoding only: <Z_i> = cos(x_i)    0
    qn/random        frozen random L=4 circuit          0
    cn/linear        Linear(8, 2) on the features       0
    cn/dense_<act>   Linear(8, 8) + act                 72
    cn/mlp5_<act>    Linear(8,5) act Linear(5,8) act    93
    cn/ann_nb        the notebook ANN 8-10-4-2          144 in total (reference)
act = tanh | relu. The block plays the part of the circuit: 8 numbers in,
8 numbers out, at most 96 parameters (the notebook circuit).

Simulator: per layer, the 8 Rot gates are two 16x16 Kronecker products applied
to the state as K_hi @ psi @ K_lo^T (psi as 16x16), then the CNOT ring as an
index permutation. verify_qnn() checks it against the full-unitary path
(sel_unitary) and a separate gate-by-gate simulation.
"""
import math

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import config
from .arch import _rot
from .data import create_dataframes, _write_manifest

N_FEAT = 8
FEAT_SIZE = 128


# ============================================================================
# features
# ============================================================================
def extract_features(path, size=FEAT_SIZE):
    """The notebook's extract_combined_features, unchanged."""
    from skimage.feature import hog, local_binary_pattern
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ch = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(ch, ch)
    P, R = 24, 3
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    lh, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    lh = lh.astype('float') / (lh.sum() + 1e-6)
    return np.hstack([h, ch.flatten(), lh])


def _features(df, label_map):
    X, y = [], []
    for path, label in zip(df['Filepaths'], df['Labels']):
        f = extract_features(path)
        if f is not None:
            X.append(f)
            y.append(label_map[label])
    return np.array(X), np.array(y, dtype=np.int64)


class FeatureData:
    """Same splits as CropData; X_* are the 8 PCA features (float32).
    Works with make_loaders / run_one unchanged."""

    def __init__(self, crop, n_components=N_FEAT, use_cache=True):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        self.crop = crop
        train_df, test_df, valid_df = create_dataframes(config.DATA_DIR, crop)
        self.split_hash = _write_manifest(
            crop, {'train': train_df, 'validation': valid_df, 'test': test_df})
        self.label_map = {l: i for i, l in enumerate(sorted(train_df['Labels'].unique()))}
        self.num_classes = len(self.label_map)

        cache = config.CACHE_DIR / f'{crop}_feat{FEAT_SIZE}.npz'
        raw = None
        if use_cache and cache.exists():
            z = np.load(cache, allow_pickle=False)
            if str(z['split_hash']) == self.split_hash:
                raw = {k: z[k] for k in z.files if k != 'split_hash'}
        if raw is None:
            raw = {}
            for s, df in (('train', train_df), ('val', valid_df), ('test', test_df)):
                raw[f'F_{s}'], raw[f'y_{s}'] = _features(df, self.label_map)
            if use_cache:
                config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
                np.savez(cache, split_hash=self.split_hash, **raw)

        pca = PCA(n_components=n_components, svd_solver='full')
        sc = StandardScaler()
        self.X_train = sc.fit_transform(pca.fit_transform(raw['F_train'])).astype(np.float32)
        self.X_val = sc.transform(pca.transform(raw['F_val'])).astype(np.float32)
        self.X_test = sc.transform(pca.transform(raw['F_test'])).astype(np.float32)
        self.y_train, self.y_val, self.y_test = raw['y_train'], raw['y_val'], raw['y_test']
        self.explained = float(pca.explained_variance_ratio_.sum())

    def __repr__(self):
        return (f"FeatureData({self.crop}: train={len(self.y_train)} val={len(self.y_val)} "
                f"test={len(self.y_test)} PCA{N_FEAT} keeps {self.explained:.1%} "
                f"split={self.split_hash})")


# ============================================================================
# 8-qubit simulator (wire 0 = most significant bit, as PennyLane)
# ============================================================================
def _cnot_ring_perm(n, r):
    """Index map of CNOT(i, (i+r) % n) for i = 0..n-1, applied in that order:
    new_state[k] = old_state[perm[k]]."""
    perm = np.arange(2 ** n)
    for i in range(n):
        c, t = i, (i + r) % n
        bc, bt = n - 1 - c, n - 1 - t
        k = np.arange(2 ** n)
        src = np.where((k >> bc) & 1, k ^ (1 << bt), k)   # CNOT is its own inverse
        perm = perm[src]
    return torch.as_tensor(perm)


def _zsign(n):
    k = np.arange(2 ** n)[:, None]
    bits = (k >> (n - 1 - np.arange(n))[None, :]) & 1
    return torch.as_tensor(1.0 - 2.0 * bits, dtype=torch.float32)       # (2^n, n)


def sel_unitary(w):
    """w (L, n, 3) -> (2^n, 2^n): qml.StronglyEntanglingLayers, default ranges."""
    L, n, _ = w.shape
    U = None
    for l in range(L):
        R = _rot(w[l])                                       # (n, 2, 2)
        K = R[0]
        for i in range(1, n):
            K = torch.kron(K, R[i])
        if n > 1:
            K = K[_cnot_ring_perm(n, l % (n - 1) + 1).to(w.device)]
        U = K if U is None else K @ U
    return U


def angle_states(x):
    """(B, n) -> (B, 2^n): prod_i RX(x_i)|0>."""
    c, s = torch.cos(x / 2), torch.sin(x / 2)
    v = torch.stack([torch.complex(c, torch.zeros_like(c)),
                     torch.complex(torch.zeros_like(s), -s)], -1)   # (B, n, 2)
    psi = v[:, 0]
    for i in range(1, x.shape[1]):
        psi = (psi[:, :, None] * v[:, i, None, :]).reshape(x.shape[0], -1)
    return psi


class _Circuit(nn.Module):
    def __init__(self, n=N_FEAT, layers=4, frozen=False):
        super().__init__()
        self.layers = layers
        if layers:
            self.w = nn.Parameter(torch.rand(layers, n, 3) * 2 * math.pi,
                                  requires_grad=not frozen)
        self.register_buffer('zs', _zsign(n))
        self.register_buffer('perm', torch.stack([_cnot_ring_perm(n, r + 1) for r in range(n - 1)]))

    def forward(self, x):
        psi = angle_states(x)
        B, n = x.shape
        h = n // 2
        for l in range(self.layers):
            # Rot layer = K_hi (x) K_lo: apply as K_hi @ psi(16x16) @ K_lo^T, then the CNOT ring
            R = _rot(self.w[l])
            Khi, Klo = R[0], R[h]
            for i in range(1, h):
                Khi, Klo = torch.kron(Khi, R[i]), torch.kron(Klo, R[h + i])
            psi = (Khi @ psi.reshape(B, 2 ** h, 2 ** (n - h)) @ Klo.transpose(0, 1)).reshape(B, -1)
            psi = psi[:, self.perm[l % (n - 1)]]
        p = psi.real ** 2 + psi.imag ** 2
        return p @ self.zs                                   # <Z_i>, (B, n)


class FeatNet(nn.Module):
    """.filt = the block (circuit or classical), .fc = the head."""

    def __init__(self, filt, fc):
        super().__init__()
        self.filt, self.fc = filt, fc

    def forward(self, x):
        return self.fc(self.filt(x))


_ACTS = {'tanh': nn.Tanh, 'relu': nn.ReLU}
QNN_FAMILIES = {
    'qnn': ['qn/sel_L1', 'qn/sel_L2', 'qn/sel_L3', 'qn/sel_L4'],
    'ann': ['cn/dense_tanh', 'cn/dense_relu', 'cn/mlp5_tanh', 'cn/mlp5_relu'],
}


def build_feat_model(name, num_classes=2, seed=0):
    kind, tag = name.split('/')
    n = N_FEAT
    head = lambda: nn.Linear(n, num_classes)
    if kind == 'qn':
        if tag == 'identity':
            return FeatNet(_Circuit(n, 0), head())
        if tag == 'random':
            return FeatNet(_Circuit(n, 4, frozen=True), head())
        if tag.startswith('sel_L'):
            return FeatNet(_Circuit(n, int(tag[5:])), head())
    if kind == 'cn':
        if tag == 'linear':
            return FeatNet(nn.Sequential(), head())
        if tag == 'ann_nb':                       # notebook ANN, 8-10-4-2
            return FeatNet(nn.Sequential(nn.Linear(n, 10), nn.ReLU(), nn.Linear(10, 4), nn.ReLU()),
                           nn.Linear(4, num_classes))
        base, _, act = tag.partition('_')
        if act in _ACTS and base == 'dense':
            return FeatNet(nn.Sequential(nn.Linear(n, n), _ACTS[act]()), head())
        if act in _ACTS and base == 'mlp5':
            return FeatNet(nn.Sequential(nn.Linear(n, 5), _ACTS[act](),
                                         nn.Linear(5, n), _ACTS[act]()), head())
    raise ValueError(f"bad feature-model name '{name}'")


# ============================================================================
# check
# ============================================================================
def _dense_reference(x, w):
    """Gate by gate on a (2,)*n tensor, no shared code with the fast path
    except the Rot matrix."""
    n = x.shape[0]
    psi = torch.zeros([2] * n, dtype=torch.complex128)
    psi[(0,) * n] = 1

    def one(psi, G, q):
        return torch.movedim(torch.tensordot(G, psi, dims=([1], [q])), 0, q)

    for q in range(n):
        c, s = math.cos(x[q] / 2), math.sin(x[q] / 2)
        psi = one(psi, torch.tensor([[c, -1j * s], [-1j * s, c]], dtype=torch.complex128), q)
    for l in range(w.shape[0]):
        for q in range(n):
            psi = one(psi, _rot(w[l, q].double()).to(torch.complex128), q)
        r = l % (n - 1) + 1
        for q in range(n):
            c, t = q, (q + r) % n
            idx = [slice(None)] * n
            idx[c] = 1
            sub = psi[tuple(idx)].clone()
            tt = t if t < c else t - 1
            psi[tuple(idx)] = sub.flip(tt)
    p = psi.abs() ** 2
    return torch.stack([p.sum(dim=[d for d in range(n) if d != q])[0] -
                        p.sum(dim=[d for d in range(n) if d != q])[1] for q in range(n)])


def verify_qnn(trials=3):
    torch.manual_seed(0)
    worst = 0.0
    for L in (1, 2, 4):
        c = _Circuit(N_FEAT, L)
        x = torch.randn(trials, N_FEAT) * 2
        fast = c(x).double()
        for b in range(trials):
            worst = max(worst, (fast[b] - _dense_reference(x[b].double(), c.w.detach())).abs().max().item())
        psi = angle_states(x) @ sel_unitary(c.w.detach()).transpose(0, 1)
        full = (psi.real ** 2 + psi.imag ** 2) @ c.zs
        worst = max(worst, (fast - full.double()).abs().max().item())
    x = torch.randn(5, N_FEAT)
    ident = (_Circuit(N_FEAT, 0)(x) - torch.cos(x)).abs().max().item()
    print(f"QNN fast vs gate-by-gate: max err {worst:.2e}   identity vs cos(x): {ident:.2e}")
    assert worst < 1e-4 and ident < 1e-5, 'QNN simulator mismatch'
    from .models import count_params
    for fam in [*QNN_FAMILIES.values(), ['qn/identity', 'qn/random', 'cn/linear', 'cn/ann_nb']]:
        for m in fam:
            print(f"  {m:<14} {count_params(build_feat_model(m))}")
    print('verify_qnn: OK')
