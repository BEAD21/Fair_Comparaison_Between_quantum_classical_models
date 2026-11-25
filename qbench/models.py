"""Every model in the benchmark, built from one name.

All models share one skeleton, so only the filter differs:

    x (B,32,32) -> [encoding] -> filter -> [norm] -> [act] -> MaxPool2
                -> flatten (2 x 8 x 8 = 128) -> Linear(128, C)   head = 258 for C=2

Names
    q/<ansatz>             quantum filter, e.g. q/ry_cnot_B6, q/hea_L2, q/identity
    c/<filter>/<encoding>  classical filter, encoding 'raw' or 'amp',
                           e.g. c/Conv2x2/raw, c/Separable3x3/amp

Quantum ansatz tags (same tags as the earlier CSVs):
    identity, random, qaoa_p<P>, hea_L<L>, ry_only_L<L>, ry_cnot_B<B>, wgate_P<P>,
    su4 (general 2-qubit gate, 15 params)

Recipe options (keyword overrides to build_model), defaults reproduce the
earlier notebooks exactly:
    act    'none' | 'relu' | 'leaky' | 'tanh'   quantum default 'none', classical 'relu'
    norm   'none' | 'bn'    'bn' = BatchNorm2d(affine=False): 0 trainable params
    frozen True -> filter weights not trained (random-feature baseline)
"""
import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# encoding
# ============================================================================
class AmplitudeEncode(nn.Module):
    """abs() + per-patch L2 normalisation -- exactly the quantum encoding.

    Non-overlapping patches, so fold() is an exact inverse of unfold().
    Parameter-free.
    """

    def __init__(self, patch=2):
        super().__init__()
        self.patch = patch

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.patch
        p = F.unfold(x, kernel_size=k, stride=k)
        p = torch.abs(p)
        p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return F.fold(p, output_size=(H, W), kernel_size=k, stride=k)


# ============================================================================
# quantum filter (analytic statevector simulation)
# ============================================================================
def _cnot_matrix(n, pairs):
    """Dense matrix for the CNOTs in `pairs` (control, target), applied in
    order. Wire 0 is the most significant bit."""
    dim = 2 ** n
    out = torch.eye(dim, dtype=torch.complex64)
    for c, t in pairs:
        m = torch.zeros(dim, dim, dtype=torch.complex64)
        for k in range(dim):
            bits = [(k >> (n - 1 - w)) & 1 for w in range(n)]
            if bits[c] == 1:
                bits[t] ^= 1
            k2 = 0
            for w in range(n):
                k2 = (k2 << 1) | bits[w]
            m[k2, k] = 1
        out = m @ out
    return out


def _ry(t):
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(torch.complex64)


def _rz(t):
    z = torch.zeros((), dtype=t.dtype, device=t.device)
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    m = torch.stack([torch.stack([c, z]), torch.stack([z, c])]).to(torch.complex64)
    i = torch.stack([torch.stack([-s, z]), torch.stack([z, s])]).to(torch.complex64)
    return m + 1j * i


def _rx(t):
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    z = torch.zeros((), dtype=c.dtype, device=c.device)
    m = torch.stack([torch.stack([c, z]), torch.stack([z, c])]).to(torch.complex64)
    i = torch.stack([torch.stack([z, -s]), torch.stack([-s, z])]).to(torch.complex64)
    return m + 1j * i


class QuantumConv(nn.Module):
    """Quantum conv layer with a selectable ansatz.

    Every ansatz shares the same encoding (abs, then L2-normalise each
    kernel_size x kernel_size patch into a 2^n amplitude vector) and the same
    readout (<Z_q> on each wire), so only the unitary differs.
    """

    def __init__(self, ansatz, kernel_size=2, n_qubits=2, stride=2,
                 p=6, n_layers=2, n_blocks=3, seed=0):
        super().__init__()
        self.kernel_size, self.n_qubits, self.stride = kernel_size, n_qubits, stride
        self.ansatz, self.p, self.n_layers, self.n_blocks = ansatz, p, n_layers, n_blocks
        n, dim = n_qubits, 2 ** n_qubits

        idx = torch.arange(dim)
        shifts = torch.arange(n - 1, -1, -1)
        bits = (idx.unsqueeze(1) >> shifts.unsqueeze(0)) & 1
        sgn = (1 - 2 * bits).float()
        self.register_buffer('sgn', sgn)
        self.register_buffer('zz', (sgn[:, :-1] * sgn[:, 1:]).sum(dim=1))

        if ansatz == 'qaoa':
            self.betas = nn.Parameter(torch.randn(p) * 0.1)
            self.gammas = nn.Parameter(torch.randn(p) * 0.1)
        elif ansatz == 'hea':
            self.theta = nn.Parameter(torch.randn(n_layers, n, 2) * 0.1)
            self.register_buffer('ring', _cnot_matrix(n, [(c, (c + 1) % n) for c in range(n)]))
        elif ansatz == 'ry_only':
            self.theta = nn.Parameter(torch.randn(n_layers, n) * 0.1)
            self.register_buffer('ring', _cnot_matrix(n, [(c, (c + 1) % n) for c in range(n)]))
        elif ansatz == 'ry_cnot':
            # RY on every wire, CNOT chain, RY, CNOT chain, ..., RY
            self.theta = nn.Parameter(torch.randn(n_blocks, n) * 0.1)
            self.register_buffer('cnot_chain', _cnot_matrix(n, [(c, c + 1) for c in range(n - 1)]))
        elif ansatz == 'wgate':
            # the notebooks' w_gate_pl x P: (RY, CNOT chain, RY, CNOT chain, RY) x P, 6P params on 2 wires
            self.theta = nn.Parameter(torch.randn(n_blocks * 3, n) * 0.1)
            self.register_buffer('cnot_chain', _cnot_matrix(n, [(c, c + 1) for c in range(n - 1)]))
        elif ansatz == 'su4':
            # general 2-qubit gate, KAK form (15 params):
            #   U = (A1 x A2) exp(i(a XX + b YY + c ZZ)) (A3 x A4),  A = RZ RY RZ
            if n != 2:
                raise ValueError('su4 is defined for 2 qubits only')
            self.theta = nn.Parameter(torch.randn(4, 3) * 0.1)     # A1..A4, 3 angles each
            self.kak = nn.Parameter(torch.randn(3) * 0.1)          # a, b, c
            X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
            Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
            Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
            self.register_buffer('xyz', torch.stack([torch.kron(P, P) for P in (X, Y, Z)]))
        elif ansatz == 'random':
            # fixed Haar-random unitary, frozen: 0 parameters
            g = torch.Generator().manual_seed(seed)
            a = torch.randn(dim, dim, generator=g)
            b = torch.randn(dim, dim, generator=g)
            q, r = torch.linalg.qr(torch.complex(a, b))
            q = q * (torch.diagonal(r) / torch.diagonal(r).abs()).unsqueeze(0)
            self.register_buffer('U', q)
        elif ansatz == 'identity':
            pass  # no circuit: measure the encoded state directly
        else:
            raise ValueError(f"unknown ansatz: {ansatz}")

    def unitary_apply(self, s):
        """s: (N, 2^n) complex states as rows. Returns the evolved states."""
        n = self.n_qubits
        if self.ansatz == 'identity':
            return s
        if self.ansatz == 'random':
            return s @ self.U.transpose(0, 1)
        if self.ansatz == 'qaoa':
            for j in range(self.p):
                s = s * torch.exp(1j * (self.gammas[j] / 2) * self.zz).unsqueeze(0)
                r = _rx(2 * self.betas[j])
                m = r
                for _ in range(n - 1):
                    m = torch.kron(m, r)
                s = s @ m.transpose(0, 1)
            return s
        if self.ansatz == 'ry_cnot':
            for b in range(self.n_blocks):
                m = None
                for q in range(n):
                    g = _ry(self.theta[b, q])
                    m = g if m is None else torch.kron(m, g)
                s = s @ m.transpose(0, 1)
                if b < self.n_blocks - 1:
                    s = s @ self.cnot_chain.transpose(0, 1)
            return s
        if self.ansatz == 'su4':
            def local(i):                                   # RZ RY RZ on one wire
                t = self.theta[i]
                return _rz(t[2]) @ _ry(t[1]) @ _rz(t[0])
            core = torch.linalg.matrix_exp(1j * (self.kak.to(torch.complex64)[:, None, None] * self.xyz).sum(0))
            U = torch.kron(local(0), local(1)) @ core @ torch.kron(local(2), local(3))
            return s @ U.transpose(0, 1)
        if self.ansatz == 'wgate':
            for b in range(self.n_blocks * 3):
                m = None
                for q in range(n):
                    g = _ry(self.theta[b, q])
                    m = g if m is None else torch.kron(m, g)
                s = s @ m.transpose(0, 1)
                if b % 3 < 2:
                    s = s @ self.cnot_chain.transpose(0, 1)
            return s
        # hea / ry_only: per-wire rotations, then a CNOT ring, per layer
        for l in range(self.n_layers):
            m = None
            for q in range(n):
                if self.ansatz == 'hea':
                    g = _rz(self.theta[l, q, 1]) @ _ry(self.theta[l, q, 0])
                else:
                    g = _ry(self.theta[l, q])
                m = g if m is None else torch.kron(m, g)
            s = s @ m.transpose(0, 1)
            s = s @ self.ring.transpose(0, 1)
        return s

    def forward(self, x):
        B, _, H, W = x.shape
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        L = patches.shape[2]
        patches = patches.permute(0, 2, 1).reshape(B * L, -1)

        mask = torch.norm(patches, p=2, dim=1) > 0
        feats = torch.zeros(B * L, self.n_qubits, device=x.device, dtype=x.dtype)
        valid = patches[mask]
        if valid.shape[0] > 0:
            valid = F.normalize(torch.abs(valid), p=2, dim=1)
            s = self.unitary_apply(valid.to(torch.complex64))
            probs = s.real ** 2 + s.imag ** 2
            feats[mask] = (probs @ self.sgn).to(x.dtype)

        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        return feats.view(B, L, self.n_qubits).permute(0, 2, 1).reshape(
            B, self.n_qubits, H_out, W_out)


# ============================================================================
# classical filters  (all: 1 -> 2 channels, stride 2, 32x32 -> 16x16)
# ============================================================================
class Conv2x2(nn.Module):
    """2 dense 2x2 kernels + bias = 10 params. Same 2x2 patches as the circuit."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 2, stride=2, bias=True)

    def forward(self, x):
        return self.conv(x)


class PatchMLP(nn.Module):
    """Strict counterpart of a 12-parameter two-qubit filter (added 2026-11-10):
    the same non-overlapping 2x2 patches, two outputs per patch, exactly 12
    params. Per patch: 4 -> 2 linear (8), tanh, 2 -> 2 linear (4), no biases.
    Like the circuit (amplitude encoding then <Z>), each output is a nonlinear
    function of the 4 pixels of its own patch only."""

    def __init__(self):
        super().__init__()
        self.inner = nn.Conv2d(1, 2, 2, stride=2, bias=False)
        self.outer = nn.Conv2d(2, 2, 1, bias=False)

    def forward(self, x):
        return self.outer(torch.tanh(self.inner(x)))


class Separable3x3(nn.Module):
    """2 rank-1 3x3 kernels (K = v h^T) = 12 params."""

    def __init__(self):
        super().__init__()
        self.v = nn.Parameter(torch.randn(2, 3) * 0.5)
        self.h = nn.Parameter(torch.randn(2, 3) * 0.5)

    def forward(self, x):
        k = torch.einsum('oi,oj->oij', self.v, self.h).unsqueeze(1)
        return F.conv2d(x, k, stride=2, padding=1)


class FixedBasis6(nn.Module):
    """6 frozen 3x3 DCT bases, 2x6 learned mixing coefficients = 12 params."""

    def __init__(self, n_basis=6, k=3):
        super().__init__()
        self.k = k
        self.register_buffer('basis', self._dct(n_basis, k))
        self.coef = nn.Parameter(torch.randn(2, n_basis) * 0.5)

    @staticmethod
    def _dct(n, k):
        mats = []
        for u in range(k):
            for v in range(k):
                m = torch.zeros(k, k)
                cu = math.sqrt(1 / k) if u == 0 else math.sqrt(2 / k)
                cv = math.sqrt(1 / k) if v == 0 else math.sqrt(2 / k)
                for i in range(k):
                    for j in range(k):
                        m[i, j] = (cu * cv
                                   * math.cos((2 * i + 1) * u * math.pi / (2 * k))
                                   * math.cos((2 * j + 1) * v * math.pi / (2 * k)))
                mats.append(m)
        return torch.stack(mats[:n])

    def forward(self, x):
        k = torch.einsum('on,nij->oij', self.coef, self.basis).unsqueeze(1)
        return F.conv2d(x, k, stride=2, padding=(self.k - 1) // 2)


class DWSep3x3(nn.Module):
    """Depthwise 3x3 (9 + 1 bias) + pointwise 1->2 (2) = 12 params."""

    def __init__(self):
        super().__init__()
        self.dw = nn.Conv2d(1, 1, 3, stride=2, padding=1, bias=True)
        self.pw = nn.Conv2d(1, 2, 1, bias=False)

    def forward(self, x):
        return self.pw(self.dw(x))


class PaperTwin(nn.Module):
    """2 dense 3x3 kernels + bias = 20 params -- the submitted paper's twin."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, stride=2, padding=1, bias=True)

    def forward(self, x):
        return self.conv(x)


class Pool2(nn.Module):
    """No learned filter (0 params): 2x2 average and 2x2 max as the 2 channels.

    The classical counterpart of q/identity -- tells you what the head alone
    gets from the (raw or amplitude-encoded) input.
    """

    def forward(self, x):
        return torch.cat([F.avg_pool2d(x, 2), F.max_pool2d(x, 2)], dim=1)


FILTERS = {
    'Conv2x2': Conv2x2,            # 10
    'PatchMLP': PatchMLP,          # 12  same 2x2 patches as the circuit
    'Separable3x3': Separable3x3,  # 12
    'FixedBasis6': FixedBasis6,    # 12
    'DWSep3x3': DWSep3x3,          # 12
    'PaperTwin': PaperTwin,        # 20  reference only
    'Pool2': Pool2,                # 0   no-filter baseline
}


# ============================================================================
# the shared skeleton
# ============================================================================
ACTS = {
    'none': nn.Identity,
    'relu': nn.ReLU,
    'leaky': lambda: nn.LeakyReLU(0.1),
    'tanh': nn.Tanh,
}


class Net(nn.Module):
    """[encoding] -> filter -> [norm] -> [act] -> MaxPool2 -> Linear."""

    def __init__(self, filt, num_classes=2, img_size=32, encoding='raw',
                 act='relu', norm='none', frozen=False):
        super().__init__()
        self.enc = AmplitudeEncode(2) if encoding == 'amp' else nn.Identity()
        self.filt = filt
        if frozen:
            for prm in self.filt.parameters():
                prm.requires_grad_(False)
        # Exactly one randn probe between filter init and head init: the old
        # notebooks drew the same one, so a given seed gives the same weights.
        with torch.no_grad():
            z = self.filt(self.enc(torch.randn(1, 1, img_size, img_size)))
        n_ch = z.shape[1]
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(n_ch, affine=False)
        elif norm == 'none':
            self.norm = nn.Identity()
        else:
            raise ValueError(f"unknown norm: {norm}")
        self.act = ACTS[act]()
        # big filters (16x16 patches) leave a 2x2 map; never pool below 1x1
        self.pool = nn.MaxPool2d(2) if min(z.shape[-2:]) >= 2 else nn.Identity()
        n_feat = self.pool(z).numel()
        self.fc = nn.Linear(n_feat, num_classes)

    def features(self, x):
        x = self.enc(x.unsqueeze(1))
        x = self.pool(self.act(self.norm(self.filt(x))))
        return x.reshape(x.size(0), -1)

    def forward(self, x):
        return self.fc(self.features(x))


# ============================================================================
# registry
# ============================================================================
_QTAG = re.compile(r'^(qaoa|hea|ry_only|ry_cnot|wgate)_([pLBP])(\d+)(?:_k(\d+))?$')


def _parse_ansatz(tag):
    """'ry_cnot_B6' -> 2x2 patches, 2 qubits. Optional '_k<K>' (filter-size
    ablation): KxK patches, stride K, log2(K*K) qubits, e.g. ry_cnot_B6_k4."""
    if tag in ('identity', 'random', 'su4'):
        return tag, {}
    m = _QTAG.match(tag)
    if not m:
        raise ValueError(f"bad ansatz tag '{tag}' (e.g. ry_cnot_B6, hea_L2, qaoa_p6, ry_cnot_B6_k4)")
    name, letter, val = m.group(1), m.group(2), int(m.group(3))
    key = {'p': 'p', 'L': 'n_layers', 'B': 'n_blocks', 'P': 'n_blocks'}[letter]
    expected = {'qaoa': 'p', 'hea': 'L', 'ry_only': 'L', 'ry_cnot': 'B', 'wgate': 'P'}[name]
    if letter != expected:
        raise ValueError(f"'{name}' takes _{expected}<n>, got '{tag}'")
    kw = {key: val}
    if m.group(4):
        k = int(m.group(4))
        n = int(round(math.log2(k * k)))
        if 2 ** n != k * k or k > 16:
            raise ValueError(f"filter size k must be 2, 4, 8 or 16, got {k}")
        kw.update(kernel_size=k, stride=k, n_qubits=n)
    return name, kw


def build_model(name, num_classes=2, img_size=32, seed=0, **opts):
    """Build any benchmark model from its name. See module docstring."""
    parts = name.split('/')
    if parts[0] == 'rn':
        # ResNet-50 head on cached features (qbench/router.py); needs RouterData
        from .router import build_router_model
        return build_router_model(name, num_classes)
    if parts[0] in ('qn', 'cn') and len(parts) == 2:
        # QNN / ANN on the 8 PCA features (qbench/qnn.py); needs FeatureData
        if opts:
            raise ValueError(f"{name} takes no model options, got {opts}")
        from .qnn import build_feat_model
        return build_feat_model(name, num_classes, seed=seed)
    if parts[0] == 'qa' and len(parts) == 2:
        # whole-image architectures (qbench/arch.py): no encoding/norm/act options
        from .arch import build_arch
        return build_arch(parts[1], num_classes, seed=seed, **opts)
    if parts[0] == 'q' and len(parts) == 2:
        ansatz, kw = _parse_ansatz(parts[1])
        filt = QuantumConv(ansatz, seed=seed, **kw)
        opts.setdefault('act', 'none')
        # the circuit does its own amplitude encoding
        return Net(filt, num_classes, img_size, encoding='raw', **opts)
    if parts[0] == 'c' and len(parts) == 3:
        fname, enc = parts[1], parts[2]
        if fname not in FILTERS:
            raise ValueError(f"unknown filter '{fname}', choose from {list(FILTERS)}")
        if enc not in ('raw', 'amp'):
            raise ValueError(f"encoding must be raw or amp, got '{enc}'")
        opts.setdefault('act', 'relu')
        return Net(FILTERS[fname](), num_classes, img_size, encoding=enc, **opts)
    raise ValueError(f"bad model name '{name}' (e.g. q/ry_cnot_B6, c/Conv2x2/amp)")


def count_params(model):
    """Trainable parameters, split into filter and head."""
    f = sum(p.numel() for p in model.filt.parameters() if p.requires_grad)
    h = sum(p.numel() for p in model.fc.parameters() if p.requires_grad)
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert t == f + h, "parameters outside filter/head"
    return {'filter_params': f, 'head_params': h, 'total_params': t}


# ============================================================================
# checks
# ============================================================================
QUANTUM_TAGS = ['identity', 'random', 'ry_only_L2', 'hea_L2', 'hea_L3',
                'ry_cnot_B3', 'ry_cnot_B6', 'qaoa_p6', 'qaoa_p12']
CLASSICAL_FILTERS = list(FILTERS)


def audit(num_classes=2):
    """Print filter/head/total params for every model; assert identical heads."""
    print(f"{'model':<26}{'filter':>8}{'head':>7}{'total':>7}")
    print('-' * 48)
    names = [f'q/{t}' for t in QUANTUM_TAGS] + \
            [f'c/{f}/{e}' for f in CLASSICAL_FILTERS for e in ('raw', 'amp')]
    heads = set()
    x = torch.rand(3, 32, 32)
    for nm in names:
        m = build_model(nm, num_classes)
        c = count_params(m)
        with torch.no_grad():
            assert m(x).shape == (3, num_classes), nm
        heads.add(c['head_params'])
        print(f"{nm:<26}{c['filter_params']:>8}{c['head_params']:>7}{c['total_params']:>7}")
    assert len(heads) == 1, f"heads differ: {heads}"
    print('-' * 48)
    print(f"all heads identical: {heads.pop()} params")


def verify_quantum(n_draws=50, tol=1e-5):
    """Analytic layer vs a literal gate-by-gate numpy simulation (2 qubits).

    Checks ry_cnot (B3, B6), hea (L2, L3) and ry_only (L2) and asserts
    max deviation < tol.
    """
    def RY(t):
        return np.array([[np.cos(t / 2), -np.sin(t / 2)],
                         [np.sin(t / 2), np.cos(t / 2)]], dtype=complex)

    def RZ(t):
        return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]])

    CNOT01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    CNOT10 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)

    def literal(ansatz, th):
        U = np.eye(4, dtype=complex)
        if ansatz == 'ry_cnot':
            for b in range(th.shape[0]):
                U = np.kron(RY(th[b, 0]), RY(th[b, 1])) @ U
                if b < th.shape[0] - 1:
                    U = CNOT01 @ U
        else:
            for l in range(th.shape[0]):
                if ansatz == 'hea':
                    g0 = RZ(th[l, 0, 1]) @ RY(th[l, 0, 0])
                    g1 = RZ(th[l, 1, 1]) @ RY(th[l, 1, 0])
                else:
                    g0, g1 = RY(th[l, 0]), RY(th[l, 1])
                U = CNOT10 @ CNOT01 @ np.kron(g0, g1) @ U
        return U

    rng = np.random.default_rng(0)
    for tag in ['ry_cnot_B3', 'ry_cnot_B6', 'hea_L2', 'hea_L3', 'ry_only_L2']:
        ansatz, kw = _parse_ansatz(tag)
        layer = QuantumConv(ansatz, **kw)
        worst = 0.0
        for _ in range(n_draws):
            th = rng.normal(0, np.pi, tuple(layer.theta.shape))
            with torch.no_grad():
                layer.theta.copy_(torch.tensor(th, dtype=torch.float32))
            psi = np.abs(rng.normal(size=(8, 4)))
            psi /= np.linalg.norm(psi, axis=1, keepdims=True)
            pr = np.abs(psi.astype(complex) @ literal(ansatz, th).T) ** 2
            ref = np.stack([pr[:, 0] + pr[:, 1] - pr[:, 2] - pr[:, 3],
                            pr[:, 0] - pr[:, 1] + pr[:, 2] - pr[:, 3]], axis=1)
            with torch.no_grad():
                s = layer.unitary_apply(torch.tensor(psi, dtype=torch.complex64))
                got = ((s.real ** 2 + s.imag ** 2) @ layer.sgn).numpy()
            worst = max(worst, float(np.abs(ref - got).max()))
        status = 'PASS' if worst < tol else 'FAIL'
        print(f"  {tag:<12} max deviation {worst:.2e}  {status}")
        assert worst < tol, tag
