"""Whole-image QCNN architectures from the original paper, on a fast simulator.

    qa/hc        Hybrid Classifier       gpu-run-qcnn-32-32-all-crops.ipynb
    qa/ancilla   Ancilla-Controlled QCNN qcnn-multiple-filters.ipynb
    qa/fqc       Fully Quantum Classifier, first rebuild from the paper text
                 (HC circuit + RY/CNOT classifier, 6 params)
    qa/fqc_sel   FQC with the notebook's classifier (StronglyEntanglingLayers)
                 and the B6 filter        fork-of-gpu-run-qcnn-32-32.ipynb
    qa/fqc_nb    FQC exactly as that notebook: 18-param filter (P_VAL=3),
                 StronglyEntanglingLayers, uniform(0, 0.1) init

All three amplitude-encode the whole 32x32 image into 10 qubits (wire 0 =
most significant bit, as PennyLane) and use the SAME 2-qubit filter as the
Baseline Hybrid: RY-CNOT B6 (12 params, the pipeline's ry_cnot definition;
verify_arch() checks it against QuantumConv). Each conv layer has its own
filter parameters (as in the notebooks); pooling is the notebooks'
ctrl-RZ / X / ctrl-RX / X gate.

Why this is fast: the notebooks ran PennyLane one image at a time. Here the
whole batch is one tensor of shape (G, B, 2, ..., 2) and every gate is one
small matmul. Three exact simplifications:
  * a filter or pool gate on a qubit pair is one 4x4 unitary;
  * all of a model's 4x4 unitaries are built together, once per forward
    pass (building them gate by gate was most of the run time);
  * the ancilla is put in |+> and only ever used as a control, so the state
    is (|0>|psi_0> + |1>|psi_1>)/sqrt2 with each branch evolved by its own
    filter. We simulate the two 10-qubit branches (G=2) and average their
    probabilities -- identical to measuring the data qubits of the
    11-qubit circuit. verify_arch() checks this against a dense,
    gate-by-gate 11-qubit simulation.
"""
import numpy as np
import torch
import torch.nn as nn

N_DATA = 10
POOL1 = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]       # (source, target) of each pool gate
ACTIVE2 = [1, 3, 5, 7, 9]                                # wires left after the first pooling


# ============================================================================
# gate builders (all unitaries of a forward pass at once)
# ============================================================================
_CNOT01 = torch.tensor([[1., 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


def _ry_pair(theta):
    """theta (..., 2) -> RY(theta_0) (x) RY(theta_1), real (..., 4, 4)."""
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)  # (..., qubit, row, col)
    R0, R1 = R[..., 0, :, :], R[..., 1, :, :]
    return torch.einsum('...ij,...mn->...imjn', R0, R1).reshape(*theta.shape[:-1], 4, 4)


def b6_unitaries(theta):
    """theta (K, B, 2) -> (K, 4, 4) complex: K RY-CNOT filters with B blocks
    (RY layer, CNOT, RY layer, ..., RY layer). Real, like RY and CNOT."""
    M = _ry_pair(theta)                                      # (K, B, 4, 4)
    C = _CNOT01.to(theta.device)
    U = M[:, 0]
    for b in range(1, theta.shape[1]):
        U = M[:, b] @ (C @ U)
    return U.to(torch.complex64)


def pool_unitaries(t):
    """t (N, 2) -> (N, 4, 4): ctrl-RZ(t0); X(src); ctrl-RX(t1); X(src) on
    (src, tgt), src = MSB. Equals blockdiag(RX(t1), RZ(t0))."""
    c0, s0 = torch.cos(t[:, 0] / 2), torch.sin(t[:, 0] / 2)
    c1, s1 = torch.cos(t[:, 1] / 2), torch.sin(t[:, 1] / 2)
    z = torch.zeros_like(c0)
    re = torch.stack([c1, z, z, z, z, c1, z, z, z, z, c0, z, z, z, z, c0], -1)
    im = torch.stack([z, -s1, z, z, -s1, z, z, z, z, z, -s0, z, z, z, z, s0], -1)
    return torch.complex(re, im).reshape(-1, 4, 4)


# ============================================================================
# state operations
# ============================================================================
def apply2(psi, U, a, b):
    """Apply 4x4 unitary(s) U to wires (a, b); a is the more significant.
    psi: (G, B, 2, ..., 2) complex. U: (4, 4) or (G, 4, 4)."""
    n = psi.dim() - 2
    x = psi.movedim((a + 2, b + 2), (n, n + 1))
    shape = x.shape
    x = x.reshape(shape[0], -1, 4)
    Ut = U.transpose(-1, -2)
    x = x @ (Ut if Ut.dim() == 3 else Ut.unsqueeze(0))
    return x.reshape(shape).movedim((n, n + 1), (a + 2, b + 2))


def encode(x):
    """(B, 32, 32) images -> (1, B, 2, ..., 2) amplitude-encoded states."""
    v = x.reshape(x.shape[0], -1).abs()
    v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return v.to(torch.complex64).reshape(1, -1, *([2] * N_DATA))


def probs(psi, wires):
    """Marginal probabilities of `wires` (in that order, first = MSB),
    averaged over the branch axis G. -> (B, 2**len(wires))."""
    p = (psi.real ** 2 + psi.imag ** 2).mean(0)               # (B, 2, ..., 2)
    keep = [w + 1 for w in wires]
    drop = [d for d in range(1, p.dim()) if d not in keep]
    p = p.sum(dim=drop) if drop else p
    order = sorted(keep)
    p = p.permute(0, *[1 + order.index(k) for k in keep])
    return p.reshape(p.shape[0], -1)


def expz(psi, wires):
    """<Z> of each wire -> (B, len(wires))."""
    z = torch.tensor([1.0, -1.0], device=psi.device)
    return torch.stack([probs(psi, [w]) @ z for w in wires], dim=1)


def _conv(psi, U, wires):
    """Staggered conv as in the notebooks: pairs (w0,w1),(w2,w3),... then
    (w1,w2),(w3,w4),..."""
    for i in range(0, len(wires) - 1, 2):
        psi = apply2(psi, U, wires[i], wires[i + 1])
    for i in range(1, len(wires) - 1, 2):
        psi = apply2(psi, U, wires[i], wires[i + 1])
    return psi


# ============================================================================
# architectures. Each has .filt (all quantum params) and .fc (classical head)
# so count_params() and the training loop work unchanged.
# ============================================================================
class _QuantumPart(nn.Module):
    """n_filters B6 filters (12 params each, init N(0, 0.1^2) as the
    pipeline's QuantumConv) + n_pool pooling gates (2 angles each)."""

    def __init__(self, n_filters, n_pool):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(n_filters, 6, 2) * 0.1)
        self.pool = nn.Parameter(torch.rand(n_pool, 2) * 0.1)

    def gates(self):
        return b6_unitaries(self.theta), pool_unitaries(self.pool)


class HybridClassifier(nn.Module):
    """3 conv + 3 pool on 10 qubits, <Z> on wires 7, 9 -> MLP 2-8-2.
    gpu-run notebook (its BCE + sigmoid head is written as 2-logit CE)."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.filt = _QuantumPart(3, 8)
        self.fc = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, num_classes))

    def circuit(self, x):
        U, P = self.filt.gates()
        psi = _conv(encode(x), U[0], list(range(N_DATA)))
        for k, (s, t) in enumerate(POOL1):
            psi = apply2(psi, P[k], s, t)
        psi = _conv(psi, U[1], ACTIVE2)
        psi = apply2(psi, P[5], 1, 3)
        psi = apply2(psi, P[6], 5, 7)
        psi = apply2(psi, U[2], 3, 7)
        psi = apply2(psi, U[2], 7, 9)
        return apply2(psi, P[7], 3, 7)

    def forward(self, x):
        return self.fc(expz(self.circuit(x), [7, 9]))


class FullyQuantum(HybridClassifier):
    """HC circuit + 3-layer variational classifier on wires 7, 9 (RY on both,
    then CNOT 7->9); class probabilities = P(wire 9). No classical
    parameters: logits are log-probabilities. Rebuilt from the paper text."""

    def __init__(self, num_classes=2, vqc_layers=3):
        if num_classes != 2:
            raise ValueError('FQC reads one qubit: binary tasks only')
        nn.Module.__init__(self)
        self.filt = _QuantumPart(3, 8)
        self.filt.vqc = nn.Parameter(torch.randn(vqc_layers, 2) * 0.1)
        self.fc = nn.Sequential()                                   # 0 params

    def forward(self, x):
        psi = self.circuit(x)
        C = _CNOT01.to(psi.device)
        V = (C @ _ry_pair(self.filt.vqc)).to(torch.complex64)     # (L, 4, 4), one per layer
        for l in range(V.shape[0]):
            psi = apply2(psi, V[l], 7, 9)
        return torch.log(probs(psi, [9]).clamp_min(1e-9))


class AncillaQCNN(nn.Module):
    """1 ancilla in |+> controls 2 filters per conv layer; 2 conv + 2 pool;
    probabilities of wires 3, 7, 9 -> MLP 8-32-16-2. multiple-filters notebook."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.filt = _QuantumPart(4, 7)            # conv1: filters 0, 1   conv2: 2, 3
        self.fc = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(),
                                nn.Linear(16, num_classes))

    def circuit(self, x):
        U, P = self.filt.gates()
        psi = encode(x)
        psi = torch.cat([psi, psi], 0)                     # one branch per ancilla value
        psi = _conv(psi, U[0:2], list(range(N_DATA)))      # branch g uses filter g
        for k, (s, t) in enumerate(POOL1):
            psi = apply2(psi, P[k], s, t)
        psi = _conv(psi, U[2:4], ACTIVE2)
        psi = apply2(psi, P[5], 1, 3)
        return apply2(psi, P[6], 5, 7)

    def forward(self, x):
        return self.fc(probs(self.circuit(x), [3, 7, 9]))


# ============================================================================
# FQC as in fork-of-gpu-run-qcnn-32-32.ipynb (cell 24): the HC circuit, then
# qml.StronglyEntanglingLayers (3 layers) on wires 7, 9, readout <Z_9>,
# p = (<Z_9> + 1) / 2 with BCE. Our log P(wire 9) logits + CE is the same loss.
# ============================================================================
def nb_filter_unitaries(theta):
    """theta (K, 6*P) -> (K, 4, 4): the notebooks' w_gate_pl, repeated P
    times: RY RY, CNOT, RY RY, CNOT, RY RY (pair[0] = MSB)."""
    K = theta.shape[0]
    M = _ry_pair(theta.reshape(K, -1, 3, 2))                 # (K, P, 3, 4, 4)
    C = _CNOT01.to(theta.device)
    U = torch.eye(4, device=theta.device).expand(K, 4, 4)
    for j in range(M.shape[1]):
        U = M[:, j, 2] @ C @ M[:, j, 1] @ C @ M[:, j, 0] @ U
    return U.to(torch.complex64)


def _rot(w):
    """PennyLane Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi); w (..., 3)."""
    phi, th, om = w[..., 0], w[..., 1], w[..., 2]
    c, s = torch.cos(th / 2), torch.sin(th / 2)
    a, b = (phi + om) / 2, (phi - om) / 2
    e = lambda x: torch.complex(torch.cos(x), torch.sin(x))
    return torch.stack([torch.stack([e(-a) * c, -e(b) * s], -1),
                        torch.stack([e(-b) * s, e(a) * c], -1)], -2)


_CNOT10 = torch.tensor([[1., 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])


def sel_unitaries(w):
    """w (L, 2, 3) -> (L, 4, 4): StronglyEntanglingLayers on 2 wires, one
    unitary per layer: Rot on both wires, CNOT(0->1), CNOT(1->0)."""
    R = _rot(w)                                              # (L, 2, 2, 2)
    RR = torch.einsum('lij,lmn->limjn', R[:, 0], R[:, 1]).reshape(-1, 4, 4)
    C = (_CNOT10 @ _CNOT01).to(torch.complex64).to(w.device)
    return C @ RR


class _FQCPart(nn.Module):
    def __init__(self, filt, init, p_val, vqc_layers):
        super().__init__()
        u = lambda *s: torch.rand(*s) * 0.1                  # notebook: uniform(0, 0.1)
        g = lambda *s: torch.randn(*s) * 0.1                 # pipeline: N(0, 0.1^2)
        draw = u if init == 'nb' else g
        self.kind = filt
        self.theta = nn.Parameter(draw(3, 6, 2) if filt == 'b6' else draw(3, 6 * p_val))
        self.pool = nn.Parameter(torch.rand(8, 2) * 0.1)
        self.vqc = nn.Parameter(draw(vqc_layers, 2, 3))

    def gates(self):
        U = b6_unitaries(self.theta) if self.kind == 'b6' else nb_filter_unitaries(self.theta)
        return U, pool_unitaries(self.pool)


class FQCNotebook(HybridClassifier):
    """filt 'b6' (12 params/filter) or 'nb' (notebook w_gate, P_VAL=3: 18);
    init 'pipeline' or 'nb'. VQC = StronglyEntanglingLayers as the notebook."""

    def __init__(self, num_classes=2, filt='nb', init='nb', p_val=3, vqc_layers=3):
        if num_classes != 2:
            raise ValueError('FQC reads one qubit: binary tasks only')
        nn.Module.__init__(self)
        self.filt = _FQCPart(filt, init, p_val, vqc_layers)
        self.fc = nn.Sequential()

    def forward(self, x):
        psi = self.circuit(x)
        for V in sel_unitaries(self.filt.vqc):
            psi = apply2(psi, V, 7, 9)
        return torch.log(probs(psi, [9]).clamp_min(1e-9))


class _Scale(nn.Module):
    """1 trainable number s: logits (s*z/2, -s*z/2), i.e. P(class 0) = sigmoid(s*z).
    s = 2 at init: sigmoid(2z) ~ (1+z)/2 for small z, the notebook's readout."""

    def __init__(self):
        super().__init__()
        self.s = nn.Parameter(torch.tensor(2.0))

    def forward(self, z):
        return torch.cat([self.s * z / 2, -self.s * z / 2], dim=1)


class FQCScaled(FQCNotebook):
    """Output-range test: the notebook FQC + 1 classical scale on <Z_9>.
    The circuit is created first, so a given seed gives the same circuit
    init as qa/fqc_nb (paired comparison)."""

    def __init__(self, num_classes=2):
        super().__init__(num_classes, filt='nb', init='nb')
        self.fc = _Scale()

    def forward(self, x):
        psi = self.circuit(x)
        for V in sel_unitaries(self.filt.vqc):
            psi = apply2(psi, V, 7, 9)
        return self.fc(expz(psi, [9]))

    def extra(self):
        """Saved with the run (train.run_one): the learned scale."""
        return {'scale_s': float(self.fc.s.detach())}


# ============================================================================
# HC and Ancilla exactly as their notebooks (filter, init, head)
# ============================================================================
class _NBPart(nn.Module):
    """n_filters notebook filters (w_gate_pl, P_VAL=p_val) + n_pool pool
    gates, all angles uniform(0, init_hi) as the notebook."""

    def __init__(self, n_filters, p_val, n_pool, init_hi):
        super().__init__()
        self.theta = nn.Parameter(torch.rand(n_filters, 6 * p_val) * init_hi)
        self.pool = nn.Parameter(torch.rand(n_pool, 2) * init_hi)

    def gates(self):
        return nb_filter_unitaries(self.theta), pool_unitaries(self.pool)


class HCNotebook(HybridClassifier):
    """gpu-run notebook: P_VAL=3 (18 params/filter), uniform(0, 0.1) init,
    head 2-8-1 + sigmoid + BCE, written as logits (0, d): CE == BCE(sigmoid(d))."""

    def __init__(self, num_classes=2):
        if num_classes != 2:
            raise ValueError('notebook HC head is binary')
        nn.Module.__init__(self)
        self.filt = _NBPart(3, 3, 8, 0.1)
        self.fc = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x):
        d = self.fc(expz(self.circuit(x), [7, 9]))
        return torch.cat([torch.zeros_like(d), d], dim=1)


class AncillaNotebook(AncillaQCNN):
    """multiple-filters notebook: P_VAL=2 (12 params/filter), torch.rand
    init (uniform(0, 1)), head 8-32-16-2."""

    def __init__(self, num_classes=2):
        nn.Module.__init__(self)
        self.filt = _NBPart(4, 2, 7, 1.0)
        self.fc = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(),
                                nn.Linear(16, num_classes))


ARCHS = {'hc': HybridClassifier, 'fqc': FullyQuantum, 'ancilla': AncillaQCNN,
         'hc_nb': HCNotebook, 'ancilla_nb': AncillaNotebook,
         'fqc_nb_scale': FQCScaled,
         'fqc_sel': lambda num_classes=2: FQCNotebook(num_classes, filt='b6', init='pipeline'),
         'fqc_nb': lambda num_classes=2: FQCNotebook(num_classes, filt='nb', init='nb')}


def build_arch(tag, num_classes=2, seed=0, **opts):
    """seed is set by the caller (train.run_one) before this is called."""
    used = {k: v for k, v in opts.items() if v not in (None, 'none', False)}
    if used:
        raise ValueError(f"qa/{tag} has no feature map: options {used} do not apply")
    return ARCHS[tag](num_classes=num_classes)


# ============================================================================
# checks: dense, gate-by-gate simulation in numpy (independent code path)
# ============================================================================
def _dense_check(model, x):
    anc = isinstance(model, AncillaQCNN)
    n = N_DATA + (1 if anc else 0)
    I2 = np.eye(2)
    P0, P1 = np.diag([1, 0]), np.diag([0, 1])
    X = np.array([[0, 1], [1, 0]])

    def op(mats):                      # mats: {wire: 2x2}, wire 0 = MSB
        out = np.array([[1.0 + 0j]])
        for w in range(n):
            out = np.kron(out, mats.get(w, I2))
        return out

    def ry(t): c, s = np.cos(t / 2), np.sin(t / 2); return np.array([[c, -s], [s, c]])
    def rz(t): return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])
    def rx(t): c, s = np.cos(t / 2), np.sin(t / 2); return np.array([[c, -1j * s], [-1j * s, c]])

    def b6(theta, a, b, cval=None):
        """theta (B, 2): B6 -- RY layer, CNOT a->b, ..., RY layer.
        theta (6P,): notebook w_gate_pl -- (RY RY, CNOT, RY RY, CNOT, RY RY) x P."""
        U = np.eye(2 ** n, dtype=complex)
        cx = op({a: P0}) + op({a: P1, b: X})
        if theta.ndim == 1:
            for j in range(len(theta) // 6):
                p = theta[6 * j: 6 * j + 6]
                for g in [op({a: ry(p[0]), b: ry(p[1])}), cx, op({a: ry(p[2]), b: ry(p[3])}), cx,
                          op({a: ry(p[4]), b: ry(p[5])})]:
                    U = g @ U
        for blk in range(theta.shape[0] if theta.ndim == 2 else 0):
            U = op({a: ry(theta[blk, 0]), b: ry(theta[blk, 1])}) @ U
            if blk < theta.shape[0] - 1:
                U = cx @ U
        if cval is None:
            return U
        sel = op({N_DATA: P1 if cval else P0})            # controlled on the ancilla
        return sel @ U + (np.eye(2 ** n) - sel)

    def pool(t0, t1, s, t):            # gates applied: ctrl-RZ, X, ctrl-RX, X
        crz = op({s: P0}) + op({s: P1, t: rz(t0)})
        crx = op({s: P0}) + op({s: P1, t: rx(t1)})
        return op({s: X}) @ crx @ op({s: X}) @ crz

    def conv(U_of, wires):
        out = np.eye(2 ** n, dtype=complex)
        pairs = [(wires[i], wires[i + 1]) for i in range(0, len(wires) - 1, 2)] + \
                [(wires[i], wires[i + 1]) for i in range(1, len(wires) - 1, 2)]
        for a, b in pairs:
            out = U_of(a, b) @ out
        return out

    th = model.filt.theta.detach().cpu().numpy()
    pl = model.filt.pool.detach().cpu().numpy()
    v = x.reshape(-1).abs().cpu().numpy().astype(complex)
    v = v / np.linalg.norm(v)

    if anc:
        psi = np.kron(v, np.array([1, 1]) / np.sqrt(2))
        psi = conv(lambda a, b: b6(th[1], a, b, 1) @ b6(th[0], a, b, 0), list(range(10))) @ psi
        for k, (s, t) in enumerate(POOL1):
            psi = pool(*pl[k], s, t) @ psi
        psi = conv(lambda a, b: b6(th[3], a, b, 1) @ b6(th[2], a, b, 0), ACTIVE2) @ psi
        psi = pool(*pl[5], 1, 3) @ psi
        psi = pool(*pl[6], 5, 7) @ psi
        p = np.abs(psi.reshape([2] * n)) ** 2
        return p.sum(axis=(0, 1, 2, 4, 5, 6, 8, 10)).reshape(-1)

    psi = conv(lambda a, b: b6(th[0], a, b), list(range(10))) @ v
    for k, (s, t) in enumerate(POOL1):
        psi = pool(*pl[k], s, t) @ psi
    psi = conv(lambda a, b: b6(th[1], a, b), ACTIVE2) @ psi
    psi = pool(*pl[5], 1, 3) @ psi
    psi = pool(*pl[6], 5, 7) @ psi
    psi = b6(th[2], 3, 7) @ psi
    psi = b6(th[2], 7, 9) @ psi
    psi = pool(*pl[7], 3, 7) @ psi
    p = np.abs(psi.reshape([2] * n)) ** 2
    pz = lambda w: p.sum(axis=tuple(d for d in range(n) if d != w))
    return np.array([pz(7) @ [1, -1], pz(9) @ [1, -1]])


def verify_arch(seed=0, tol=1e-4):
    """(1) b6_unitaries == the pipeline's QuantumConv ry_cnot_B6;
    (2) fast simulator == dense gate-by-gate simulation, HC and Ancilla."""
    from .models import QuantumConv
    torch.manual_seed(seed)
    ok = True
    qc = QuantumConv('ry_cnot', n_blocks=6)
    with torch.no_grad():
        qc.theta.uniform_(-3, 3)
        ref = qc.unitary_apply(torch.eye(4, dtype=torch.complex64)).transpose(0, 1)
        err = (b6_unitaries(qc.theta.unsqueeze(0))[0] - ref).abs().max().item()
    ok &= err < tol
    print(f"B6 filter vs pipeline QuantumConv: max err {err:.2e}")

    x = torch.rand(2, 32, 32)
    hc_feat = lambda m, x: expz(m.circuit(x), [7, 9])
    anc_feat = lambda m, x: probs(m.circuit(x), [3, 7, 9])
    for cls, feat in [(HybridClassifier, hc_feat), (AncillaQCNN, anc_feat),
                      (HCNotebook, hc_feat), (AncillaNotebook, anc_feat)]:
        m = cls()
        with torch.no_grad():
            m.filt.pool.uniform_(-3, 3)                    # non-trivial angles
            m.filt.theta.uniform_(-3, 3)
            fast = feat(m, x).numpy()
        for i in range(x.shape[0]):
            err = np.abs(fast[i] - _dense_check(m, x[i])).max()
            ok &= err < tol
            print(f"{cls.__name__:<18} image {i}: max |fast - dense| = {err:.2e}")
    ok &= verify_fqc_pieces(tol)
    print('verify_arch:', 'OK' if ok else 'MISMATCH')
    return ok


def verify_fqc_pieces(tol=1e-5):
    """Notebook filter and StronglyEntanglingLayers vs gate-by-gate numpy,
    using PennyLane's gate definitions (wire order: first wire = MSB)."""
    I2 = np.eye(2)
    ry = lambda t: np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]])
    rz = lambda t: np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])
    cnot01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    cnot10 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    on0 = lambda g: np.kron(g, I2)
    on1 = lambda g: np.kron(I2, g)
    rng = np.random.default_rng(0)

    p = rng.uniform(-3, 3, 18)                                 # w_gate_pl, P_VAL = 3
    U = np.eye(4, dtype=complex)
    for j in range(3):
        for g in [on0(ry(p[6 * j])), on1(ry(p[6 * j + 1])), cnot01,
                  on0(ry(p[6 * j + 2])), on1(ry(p[6 * j + 3])), cnot01,
                  on0(ry(p[6 * j + 4])), on1(ry(p[6 * j + 5]))]:
            U = g @ U
    e1 = np.abs(nb_filter_unitaries(torch.tensor(p, dtype=torch.float32)[None])[0].numpy() - U).max()

    w = rng.uniform(-3, 3, (3, 2, 3))                          # StronglyEntanglingLayers
    V = np.eye(4, dtype=complex)
    for l in range(3):
        rot = lambda a: rz(a[2]) @ ry(a[1]) @ rz(a[0])
        for g in [on0(rot(w[l, 0])), on1(rot(w[l, 1])), cnot01, cnot10]:
            V = g @ V
    S = sel_unitaries(torch.tensor(w, dtype=torch.float32))
    Vt = S[2] @ S[1] @ S[0]
    e2 = np.abs(Vt.numpy() - V).max()
    print(f"notebook filter: max err {e1:.2e}   StronglyEntanglingLayers: max err {e2:.2e}")
    return e1 < tol and e2 < tol
