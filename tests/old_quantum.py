import torch
import torch.nn as nn
import torch.nn.functional as F
# ############################################################################
# CELL G --- MULTI-ANSATZ QUANTUM CONV LAYER
# ############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlexibleQuantumConvLayer(nn.Module):
    """Analytic quantum conv layer with a selectable ansatz.

    All variants share the same amplitude encoding (|abs| then L2-normalize
    per patch) and the same readout (<Z_q> on each wire), so the ONLY thing
    that changes between them is the unitary in the middle. That is what
    isolates the filter's contribution.
    """

    def __init__(self, kernel_size, n_qubits, stride,
                 ansatz='qaoa', p=6, n_layers=2, n_blocks=3, seed=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_qubits = n_qubits
        self.stride = stride
        self.ansatz = ansatz
        self.p = p
        self.n_layers = n_layers
        self.n_blocks = n_blocks
        dim = 2 ** n_qubits

        # ---- shared: Z-eigenvalue table and the ZZ-chain phase pattern
        idx = torch.arange(dim)
        shifts = torch.arange(n_qubits - 1, -1, -1)
        bits = (idx.unsqueeze(1) >> shifts.unsqueeze(0)) & 1
        sgn = (1 - 2 * bits).float()
        self.register_buffer("sgn", sgn)
        self.register_buffer("zz", (sgn[:, :-1] * sgn[:, 1:]).sum(dim=1))

        if ansatz == 'qaoa':
            self.betas = nn.Parameter(torch.randn(p) * 0.1)
            self.gammas = nn.Parameter(torch.randn(p) * 0.1)
            self.n_trainable = 2 * p

        elif ansatz == 'hea':
            self.theta = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)
            self.register_buffer("ring", self._cnot_ring(n_qubits))
            self.n_trainable = 2 * n_qubits * n_layers

        elif ansatz == 'ry_only':
            self.theta = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
            self.register_buffer("ring", self._cnot_ring(n_qubits))
            self.n_trainable = n_qubits * n_layers

        elif ansatz == 'ry_cnot':
            # The drawn circuit: RY on both wires, CNOT, RY, CNOT, RY.
            # n_blocks rotation layers with (n_blocks - 1) CNOTs between them.
            # n_blocks=3, n_qubits=2 -> 6 parameters (the figure).
            self.n_blocks = n_blocks
            self.theta = nn.Parameter(torch.randn(n_blocks, n_qubits) * 0.1)
            self.register_buffer("cnot_chain", self._cnot_chain(n_qubits))
            self.n_trainable = n_blocks * n_qubits

        elif ansatz == 'random':
            # A fixed Haar-random unitary. Frozen: contributes 0 parameters.
            g = torch.Generator().manual_seed(seed)
            a = torch.randn(dim, dim, generator=g)
            b = torch.randn(dim, dim, generator=g)
            q, r = torch.linalg.qr(torch.complex(a, b))
            q = q * (torch.diagonal(r) / torch.diagonal(r).abs()).unsqueeze(0)
            self.register_buffer("U", q)
            self.n_trainable = 0

        elif ansatz == 'identity':
            # No circuit. Measures the amplitude encoding directly.
            self.n_trainable = 0

        else:
            raise ValueError(f"unknown ansatz: {ansatz}")

    @staticmethod
    def _cnot_ring(n):
        """Dense matrix for CNOT(q, (q+1) mod n) applied for every q."""
        dim = 2 ** n
        perm = torch.arange(dim)
        out = torch.eye(dim, dtype=torch.complex64)
        for c in range(n):
            t = (c + 1) % n
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

    @staticmethod
    def _cnot_chain(n):
        """CNOT(q, q+1) along the chain -- for n=2 this is the single CNOT
        drawn in the figure (control = upper wire, target = lower wire)."""
        dim = 2 ** n
        out = torch.eye(dim, dtype=torch.complex64)
        for c in range(n - 1):
            t = c + 1
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

    @staticmethod
    def _ry(t):
        c, s = torch.cos(t / 2), torch.sin(t / 2)
        z = torch.zeros((), dtype=c.dtype, device=c.device)
        return torch.stack([torch.stack([c, -s]),
                            torch.stack([s, c])]).to(torch.complex64)

    @staticmethod
    def _rz(t):
        z = torch.zeros((), dtype=t.dtype, device=t.device)
        c, s = torch.cos(t / 2), torch.sin(t / 2)
        m = torch.stack([torch.stack([c, z]), torch.stack([z, c])]).to(torch.complex64)
        i = torch.stack([torch.stack([-s, z]), torch.stack([z, s])]).to(torch.complex64)
        return m + 1j * i

    @staticmethod
    def _rx(t):
        c, s = torch.cos(t / 2), torch.sin(t / 2)
        z = torch.zeros((), dtype=c.dtype, device=c.device)
        m = torch.stack([torch.stack([c, z]), torch.stack([z, c])]).to(torch.complex64)
        i = torch.stack([torch.stack([z, -s]), torch.stack([-s, z])]).to(torch.complex64)
        return m + 1j * i

    def _apply_ansatz(self, s):
        """s: (N, dim) complex. Applies the selected ansatz."""
        n = self.n_qubits

        if self.ansatz == 'identity':
            return s

        if self.ansatz == 'random':
            return s @ self.U.transpose(0, 1)

        if self.ansatz == 'qaoa':
            for j in range(self.p):
                phase = torch.exp(1j * (self.gammas[j] / 2) * self.zz.to(torch.float32))
                s = s * phase.unsqueeze(0)
                r = self._rx(2 * self.betas[j])
                m = r
                for _ in range(n - 1):
                    m = torch.kron(m, r)
                s = s @ m.transpose(0, 1)
            return s

        if self.ansatz == 'ry_cnot':
            # RY block, CNOT, RY block, CNOT, RY block  (as drawn)
            for b in range(self.n_blocks):
                m = None
                for q in range(n):
                    g = self._ry(self.theta[b, q])
                    m = g if m is None else torch.kron(m, g)
                s = s @ m.transpose(0, 1)
                if b < self.n_blocks - 1:
                    s = s @ self.cnot_chain.transpose(0, 1)
            return s

        # hea / ry_only: per-wire rotations then a CNOT ring
        for l in range(self.n_layers):
            m = None
            for q in range(n):
                if self.ansatz == 'hea':
                    g = self._rz(self.theta[l, q, 1]) @ self._ry(self.theta[l, q, 0])
                else:
                    g = self._ry(self.theta[l, q])
                m = g if m is None else torch.kron(m, g)
            s = s @ m.transpose(0, 1)
            s = s @ self.ring.transpose(0, 1)
        return s

    def forward(self, x):
        batch_size, _, H_in, W_in = x.shape
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        num_patches = patches.shape[2]
        patches = patches.permute(0, 2, 1).reshape(batch_size * num_patches, -1)

        norms = torch.norm(patches, p=2, dim=1)
        mask = norms > 0

        features = torch.zeros(batch_size * num_patches, self.n_qubits,
                               device=x.device, dtype=x.dtype)

        valid = patches[mask]
        if valid.shape[0] > 0:
            valid = F.normalize(torch.abs(valid), p=2, dim=1)
            s = self._apply_ansatz(valid.to(torch.complex64))
            probs = s.real ** 2 + s.imag ** 2
            features[mask] = (probs @ self.sgn).to(x.dtype)

        H_out = (H_in - self.kernel_size) // self.stride + 1
        W_out = (W_in - self.kernel_size) // self.stride + 1
        features = features.view(batch_size, num_patches, self.n_qubits)
        return features.permute(0, 2, 1).view(batch_size, self.n_qubits, H_out, W_out)
class FlexibleQCNN(nn.Module):
    def __init__(self, num_classes, img_size, kernel_size, n_qubits,
                 ansatz='qaoa', p=6, n_layers=2, n_blocks=3, seed=0):
        super().__init__()
        self.n_qubits = n_qubits
        self.quantum_conv = FlexibleQuantumConvLayer(
            kernel_size, n_qubits, kernel_size,
            ansatz=ansatz, p=p, n_layers=n_layers,
            n_blocks=n_blocks, seed=seed)
        self.pool = nn.MaxPool2d(2)

        with torch.no_grad():
            d = torch.randn(1, 1, img_size, img_size)
            o = self.quantum_conv(d)
            if n_qubits != 10:
                o = self.pool(o)
            n_features = o.numel()

        self.fc = nn.Linear(n_features, num_classes)
        self.n_trainable_q = self.quantum_conv.n_trainable

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.quantum_conv(x)
        if self.n_qubits != 10:
            x = self.pool(x)
        return self.fc(x.reshape(x.size(0), -1))

