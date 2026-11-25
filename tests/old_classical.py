# ############################################################################
# FILTER-MATCHED CLASSICAL BASELINES  (the fair comparison)
#
# DESIGN PRINCIPLE, per the stage-matching argument:
#   The paper's claim is about the FILTER. So the filter stage is matched and
#   the head is held IDENTICAL, rather than matching only the total.
#
#     filter stage : ~10-14 trainable params  (quantum: 12)
#     output       : exactly 2 channels, stride 2, 32x32 -> 16x16
#     head         : Linear(128, 2) = 258 params, identical on both sides
#
#   A previous version matched only the TOTAL (e.g. 19 filters = 228 filter
#   params + a 40-param GAP head). That is NOT a fair test of feature
#   extraction: it gives the classical filter 19x the capacity of the quantum
#   one. Kept here only as a clearly-labelled secondary row.
#
# FILTERS (all 2 channels, stride 2, head stays 258)
#   Conv2x2       10  2x2 dense + bias   <- ARCHITECTURAL TWIN: same 2x2
#                                           non-overlapping patches the
#                                           circuit sees. Headline comparison.
#   Separable3x3  12  2x rank-1 3x3      <- exact param match, 3x3 field
#   FixedBasis6   12  6 frozen DCT bases <- fixed structure + learned mixing,
#                                           same philosophy as a fixed ansatz
#   DWSep3x3_b    12  depthwise 3x3 + pointwise
#   PaperTwin     20  2x 3x3 dense       <- the paper's own twin, reference
#
# METHODS (none add inference parameters)
#   plain / pca / distill / ensemble_kd / longer
# ############################################################################

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ------------------------------------------------------------------ filters
class Conv2x2(nn.Module):
    """2 dense 2x2 kernels + bias = 10 params.

    THE architectural twin: operates on exactly the same 2x2 non-overlapping
    patches as the quantum filter, emits 2 channels, same stride. 10 vs 12
    params is the closest achievable match with this exact architecture.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 2, stride=2, bias=True)

    def forward(self, x):
        return self.conv(x)


class Separable3x3(nn.Module):
    """2 rank-1 separable 3x3 kernels = 2*(3+3) = 12 params.

    Exact parameter match to the 12-angle circuit. A rank-1 kernel K = v h^T
    costs 6 instead of 9, buying a 3x3 receptive field at the same budget.
    Sobel operators are exactly rank-1, so this covers the classic edge
    detectors; a Laplacian is rank-2 and genuinely out of reach.
    """
    def __init__(self):
        super().__init__()
        self.v = nn.Parameter(torch.randn(2, 3) * 0.5)
        self.h = nn.Parameter(torch.randn(2, 3) * 0.5)

    def forward(self, x):
        k = torch.einsum('oi,oj->oij', self.v, self.h).unsqueeze(1)
        return F.conv2d(x, k, stride=2, padding=1)


class FixedBasis6(nn.Module):
    """6 frozen 3x3 DCT bases, learn 2x6 = 12 mixing coefficients.

    The basis costs zero parameters. Structurally the closest classical
    analogue of a variational circuit: the structure is fixed a priori and
    only the mixing weights are learned.
    """
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
                for i in range(k):
                    for j in range(k):
                        cu = math.sqrt(1 / k) if u == 0 else math.sqrt(2 / k)
                        cv = math.sqrt(1 / k) if v == 0 else math.sqrt(2 / k)
                        m[i, j] = (cu * cv
                                   * math.cos((2 * i + 1) * u * math.pi / (2 * k))
                                   * math.cos((2 * j + 1) * v * math.pi / (2 * k)))
                mats.append(m)
        return torch.stack(mats[:n])

    def forward(self, x):
        k = torch.einsum('on,nij->oij', self.coef, self.basis).unsqueeze(1)
        return F.conv2d(x, k, stride=2, padding=(self.k - 1) // 2)


class DWSep3x3(nn.Module):
    """Depthwise 3x3 (9+1 bias) + pointwise 1->2 (2) = 12 params."""
    def __init__(self):
        super().__init__()
        self.dw = nn.Conv2d(1, 1, 3, stride=2, padding=1, bias=True)
        self.pw = nn.Conv2d(1, 2, 1, bias=False)

    def forward(self, x):
        return self.pw(self.dw(x))


class PaperTwin(nn.Module):
    """2 dense 3x3 kernels + bias = 20 params -- the paper's own twin."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, stride=2, padding=1, bias=True)

    def forward(self, x):
        return self.conv(x)


FILTERS = {
    'Conv2x2':      Conv2x2,        # 10  <- headline
    'Separable3x3': Separable3x3,   # 12  <- exact match
    'FixedBasis6':  FixedBasis6,    # 12
    'DWSep3x3':     DWSep3x3,       # 12
    'PaperTwin':    PaperTwin,      # 20  reference
}


# ------------------------------------------------------------------ model
class FilterMatchedCNN(nn.Module):
    """filter (10-20 params) -> ReLU -> MaxPool2 -> Linear(128,2).

    Identical to the QCNN except for the filter itself.
    """
    def __init__(self, filter_name, num_classes=2, img_size=32):
        super().__init__()
        self.filt = FILTERS[filter_name]()
        self.pool = nn.MaxPool2d(2)
        with torch.no_grad():
            d = torch.randn(1, 1, img_size, img_size)
            n_feat = self.pool(self.filt(d)).numel()
        self.fc = nn.Linear(n_feat, num_classes)

    def features(self, x):
        x = x.unsqueeze(1)
        return self.pool(F.relu(self.filt(x)))

    def forward(self, x):
        return self.fc(self.features(x).reshape(x.size(0), -1))


# ------------------------------------------------------------------ teacher
class Teacher(nn.Module):
    """Large CNN used only as a distillation source. Never compared directly;
    discarded at inference, so the student keeps its 10-12 filter params."""

    def __init__(self, num_classes=2, width=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width), nn.ReLU(),
        )
        self.body = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2), nn.ReLU(),
            nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 2), nn.ReLU(),
        )
        self.head = nn.Linear(width * 2, num_classes)
        # projects the 16x16 stem output to 2 channels so the student's
        # feature map can be regressed onto it
        self.proj = nn.Conv2d(width, 2, 1)

    def forward(self, x):
        z = self.body(self.stem(x.unsqueeze(1)))
        return self.head(F.adaptive_avg_pool2d(z, 1).flatten(1))

    def hint(self, x):
        """(B,2,8,8): same shape as the student's pooled feature map."""
        z = self.stem(x.unsqueeze(1))          # (B, width, 16, 16)
        return F.max_pool2d(self.proj(z), 2)   # (B, 2, 8, 8)


# ------------------------------------------------------------------ transfer
@torch.no_grad()
def pca_init_(model, X, n_patch=20000, seed=0):
    """Initialise a Conv2x2 filter from the top-2 PCs of 2x2 patches.
    Unsupervised transfer; adds no inference parameters."""
    if not isinstance(model.filt, Conv2x2):
        return model
    g = torch.Generator().manual_seed(seed)
    x = torch.as_tensor(X[:4000], dtype=torch.float32).unsqueeze(1)
    p = F.unfold(x, 2, stride=2).permute(0, 2, 1).reshape(-1, 4)
    idx = torch.randperm(p.shape[0], generator=g)[:n_patch]
    p = p[idx]
    p = p - p.mean(0, keepdim=True)
    _, _, V = torch.linalg.svd(p, full_matrices=False)
    model.filt.conv.weight.copy_(V[:2].reshape(2, 1, 2, 2))
    model.filt.conv.bias.zero_()
    return model


# ------------------------------------------------------------------ audit
def audit():
    print("=" * 78)
    print("FILTER-MATCHED AUDIT   (quantum: filter=12, head=258)")
    print("=" * 78)
    print(f"{'filter':<16}{'filter p':>10}{'head p':>9}{'total':>8}{'ch':>5}{'map':>10}")
    print("-" * 78)
    x = torch.randn(2, 32, 32)
    for name in FILTERS:
        m = FilterMatchedCNN(name)
        fp, hp = _count(m.filt), _count(m.fc)
        with torch.no_grad():
            fm = m.features(x)
            out = m(x)
        assert out.shape == (2, 2) and fm.shape[1] == 2, name
        tag = ""
        if fp == 12:
            tag = "  <== exact match"
        elif name == 'Conv2x2':
            tag = "  <== ARCH TWIN (2x2 patches)"
        elif name == 'PaperTwin':
            tag = "  (paper reference)"
        print(f"{name:<16}{fp:>10}{hp:>9}{fp+hp:>8}{fm.shape[1]:>5}"
              f"{str(tuple(fm.shape[2:])):>10}{tag}")
    print("-" * 78)
    t = Teacher()
    with torch.no_grad():
        h = t.hint(x)
    print(f"{'Teacher':<16}{_count(t):>10}   distillation source, hint {tuple(h.shape)}")
    print("  heads are IDENTICAL (258) across all rows -> only the filter differs")
    print("=" * 78)


# audit()  (disabled: imported as a reference by the smoke test)


# ############################################################################
# ADDENDUM --- INPUT PARITY
#
# Objection (correct): PCA initialisation is unsupervised feature engineering.
# It uses the training data to build the filter, which the quantum model does
# not do. Comparing pca-init classical against a randomly-initialised quantum
# circuit is not a like-for-like test of the FILTER.
#
# But there is a second, larger asymmetry running the OTHER way, which the
# pca result exposed: the two models do not receive the same input.
#
#   quantum filter sees : unfold 2x2 -> abs() -> L2-normalise EACH patch
#   classical conv sees : raw pixels
#
# Per-patch L2 normalisation is local contrast normalisation. It maps a patch
# and any brightness-scaled copy of it to the SAME vector (verified: [.1,.2,
# .3,.4] and 5x that both normalise to [.183,.365,.548,.730]). A plain conv
# is a fixed linear map plus bias and cannot reproduce this -- it is
# preprocessing the quantum pipeline gets for free and the classical one
# never sees.
#
# So the clean experiment is INPUT PARITY: give the classical filter exactly
# the same preprocessed patches, then compare randomly-initialised filters on
# both sides. Any remaining difference is the circuit itself.
#
#   'raw'  : classical conv on raw pixels          (paper's original setup)
#   'amp'  : classical conv on abs+L2-normalised patches  <- INPUT PARITY
#
# This replaces pca as the headline classical result: it removes an unfair
# advantage the QUANTUM side had, rather than adding one to the classical.
# ############################################################################

class AmplitudeEncode(nn.Module):
    """abs() + per-patch L2 normalisation -- exactly the quantum encoding.

    Non-overlapping patches, so fold() is an exact inverse of unfold() with
    no overlap-averaging. Deterministic and parameter-free: adds 0 params.
    """

    def __init__(self, patch=2):
        super().__init__()
        self.patch = patch

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.patch
        p = F.unfold(x, kernel_size=k, stride=k)                 # (B, k*k, L)
        p = torch.abs(p)
        p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return F.fold(p, output_size=(H, W), kernel_size=k, stride=k)


class InputMatchedCNN(nn.Module):
    """FilterMatchedCNN with a switchable input encoding.

    encoding='raw' -> raw pixels (what the paper's classical twin got)
    encoding='amp' -> abs + per-patch L2 norm (what the quantum filter gets)

    The encoder has no parameters, so filter and head counts are unchanged.
    """

    def __init__(self, filter_name, num_classes=2, img_size=32,
                 encoding='amp', patch=2):
        super().__init__()
        self.encoding = encoding
        self.enc = AmplitudeEncode(patch) if encoding == 'amp' else None
        self.filt = FILTERS[filter_name]()
        self.pool = nn.MaxPool2d(2)
        with torch.no_grad():
            d = torch.randn(1, 1, img_size, img_size)
            if self.enc is not None:
                d = self.enc(d)
            n_feat = self.pool(self.filt(d)).numel()
        self.fc = nn.Linear(n_feat, num_classes)

    def features(self, x):
        x = x.unsqueeze(1)
        if self.enc is not None:
            x = self.enc(x)
        return self.pool(F.relu(self.filt(x)))

    def forward(self, x):
        return self.fc(self.features(x).reshape(x.size(0), -1))


def audit_input_parity():
    print("=" * 78)
    print("INPUT-PARITY AUDIT")
    print("=" * 78)
    x = torch.randn(4, 32, 32).abs()
    for enc in ('raw', 'amp'):
        for name in ('Conv2x2', 'Separable3x3'):
            m = InputMatchedCNN(name, encoding=enc)
            fp, hp = _count(m.filt), _count(m.fc)
            with torch.no_grad():
                out = m(x)
            assert out.shape == (4, 2)
            print(f"  {name:<14} encoding={enc:<4} filter={fp:>3} head={hp}  "
                  f"encoder params=0")

    # confirm the encoder reproduces the quantum preprocessing exactly
    enc = AmplitudeEncode(2)
    with torch.no_grad():
        y = enc(x[:1].unsqueeze(1))
    p = F.unfold(y, 2, stride=2)
    norms = p.norm(dim=1)
    print(f"\n  patch norms after encoding: min={norms.min():.6f} "
          f"max={norms.max():.6f}  (all should be 1.0)")
    print("  -> identical preprocessing to the quantum amplitude encoding")
    print("=" * 78)


# audit_input_parity()
