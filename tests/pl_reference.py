"""PennyLane side of the simulator check (run_local.py pl_check).

Runs in its own environment (.venv_pl: pennylane, no torch), so installing
PennyLane cannot change the packages the results were produced with.

Reads the parameters and inputs written by qbench/plcheck.py, runs the circuits
on default.qubit, writes the outputs. The whole-image circuits follow the
notebooks' code: w_gate_pl, apply_pool_gate_pl, qcnn_circuit (fork-of-gpu-run,
cell 24), qcnn_circuit_FC (gpu-run-all-crops, cell 20), qcnn_feature_extractor
and multi_filter_conv_layer* (qcnn-multiple-filters).
"""
import json
import sys

import numpy as np
import pennylane as qml

POOL1 = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
ACTIVE2 = [1, 3, 5, 7, 9]


def w_gate_pl(p, pair):
    """The notebooks' filter: (RY RY, CNOT, RY RY, CNOT, RY RY) x P_VAL."""
    for j in range(len(p) // 6):
        qml.RY(p[6 * j], wires=pair[0])
        qml.RY(p[6 * j + 1], wires=pair[1])
        qml.CNOT(wires=[pair[0], pair[1]])
        qml.RY(p[6 * j + 2], wires=pair[0])
        qml.RY(p[6 * j + 3], wires=pair[1])
        qml.CNOT(wires=[pair[0], pair[1]])
        qml.RY(p[6 * j + 4], wires=pair[0])
        qml.RY(p[6 * j + 5], wires=pair[1])


def b6_gate(p, pair):
    """The pipeline's RY-CNOT B6: RY RY, CNOT, ..., RY RY. p (6, 2)."""
    for b in range(len(p)):
        qml.RY(p[b, 0], wires=pair[0])
        qml.RY(p[b, 1], wires=pair[1])
        if b < len(p) - 1:
            qml.CNOT(wires=[pair[0], pair[1]])


def apply_pool_gate_pl(source, target, theta0, theta1):
    qml.ctrl(qml.RZ, control=source)(theta0, wires=target)
    qml.X(wires=source)
    qml.ctrl(qml.RX, control=source)(theta1, wires=target)
    qml.X(wires=source)


def staggered(gate, wires):
    for i in range(0, len(wires) - 1, 2):
        gate([wires[i], wires[i + 1]])
    for i in range(1, len(wires) - 1, 2):
        gate([wires[i], wires[i + 1]])


def whole_image(meta, a, x):
    filt = {'b6': b6_gate, 'nb': w_gate_pl}[meta['filt']]
    th, pl = a['theta'], a['pool']
    anc = meta['ancilla']
    dev = qml.device('default.qubit', wires=11 if anc else 10)

    @qml.qnode(dev)
    def circuit(v):
        qml.AmplitudeEmbedding(features=np.abs(v), wires=range(10), normalize=True)
        if anc:
            qml.Hadamard(wires=10)

            def multi(k0, k1):
                def gate(pair):
                    qml.ctrl(filt, control=[10], control_values=[0])(th[k0], pair)
                    qml.ctrl(filt, control=[10], control_values=[1])(th[k1], pair)
                return gate
            staggered(multi(0, 1), list(range(10)))
            for k, (s, t) in enumerate(POOL1):
                apply_pool_gate_pl(s, t, pl[k, 0], pl[k, 1])
            staggered(multi(2, 3), ACTIVE2)
            apply_pool_gate_pl(1, 3, pl[5, 0], pl[5, 1])
            apply_pool_gate_pl(5, 7, pl[6, 0], pl[6, 1])
            return qml.probs(wires=[3, 7, 9])

        staggered(lambda pair: filt(th[0], pair), list(range(10)))
        for k, (s, t) in enumerate(POOL1):
            apply_pool_gate_pl(s, t, pl[k, 0], pl[k, 1])
        staggered(lambda pair: filt(th[1], pair), ACTIVE2)
        apply_pool_gate_pl(1, 3, pl[5, 0], pl[5, 1])
        apply_pool_gate_pl(5, 7, pl[6, 0], pl[6, 1])
        filt(th[2], [3, 7])
        filt(th[2], [7, 9])
        apply_pool_gate_pl(3, 7, pl[7, 0], pl[7, 1])

        vqc = meta['vqc']
        if vqc == 'sel':
            qml.StronglyEntanglingLayers(weights=a['vqc'], wires=[7, 9])
        elif vqc == 'ry':                  # first FQC rebuild: RY RY, CNOT, per layer
            for l in range(len(a['vqc'])):
                qml.RY(a['vqc'][l, 0], wires=7)
                qml.RY(a['vqc'][l, 1], wires=9)
                qml.CNOT(wires=[7, 9])
        out = meta['out']
        if out == 'z79':
            return [qml.expval(qml.PauliZ(w)) for w in (7, 9)]
        if out == 'z9':
            return [qml.expval(qml.PauliZ(9))]
        return qml.probs(wires=[9])
    return np.array([np.array(circuit(v)).reshape(-1) for v in x])


def patch_filter(meta, a, x):
    n, name = meta['n'], meta['ansatz']
    dev = qml.device('default.qubit', wires=n)
    ring = [(c, (c + 1) % n) for c in range(n)]

    @qml.qnode(dev)
    def circuit(v):
        qml.AmplitudeEmbedding(features=np.abs(v), wires=range(n), normalize=True)
        if name == 'random':
            qml.QubitUnitary(a['U'], wires=range(n))
        elif name == 'ry_cnot':
            th = a['theta']
            for b in range(len(th)):
                for q in range(n):
                    qml.RY(th[b, q], wires=q)
                if b < len(th) - 1:
                    for c in range(n - 1):
                        qml.CNOT(wires=[c, c + 1])
        elif name == 'wgate':              # the notebooks' w_gate_pl, row by row
            w_gate_pl(a['theta'].reshape(-1), [0, 1])
        elif name == 'qaoa':               # cost exp(i gamma/2 ZZ) then mixer RX(2 beta)
            for g, b in zip(a['gammas'], a['betas']):
                qml.IsingZZ(-g, wires=[0, 1])
                qml.RX(2 * b, wires=0)
                qml.RX(2 * b, wires=1)
        elif name in ('hea', 'ry_only'):
            th = a['theta']
            for l in range(len(th)):
                for q in range(n):
                    if name == 'hea':
                        qml.RY(th[l, q, 0], wires=q)
                        qml.RZ(th[l, q, 1], wires=q)
                    else:
                        qml.RY(th[l, q], wires=q)
                for c, t in ring:
                    qml.CNOT(wires=[c, t])
        elif name != 'identity':
            raise ValueError(name)
        return [qml.expval(qml.PauliZ(q)) for q in range(n)]
    return np.array([np.array(circuit(v)) for v in x])


def qnn(meta, a, x):
    n = x.shape[1]
    dev = qml.device('default.qubit', wires=n)

    @qml.qnode(dev)
    def circuit(v):                        # the notebook's quantum_circuit_qnn
        qml.AngleEmbedding(v, wires=range(n))
        qml.StronglyEntanglingLayers(a['w'], wires=range(n))
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]
    return np.array([np.array(circuit(v)) for v in x])


def main(folder):
    cases = json.load(open(f'{folder}/cases.json'))
    z = np.load(f'{folder}/cases.npz')
    out = {}
    run = {'arch': whole_image, 'filter': patch_filter, 'qnn': qnn}
    for i, c in enumerate(cases):
        a = {k.split('/', 1)[1]: z[k].astype(np.float64 if z[k].dtype.kind == 'f' else z[k].dtype)
             for k in z.files if k.startswith(f'{i}/')}
        out[str(i)] = run[c['kind']](c, a, a['x'])
        print(f"  [{i + 1}/{len(cases)}] {c['name']}", flush=True)
    np.savez(f'{folder}/reference.npz', **out)
    json.dump({'pennylane': qml.__version__, 'numpy': np.__version__, 'device': 'default.qubit'},
              open(f'{folder}/reference_versions.json', 'w'))


if __name__ == '__main__':
    main(sys.argv[1])
