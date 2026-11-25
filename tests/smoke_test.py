import os, sys, shutil
# Runs on synthetic images in a temp dir; touches no real data or results.
# Usage (Kaggle):  !python /kaggle/input/qbench-code/tests/smoke_test.py
import tempfile
SP = os.path.dirname(os.path.abspath(__file__))       # holds old_*.py references
PKG = os.path.dirname(SP)                              # pipeline/ (contains qbench/)
tmp = tempfile.mkdtemp(prefix='qbench_smoke_')
root = os.path.join(tmp, 'fake'); work = os.path.join(tmp, 'work')
os.environ['QBENCH_DATA'] = root; os.environ['QBENCH_WORK'] = work
os.environ['QBENCH_RESTORE'] = os.path.join(tmp, 'none')
import numpy as np, cv2, torch
rng = np.random.default_rng(0)
for cls, bright in [('healthy', 0.3), ('diseased', 0.7)]:
    for split, n in [('train', 40), ('validation', 12), ('test', 12)]:
        d = os.path.join(root, 'Corn', cls, split); os.makedirs(d)
        for i in range(n):
            img = np.clip(rng.normal(bright, 0.2, (64, 64)), 0, 1)
            cv2.imwrite(os.path.join(d, f'{i}.png'), (img * 255).astype(np.uint8))

sys.path.insert(0, PKG)
import qbench
from qbench import *
qbench.restore()
verify_quantum(); audit()

# --- same seed -> same initial weights as the OLD notebook classes
sys.path.insert(0, SP)
sys.dont_write_bytecode = True
import contextlib, io
import old_quantum
with contextlib.redirect_stdout(io.StringIO()):
    import old_classical
def same(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    va = [v for k, v in sa.items() if v.dtype.is_floating_point and 'running' not in k]
    vb = [v for k, v in sb.items() if v.dtype.is_floating_point and 'running' not in k]
    return len(va) == len(vb) and all(torch.equal(x, y) for x, y in zip(va, vb))
for tag, kw in [('ry_cnot_B6', dict(ansatz='ry_cnot', n_blocks=6)), ('hea_L2', dict(ansatz='hea', n_layers=2)),
                ('qaoa_p6', dict(ansatz='qaoa', p=6)), ('random', dict(ansatz='random')), ('identity', dict(ansatz='identity'))]:
    for seed in (0, 7):
        torch.manual_seed(seed); old = old_quantum.FlexibleQCNN(2, 32, 2, 2, seed=seed, **kw)
        torch.manual_seed(seed); new = build_model('q/' + tag, 2, seed=seed)
        x = torch.rand(5, 32, 32)
        assert same(old, new), tag
        with torch.no_grad(): assert torch.allclose(old(x), new(x), atol=1e-6), tag
for f in ['Conv2x2', 'Separable3x3', 'FixedBasis6', 'DWSep3x3', 'PaperTwin']:
    for enc in ['raw', 'amp']:
        torch.manual_seed(3); old = old_classical.InputMatchedCNN(f, 2, 32, encoding=enc)
        torch.manual_seed(3); new = build_model(f'c/{f}/{enc}', 2)
        x = torch.rand(5, 32, 32)
        assert same(old, new), (f, enc)
        with torch.no_grad(): assert torch.allclose(old(x), new(x), atol=1e-6), (f, enc)
print('OLD-vs-NEW: identical initial weights and outputs for all quantum + classical models')

# --- training: tune never touches test; final does; resume skips
qbench.config.NUM_EPOCHS = 2
R = {'epochs': 2}
df = run('t', ['Corn'], ['q/ry_cnot_B6', 'c/Conv2x2/amp', 'c/Pool2/raw'], seeds=[0, 1], recipe=R, phase='tune')
assert df['test_acc'].isna().all()
df = run('t', ['Corn'], ['q/ry_cnot_B6', 'c/Conv2x2/amp'], seeds=[0, 1], recipe=R, phase='final')
assert df[df.phase == 'final']['test_acc'].notna().all()
n = len(df)
df = run('t', ['Corn'], ['q/ry_cnot_B6'], seeds=[0, 1], recipe=R, phase='final')  # resume -> no new rows
assert len(df) == n, 'resume failed'
df = run('t', ['Corn'], ['c/Conv2x2/raw'], seeds=[0], phase='tune',
         recipe={'epochs': 2, 'act': 'leaky', 'norm': 'bn', 'warmup_epochs': 1, 'frozen': True})
print(df[['model', 'recipe', 'phase', 'seed', 'filter_params', 'val_acc', 'test_acc']])
fin = df[df.phase == 'final']
print(summarize(fin)); print(paired(fin, 'q/ry_cnot_B6', 'c/Conv2x2/amp', 'Corn'))
# determinism: same seed twice -> same val acc
d = CropData('Corn')
a, _ = qbench.run_one(d, 'q/ry_cnot_B3', 4, R); b, _ = qbench.run_one(d, 'q/ry_cnot_B3', 4, R)
assert a['val_acc'] == b['val_acc'] and a['best_epoch'] == b['best_epoch']
print(sorted(os.listdir(os.path.join(work, 'results'))))
print('SMOKE TEST PASSED')
