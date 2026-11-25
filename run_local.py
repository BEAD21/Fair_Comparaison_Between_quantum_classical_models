r"""Every experiment of the paper is one command:

    python run_local.py <command> [arguments]

See README.md for the commands used in the paper and their order. Results go
to work/results/ (override with the QBENCH_WORK environment variable) and are
resumable: rows that already exist are skipped.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('QBENCH_DATA', os.path.join(HERE, 'data', 'PlantVillage_Processed_700 images'))
os.environ.setdefault('QBENCH_WORK', os.path.join(HERE, 'work'))
os.environ.setdefault('QBENCH_RESTORE', os.path.join(HERE, 'work', '_none'))
sys.path.insert(0, HERE)
import pandas as pd
import qbench
from qbench import run, summarize, paired
from qbench.report import build, build_tuning
from qbench import tune as T
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 20)


def _open(path):
    """Open an HTML report in the default browser (any OS); never fails."""
    try:
        import webbrowser
        from pathlib import Path
        webbrowser.open(Path(path).resolve().as_uri())
    except Exception:                      # noqa: BLE001  (headless machine)
        pass

def repro():
    print('device:', qbench.train.device)
    df = run('repro', ['Corn', 'Cherry'], ['q/ry_cnot_B6', 'c/Conv2x2/raw'], phase='final')
    print(summarize(df).round(2).to_string(index=False))
    print(pd.DataFrame([paired(df, 'q/ry_cnot_B6', 'c/Conv2x2/raw', c)
                        for c in ['Cherry', 'Corn']]).round(4).to_string(index=False))
    report('repro', 'q/ry_cnot_B6')


def report(name, ref=None):
    path = build(name, ref=ref)
    print(f"report -> {path}")
    _open(path)

def tune(*families):
    families = families or ('classical_raw', 'classical_amp', 'quantum')   # cheap ones first
    print('device:', qbench.train.device)
    for fam in families:
        T.tune(fam, qbench.config.MAIN_CROPS)
        tune_report()
    best = T.select(qbench.config.MAIN_CROPS)
    for crop, fams in best.items():
        print(f"\n{crop}")
        for fam, b in fams.items():
            print(f"  {fam:<14} {b['model']:<22} last-5 val {b['val_last5']:.2f}  "
                  f"(best-epoch {b['val_mean']:.2f} +/- {b['val_sd']:.2f})  {b['recipe']}")


def tune_report():
    fams = [f for f in T.FAMILIES if (qbench.config.RESULTS_DIR / f'tune_{f}.csv').exists()]
    path = build_tuning(fams)
    print(f"tuning report -> {path}")


# no-filter baselines, each trained with its family's winning recipe
BASELINES = {'quantum': ['q/identity', 'q/random'],
             'classical_raw': ['c/Pool2/raw'], 'classical_amp': ['c/Pool2/amp']}


def pipeline(crop):
    """Whole pipeline on one crop: tune -> select (val) -> final 10 seeds -> report."""
    from qbench.train import DEFAULT_RECIPE, recipe_id
    from qbench.report import _label
    print('device:', qbench.train.device)
    for fam in T.FAMILIES:                                   # 1. tune (validation only)
        T.tune(fam, [crop])
    tune_report()
    best = T.select([crop], name=f'best_configs_{crop}')[crop]   # 2. select

    plan = [(best[f]['model'], best[f]['recipe']) for f in T.FAMILIES]
    if best['quantum']['model'] != 'q/ry_cnot_B6':           # B6 always as a named row
        lb = T.leaderboard('quantum', crop)
        top = lb[lb['model'] == 'q/ry_cnot_B6'].iloc[0]
        plan.append(('q/ry_cnot_B6', T.recipe_from_id(top['recipe'])))
    for fam, models in BASELINES.items():
        plan += [(m, best[fam]['recipe']) for m in models]

    store = f'final_{crop}'                                  # 3. final, test once
    for m, r in plan:
        print(f"\n=== final {m}  {recipe_id({**DEFAULT_RECIPE, **r})}")
        df = run(store, [crop], [m], recipe=r, phase='final', verbose=False)
    df['label'] = [_label(m, r) for m, r in zip(df['model'], df['recipe'])]
    ref = _label(best['quantum']['model'], recipe_id({**DEFAULT_RECIPE, **best['quantum']['recipe']}))
    print(summarize(df, by=['crop', 'label']).round(2).to_string(index=False))
    report(store, ref)                                       # 4. report


def confirm(*families):
    """Phase 2b: 10 validation seeds on the dev crops, then freeze (rule in tune.py)."""
    print('device:', qbench.train.device)
    print('candidates (from Phase 2 only):')
    for fam in T.FAMILIES:
        for m, r in T.candidates(fam):
            print(f"  {fam:<14} {m:<22} {r}")
    for fam in families or ('classical_raw', 'classical_amp', 'quantum'):
        T.confirm(fam)
    confirm_report()
    frozen = T.freeze()
    for fam, e in frozen['families'].items():
        print(f"\n{fam:<14} {e['model']:<22} score {e['score']:.2f}  {e['recipe']}")
        if 'b6_vs_winner' in e:
            t, w = e['b6_vs_winner'], e['validation_winner']
            print(f"   validation winner {w['model']} {w['score']:.2f}; B6 - winner "
                  f"{t['diff']:+.2f} [{t['lo']:+.2f}, {t['hi']:+.2f}] p={t['p_t']:.3g} "
                  f"-> chosen by {e['chosen_by']}")


def confirm_report():
    """work/reports/confirm_<family>.html: candidates per dev crop, last-5 val."""
    from qbench.report import _label
    R = qbench.config.RESULTS_DIR
    for fam in T.FAMILIES:
        rows = T.confirm_rows(fam)
        if not len(rows):
            continue
        rows.to_csv(R / f'confirm_{fam}.csv', index=False)
        cur = pd.read_csv(R / f'tune_{fam}_curves.csv')
        cur = cur.merge(rows[['crop', 'model', 'recipe', 'seed', 'phase']].drop_duplicates())
        cur.to_csv(R / f'confirm_{fam}_curves.csv', index=False)
        lead = T.confirm_board(fam).iloc[0]
        path = build(f'confirm_{fam}', ref=_label(lead['model'], lead['recipe']),
                     title=f'Phase 2b confirmation: {fam}', metric='val_last5')
        print(f"confirm report -> {path}")


def arch(study='arch', crops='all', lrs='all'):
    """Architecture study (qbench/archstudy.py): its workers run at once
    (GPU and CPU). `arch` = 40-epoch B6 study, `arch100` = notebook models.
    crops / lrs: comma lists to run a subset, e.g.  arch100 Grape 0.005"""
    import subprocess, time
    from qbench import archstudy as A
    t0 = time.time()
    workers = A.STUDIES[study]['workers']
    n_cpu = sum(dev == 'cpu' for dev, _ in workers)
    procs = []
    for dev, models in workers:
        env = {**os.environ, 'QBENCH_DEVICE': dev}
        procs.append(subprocess.Popen([sys.executable, __file__, 'arch_worker', study, dev,
                                       str(n_cpu), crops, lrs, *models], env=env))
    codes = [p.wait() for p in procs]
    print(f"\nall {len(procs)} workers done in {(time.time() - t0) / 60:.0f} min, "
          f"exit codes {codes}")
    arch_report(study, crops, lrs)


def arch100(crops='all', lrs='all'):
    arch('arch100', crops, lrs)


def _crops_lrs(crops, lrs):
    from qbench import archstudy as A
    c = A.DEV_CROPS if crops == 'all' else crops.split(',')
    l = A.LRS if lrs == 'all' else [float(x) for x in lrs.split(',')]
    return c, l


def arch_worker(study, dev, n_cpu, crops, lrs, *models):
    import torch
    from qbench import archstudy as A
    if dev == 'cpu':           # share the cores; leave some for the GPU worker's Python
        torch.set_num_threads(max(1, ((os.cpu_count() or 4) - 2) // int(n_cpu)))
    c, l = _crops_lrs(crops, lrs)
    A.run_models(models, study, crops=c, tag=f"[{dev}] ", lrs=l)


def arch_report(study='arch', crops='all', lrs='all'):
    """Print the leaderboards; work/reports/<prefix><group>.html."""
    from qbench import archstudy as A
    from qbench.report import _label
    from qbench.train import recipe_id, DEFAULT_RECIPE
    R = qbench.config.RESULTS_DIR
    S = A.STUDIES[study]
    c, l = _crops_lrs(crops, lrs)
    for group, models in S['groups'].items():
        df = A.rows(models, study, crops=c)
        if not len(df):
            continue
        if lrs != 'all':                   # only the chosen learning rates
            ids = {recipe_id({**DEFAULT_RECIPE, 'lr': lr, 'epochs': e})
                   for lr in l for e in [S['max_epochs'], *S['shorter']]}
            df = df[df['recipe'].isin(ids)]
            ep = df['recipe'].map(lambda r: T.recipe_from_id(r).get('epochs', 20))
            cols = ['val_acc', 'val_last5'] + (['scale_s'] if 'scale_s' in df else [])
            print(f"\n== {group}: every epoch count (mean over {c} x seeds)")
            print(df.assign(epochs=ep).groupby(['model', 'epochs'])[cols]
                    .agg(['mean', 'std']).round(2).to_string())
        b = A.board(group, study, crops=c, df=df)
        if not len(b):
            continue
        crops = [x for x in A.DEV_CROPS if x in b.columns]
        extra = ['scale_s'] if 'scale_s' in b else []
        print(f"\n== {group}  (val_last5: best recipe per model, mean over crops x seeds)")
        print(b[['model', 'recipe', 'score', *crops, 'worst_recipe', 'filter_params',
                 'total_params', 'sec', *extra]].round(2).to_string(index=False))
        keep = set(zip(b['model'], b['recipe']))
        best = df[[(m, r) in keep for m, r in zip(df['model'], df['recipe'])]]
        name = f"{S['prefix']}{group}" if study == 'arch' else group
        if crops != A.DEV_CROPS or lrs != 'all':          # subset: own report file
            name += '_' + '_'.join(crops) + ('' if lrs == 'all' else '_lr' + lrs.replace(',', '_'))
        best.to_csv(R / f'{name}.csv', index=False)
        cur = pd.concat([pd.read_csv(R / f'{A.store_name(m, study)}_curves.csv')
                         for m in b['model']], ignore_index=True)
        cur.merge(best[['crop', 'model', 'recipe', 'seed', 'phase']]).to_csv(
            R / f'{name}_curves.csv', index=False)
        ref = b[b['model'] == 'q/ry_cnot_B6'].iloc[0]
        path = build(name, ref=_label(ref['model'], ref['recipe']),
                     title=f'Architecture study: {group.replace("_", " ")}', metric='val_last5')
        print(f"report -> {path}")


FQC_MODELS = ['qa/fqc', 'qa/fqc_sel', 'qa/fqc_nb', 'qa/fqc_nb_scale']


def fqc_test(*models):
    """FQC variants on Grape, the notebook's settings: lr 0.005, 100 epochs,
    seeds 100-102, validation only. The 40-epoch row is read from the same run."""
    from qbench.archstudy import _prefix_row
    from qbench.data import CropData
    from qbench.results import ResultStore
    from qbench.train import run_one, recipe_id, DEFAULT_RECIPE
    print('device:', qbench.train.device)
    store, d, lr = ResultStore('fqc_test'), CropData('Grape'), 0.005
    for m in models or FQC_MODELS:
        for s in T.TUNE_SEEDS:
            if store.done('Grape', m, recipe_id({**DEFAULT_RECIPE, 'epochs': 100}), s, 'tune'):
                continue
            row, hist = run_one(d, m, s, {'lr': lr, 'epochs': 100}, 'tune')
            row['derived_from'] = 0
            store.add(row, hist)
            store.add(*_prefix_row(row, hist, 40, lr))
            print(f"{m:<12} seed {s} | best val {row['val_acc']:6.2f} @ep{row['best_epoch']:3d} | "
                  f"last5 {row['val_last5']:6.2f} | {row['seconds']:5.0f}s", flush=True)
    df = store.df()
    df['epochs'] = df['recipe'].map(lambda r: T.recipe_from_id(r).get('epochs', 20))
    cols = ['val_acc', 'val_last5', 'filter_params'] + (['scale_s'] if 'scale_s' in df else [])
    print(df.groupby(['model', 'epochs'])[cols].mean().round(2).to_string())
    report('fqc_test', 'qa/fqc_nb  {"epochs": 100}')


# ---- QNN vs ANN (qbench/qnn.py, qbench/qnnstudy.py) ------------------------
QNN_TUNE_WORKERS = [['qn/sel_L4'], ['qn/sel_L3'], ['qn/sel_L1', 'qn/sel_L2'],
                    ['cn/dense_tanh', 'cn/dense_relu', 'cn/mlp5_tanh', 'cn/mlp5_relu']]


def _spawn(cmd, groups, crops):
    """One CPU process per group; features are computed here first so the
    workers only read the cache."""
    import subprocess, time
    from qbench.qnn import FeatureData, verify_qnn
    verify_qnn()
    for c in crops:
        print(FeatureData(c), flush=True)
    t0 = time.time()
    env = {**os.environ, 'QBENCH_DEVICE': 'cpu'}
    procs = [subprocess.Popen([sys.executable, __file__, cmd, ','.join(crops), str(len(groups)), *g],
                              env=env) for g in groups]
    codes = [p.wait() for p in procs]
    print(f"\n{len(procs)} workers done in {(time.time() - t0) / 60:.0f} min, exit codes {codes}")


def _threads(n_workers):
    import torch
    torch.set_num_threads(max(1, ((os.cpu_count() or 4) - 1) // int(n_workers)))


def qnn_tune(crops='all'):
    """QNN / ANN search on the dev crops (validation only), then qnn_report."""
    from qbench import qnnstudy as Q
    c = Q.DEV_CROPS if crops == 'all' else crops.split(',')
    _spawn('qnn_worker', QNN_TUNE_WORKERS, c)
    qnn_report()


def qnn_worker(crops, n, *models):
    from qbench import qnnstudy as Q
    _threads(n)
    Q.run_models(models, crops=crops.split(','), tag=f"[{models[0]}] ")


def qnn_report():
    """Leaderboards; freezes the winners once both grids are complete;
    work/reports/qnn_tune.html (best config per model)."""
    from qbench import qnnstudy as Q
    from qbench.qnn import QNN_FAMILIES
    from qbench.report import _label
    R = qbench.config.RESULTS_DIR
    best = []
    for fam in QNN_FAMILIES:
        b = Q.board(fam)
        if not len(b):
            continue
        crops = [x for x in Q.DEV_CROPS if x in b.columns]
        cols = ['model', 'recipe', 'score', 'sd', *crops, 'filter_params', 'total_params', 'sec']
        print(f"\n== {fam}: top 8 of {len(b)} configs (val_last5, mean over crops x seeds)")
        print(b[cols].head(8).round(2).to_string(index=False))
        per = b.groupby('model').head(1)
        print(f"-- best config per model (worst config overall {b['score'].min():.2f})")
        print(per[cols].round(2).to_string(index=False))
        best.append(per)
    if not best:
        return
    try:
        fz, p = Q.freeze()
        print(f"\nfrozen -> {p}\n{json.dumps(fz, indent=2)}")
    except RuntimeError as e:
        print(f"\nnot frozen yet: {e}")
    keep = set(zip(*[pd.concat(best)[k] for k in ('model', 'recipe')]))
    models = [m for ms in QNN_FAMILIES.values() for m in ms]
    df = Q.rows(models)
    df = df[[(m, r) in keep for m, r in zip(df['model'], df['recipe'])]]
    df.to_csv(R / 'qnn_tune.csv', index=False)
    cur = pd.concat([pd.read_csv(R / f'{Q.store_name(m)}_curves.csv') for m in models
                     if (R / f'{Q.store_name(m)}_curves.csv').exists()], ignore_index=True)
    cur.merge(df[['crop', 'model', 'recipe', 'seed', 'phase']]).to_csv(R / 'qnn_tune_curves.csv', index=False)
    lead = pd.concat(best).sort_values('score', ascending=False).iloc[0]
    path = build('qnn_tune', ref=_label(lead['model'], lead['recipe']),
                 title='QNN vs ANN: tuning (validation)', metric='val_last5')
    print(f"report -> {path}")


def qnn_final(crops='all'):
    """Frozen QNN / ANN + controls + notebook rows, all crops, seeds 0-9, test once."""
    from qbench import qnnstudy as Q
    c = qbench.config.CROPS if crops == 'all' else crops.split(',')
    jobs = Q.final_jobs()
    print('jobs:', *jobs, sep='\n  ')
    _spawn('qnn_final_worker', [[json.dumps(j)] for j in jobs], c)
    qnn_rebuild()


def qnn_rebuild():
    """Merge the workers' results into qnn_final.csv (nothing is trained), report."""
    from qbench import qnnstudy as Q
    df = Q.rebuild_final()
    print(f"qnn_final.csv rebuilt: {len(df)} rows")
    print(df.groupby(['model', 'recipe']).size().to_string())
    qnn_final_report()


def qnn_matched():
    """QNN with exactly the ANN's parameter counts (L=3, 90), test once, seeds 0-9."""
    from qbench import qnnstudy as Q
    fz = Q.freeze_matched()
    print(json.dumps(fz, indent=2), flush=True)
    job = (fz['model'], fz['recipe'])
    _spawn('qnn_final_worker', [[json.dumps(job)]], qbench.config.CROPS)
    qnn_rebuild()


def qnn_final_worker(crops, n, job):
    from qbench import qnnstudy as Q
    _threads(n)
    m, r = json.loads(job)
    Q.run_final([(m, r)], crops=crops.split(','), tag=f"[{m}] ")


def qnn_final_report():
    """Test accuracy per crop, QNN - ANN paired by (crop, seed); qnn_final.html."""
    import numpy as np
    from scipy import stats
    from qbench import qnnstudy as Q
    from qbench.report import _label
    from qbench.train import recipe_id, DEFAULT_RECIPE
    fz = json.loads((qbench.config.RESULTS_DIR / Q.FROZEN).read_text())
    df = pd.read_csv(qbench.config.RESULTS_DIR / 'qnn_final.csv')
    df['label'] = [_label(m, r) for m, r in zip(df['model'], df['recipe'])]
    t = df.pivot_table(index='label', columns='crop', values='test_acc', aggfunc='mean')
    t['MEAN'] = t.mean(axis=1)
    t['params'] = df.groupby('label')['total_params'].first()
    print("\n== test accuracy, mean over seeds 0-9")
    print(t.sort_values('MEAN', ascending=False).round(2).to_string())
    lab = {f: _label(fz[f]['model'], recipe_id({**DEFAULT_RECIPE, **fz[f]['recipe']})) for f in ('qnn', 'ann')}
    pairs = [(lab['qnn'], lab['ann'])] + [(lab['qnn'], _label(c, recipe_id({**DEFAULT_RECIPE, **fz['qnn']['recipe']})))
                                          for c in Q.CONTROLS['qnn']]
    mp = qbench.config.RESULTS_DIR / Q.MATCHED
    if mp.exists():
        mq = json.loads(mp.read_text())
        pairs.append((_label(mq['model'], recipe_id({**DEFAULT_RECIPE, **mq['recipe']})), lab['ann']))
    print("\n== paired by (crop, seed), test accuracy")
    for a, b in pairs:
        x = df[df['label'] == a].set_index(['crop', 'seed'])['test_acc']
        y = df[df['label'] == b].set_index(['crop', 'seed'])['test_acc'].reindex(x.index)
        d = (x - y).dropna()
        if len(d) < 2:
            continue
        h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        print(f"{a}  -  {b}: {d.mean():+.2f} [{d.mean() - h:+.2f}, {d.mean() + h:+.2f}]  "
              f"p={stats.ttest_rel(x.reindex(d.index), y.reindex(d.index)).pvalue:.3g}  n={len(d)}")
    report('qnn_final', lab['qnn'])


# ---- whole pipeline (qbench/router.py, qbench/endtoend.py) ------------------
def router():
    """ResNet-50 crop router + one-stage 16-class reference, seeds 0-9, test once.
    First run downloads the ImageNet weights (98 MB) and computes features."""
    from qbench.router import run_router
    from qbench.report import _label
    from qbench.train import recipe_id, DEFAULT_RECIPE
    from qbench.router import RECIPE
    print('device:', qbench.train.device)
    df = run_router()
    print(df.groupby('crop')[['val_acc', 'test_acc']].agg(['mean', 'std']).round(2).to_string())
    report('router', _label('rn/head', recipe_id({**DEFAULT_RECIPE, **RECIPE})))


def final(crops='all'):
    """Frozen QCNN / CNN configs + controls, 8 crops, seeds 0-9, test once;
    keeps per-image predictions for `endtoend`."""
    from qbench import endtoend as E
    from qbench.report import _label
    from qbench.train import recipe_id, DEFAULT_RECIPE
    c = qbench.config.CROPS if crops == 'all' else crops.split(',')
    print('device:', qbench.train.device)
    plan = E.qcnn_plan()
    for m, r in plan:
        print(f"\n=== final {m}  {recipe_id({**DEFAULT_RECIPE, **r})}", flush=True)
        df = run('final', c, [m], recipe=r, phase='final', verbose=False)
    df['label'] = [_label(m, r) for m, r in zip(df['model'], df['recipe'])]
    t = df.pivot_table(index='label', columns='crop', values='test_acc', aggfunc='mean')
    t['MEAN'] = t.mean(axis=1)
    print(t.sort_values('MEAN', ascending=False).round(2).to_string())
    m, r = plan[0]
    report('final', _label(m, recipe_id({**DEFAULT_RECIPE, **r})))


def endtoend():
    """Router + stage 2 on the same test images; work/reports/endtoend.html."""
    import numpy as np
    from scipy import stats
    from qbench import endtoend as E
    from qbench.report import _label
    df, skipped = E.build_table()
    for fam, why in skipped.items():
        print(f"skipped {fam}: {why}")
    rt = pd.read_csv(qbench.config.RESULTS_DIR / 'router.csv')
    for name, what in (('ALL8', 'router (8 crops)'), ('ALL16', 'one-stage ResNet (16 classes)')):
        v = rt[rt['crop'] == name]['test_acc']
        print(f"{what}: test {v.mean():.2f} sd {v.std():.2f}")
    if not len(df):
        return
    a = df[df['crop'] == 'ALL']
    fam = {f: a[a['family'] == f].set_index('seed') for f in E.ORDER if (a['family'] == f).any()}
    print("\n== all test images pooled, mean over seeds")
    print(f"{'family':<15}{'model':<18}{'params':>7}{'perfect router':>16}{'end-to-end':>12}{'sd':>6}")
    for f, g in fam.items():
        print(f"{f:<15}{g['model'].iloc[0]:<18}{int(g['total_params'].iloc[0]):>7}"
              f"{g['oracle_acc'].mean():>16.2f}{g['test_acc'].mean():>12.2f}{g['test_acc'].std():>6.2f}")
    print("\n== end-to-end per crop")
    t = df[df['crop'] != 'ALL'].pivot_table(index='family', columns='crop', values='test_acc', aggfunc='mean')
    print(t.reindex([f for f in E.ORDER if f in t.index]).round(2).to_string())
    print("\n== paired by seed, end-to-end")
    for x, y in (('quantum', 'classical_raw'), ('quantum', 'classical_amp'), ('qnn', 'ann'),
                 ('qnn_matched', 'ann')):
        if x in fam and y in fam:
            d = fam[x]['test_acc'] - fam[y]['test_acc'].reindex(fam[x].index)
            h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            print(f"{x} - {y}: {d.mean():+.2f} [{d.mean() - h:+.2f}, {d.mean() + h:+.2f}]  "
                  f"p={stats.ttest_rel(fam[x]['test_acc'], fam[y]['test_acc'].reindex(fam[x].index)).pvalue:.3g}  n={len(d)}")
    first = next(iter(fam.values()))
    report('endtoend', _label(first['model'].iloc[0], first['recipe'].iloc[0]))


def pl_check():
    """Every circuit vs PennyLane (separate environment .venv_pl). Trains nothing."""
    from qbench import plcheck
    python = (os.path.join(HERE, '.venv_pl', 'Scripts', 'python.exe') if os.name == 'nt'
              else os.path.join(HERE, '.venv_pl', 'bin', 'python'))
    if not os.path.exists(python):
        sys.exit('create it first: a Python 3.12 venv named .venv_pl with '
                 'pip install -r requirements-pennylane.txt (see README)')
    sys.exit(0 if plcheck.run(python) else 1)


# ---- quantum filter scan, validation only (qbench/filterscan.py) ------------
def filter_scan():
    """Every quantum filter + identity/random: best of the 12-recipe grid on the dev
    crops, then all crops in config.CROPS, test never evaluated. Then the table."""
    import subprocess, time
    from qbench import filterscan as F
    from qbench.data import CropData
    for c in qbench.config.CROPS:          # cache written here, workers only read it
        CropData(c)
    groups = [F.ALL[i::5] for i in range(5)]
    t0 = time.time()
    env = {**os.environ, 'QBENCH_DEVICE': 'cpu'}
    procs = [subprocess.Popen([sys.executable, __file__, 'filter_scan_worker', '5', *g], env=env) for g in groups]
    codes = [p.wait() for p in procs]
    print(f"\n5 workers done in {(time.time() - t0) / 60:.0f} min, exit codes {codes}")
    filter_scan_report()


def filter_scan_worker(n, *models):
    from qbench import filterscan as F
    _threads(n)
    F.run_model(models, tag=f"[{models[0]}] ")


def filter_scan_report():
    from qbench import filterscan as F
    t = F.table()
    if not len(t):
        sys.exit('no scan results yet')
    t.to_csv(qbench.config.RESULTS_DIR / 'filter_scan.csv', index=False)
    show = ['filter', 'params', *qbench.config.CROPS, 'dev', 'held_out', 'all', 'seed_sd', 'minus_ref', 'ci_seeds',
            'ci_crops', 'complete']
    print(t[[c for c in show if c in t]].to_string(index=False))
    path = F.write_html(t, qbench.config.WORK_DIR / 'reports' / 'filter_scan.html')
    print(f"report -> {path}")
    _open(path)


def classical_scan():
    """Classical counterparts through exactly the same scan as the quantum
    filters (12-recipe grid, Cherry + Grape, then 7 crops x seeds 100-109),
    validation only. Then the table, with B6 as the reference."""
    import subprocess, time
    from qbench import filterscan as F
    from qbench.data import CropData
    for c in qbench.config.CROPS:
        CropData(c)
    groups = [F.CLASSICAL[i::5] for i in range(5)]
    t0 = time.time()
    env = {**os.environ, 'QBENCH_DEVICE': 'cpu'}
    procs = [subprocess.Popen([sys.executable, __file__, 'filter_scan_worker', '5', *g], env=env) for g in groups]
    codes = [p.wait() for p in procs]
    print(f"\n5 workers done in {(time.time() - t0) / 60:.0f} min, exit codes {codes}")
    classical_scan_report()


def classical_scan_report():
    from qbench import filterscan as F
    t = F.table(models=F.CLASSICAL + [F.REF, 'q/identity'])
    if not len(t):
        sys.exit('no classical scan results yet')
    t.to_csv(qbench.config.RESULTS_DIR / 'classical_scan.csv', index=False)
    show = ['filter', 'params', *qbench.config.CROPS, 'dev', 'held_out', 'all', 'seed_sd', 'minus_ref', 'ci_seeds',
            'ci_crops', 'complete']
    print(t[[c for c in show if c in t]].to_string(index=False))
    path = F.write_html(t, qbench.config.WORK_DIR / 'reports' / 'classical_scan.html',
                        title='Classical counterparts vs B6: validation only')
    print(f"report -> {path}")
    _open(path)


# ---- final test of the filter comparison, once (qbench/filtertest.py) -------
def filter_test():
    """B6, Conv2x2 raw, PatchMLP raw, identity: frozen recipes, 7 crops, seeds 0-9,
    TEST once, predictions saved. Resumable. Then filter_test_report."""
    import subprocess, time
    from qbench import filtertest as T
    from qbench.data import CropData
    fz = T.freeze()
    print('frozen ->', qbench.config.RESULTS_DIR / T.FROZEN)
    print(json.dumps(fz['models'], indent=2), flush=True)
    for c in qbench.config.CROPS:
        CropData(c)
    js = [json.dumps(j) for j in T.jobs()]
    groups = [js[i::5] for i in range(5)]
    t0 = time.time()
    env = {**os.environ, 'QBENCH_DEVICE': 'cpu'}
    procs = [subprocess.Popen([sys.executable, __file__, 'filter_test_worker', '5', *g], env=env) for g in groups]
    codes = [p.wait() for p in procs]
    print(f"\n5 workers done in {(time.time() - t0) / 60:.0f} min, exit codes {codes}")
    filter_test_report()


def filter_test_worker(n, *jobs):
    from qbench import filtertest as T
    _threads(n)
    for j in jobs:
        m, c = json.loads(j)
        T.run_job(m, c, tag=f"[{m} {c}] ")


def filter_test_report():
    from qbench import filtertest as T
    path = T.report(qbench.config.WORK_DIR / 'reports' / 'filter_test.html')
    print(f"report -> {path}")
    _open(path)


# ---- QNN vs ANN under the filter protocol (qbench/qnnscan.py) ---------------
QNN_SCAN_WORKERS = [['qn/sel_L4'], ['qn/sel_L3'], ['qn/sel_L2'], ['qn/sel_L1'],
                    ['cn/dense_tanh', 'cn/dense_relu', 'cn/mlp5_tanh', 'cn/mlp5_relu']]


def _spawn_cpu(cmd, groups):
    import subprocess, time
    from qbench.qnn import FeatureData, verify_qnn
    verify_qnn()
    for c in qbench.config.CROPS:          # feature cache written here, workers only read it
        print(FeatureData(c), flush=True)
    t0 = time.time()
    env = {**os.environ, 'QBENCH_DEVICE': 'cpu'}
    procs = [subprocess.Popen([sys.executable, __file__, cmd, str(len(groups)), *g], env=env) for g in groups]
    codes = [p.wait() for p in procs]
    print(f"\n{len(procs)} workers done in {(time.time() - t0) / 60:.0f} min, exit codes {codes}")


def qnn_scan():
    """Stage 2 (validation only): each QNN / ANN model's best Stage 1 config,
    7 crops x seeds 100-109. Resumable. Then qnn_scan_report (freezes the winners)."""
    from qbench import qnnscan as S
    for m in S.MODELS:
        print(f"{m:<14} {S.best_recipe(m)}")
    _spawn_cpu('qnn_scan_worker', QNN_SCAN_WORKERS)
    qnn_scan_report()


def qnn_scan_worker(n, *models):
    from qbench import qnnscan as S
    _threads(n)
    S.run_stage2(models, tag=f"[{models[0]}] ")


def qnn_scan_report():
    from qbench import qnnscan as S
    t = S.table()
    if not len(t):
        sys.exit('no Stage 2 results yet')
    t.to_csv(qbench.config.RESULTS_DIR / 'qnn_scan.csv', index=False)
    print(t.to_string(index=False))
    path = qbench.config.WORK_DIR / 'reports' / 'qnn_scan.html'
    path.write_text('<!DOCTYPE html><html><head><meta charset="utf-8"><title>QNN / ANN scan</title><style>'
                    'body{font:14px system-ui,sans-serif;margin:24px}table{border-collapse:collapse}'
                    'th,td{border:1px solid #ccc;padding:4px 8px;text-align:right}th{background:#f2f2f2}</style>'
                    '</head><body><h1>QNN / ANN scan: validation only</h1><p>Mean last-5-epoch validation accuracy, '
                    'seeds 100-109, each model with its best of 12 configs (Cherry + Grape, seeds 100-102). '
                    'Test data was not used.</p>' + t.to_html(index=False, border=0) + '</body></html>',
                    encoding='utf-8')
    print(f"report -> {path}")
    try:
        fz = S.freeze()
        print(f"\nfrozen -> {qbench.config.RESULTS_DIR / S.FROZEN}\n{json.dumps(fz, indent=2)}")
    except RuntimeError as e:
        print(f"\nnot frozen yet: {e}")


def qnn_test():
    """QNN winner, ANN winner, qn/identity: frozen configs, 7 crops, seeds 0-9,
    TEST once, predictions + probabilities saved. Resumable. Then qnn_test_report."""
    from qbench import qnnscan as S
    fz = S.freeze()
    print(json.dumps(fz['models'], indent=2), flush=True)
    js = [json.dumps(j) for j in S.jobs()]
    _spawn_cpu('qnn_test_worker', [js[i::5] for i in range(5)])
    qnn_test_report()


def qnn_test_worker(n, *jobs):
    from qbench import qnnscan as S
    _threads(n)
    for j in jobs:
        m, c = json.loads(j)
        S.run_job(m, c, tag=f"[{m} {c}] ")


def qnn_test_report():
    from qbench import qnnscan as S
    path = S.report(qbench.config.WORK_DIR / 'reports' / 'qnn_test.html')
    print(f"report -> {path}")
    _open(path)


def ensemble():
    """B6+QNN vs Conv2x2+ANN (and PatchMLP+ANN): 50/50 probability average of the
    frozen test runs. Trains nothing. work/reports/ensemble_test.html."""
    from qbench import ensemble as E
    path = E.report(qbench.config.WORK_DIR / 'reports' / 'ensemble_test.html')
    print(f"report -> {path}")
    _open(path)


# ---- revision robustness re-runs (qbench/robust.py, 2026-11-19) -------------
ROBUST_COST = {'qn/sel_L4': 40, 'q/ry_cnot_B6': 20, 'qn/identity': 8}


def robust(n='5'):
    """A3 + A2: every frozen test model re-run with the test set traced each epoch
    (training unchanged, checked against the original runs), plus Conv2x2 / PatchMLP
    on amplitude-encoded patches. Resumable. Then robust_check."""
    import subprocess, time
    from qbench import robust as R
    from qbench.data import CropData
    from qbench.qnn import FeatureData
    print(json.dumps(R.freeze()['models'], indent=2), flush=True)
    for c in qbench.config.CROPS:          # caches written here, workers only read them
        CropData(c), FeatureData(c)
    js = sorted(R.jobs(), key=lambda j: -ROBUST_COST.get(j[0], 6))
    groups, load = [[] for _ in range(int(n))], [0] * int(n)
    for j in js:                           # greedy balance by expected cost
        i = load.index(min(load))
        groups[i].append(json.dumps(j))
        load[i] += ROBUST_COST.get(j[0], 6)
    t0 = time.time()
    env = {**os.environ, 'QBENCH_DEVICE': 'cpu'}
    procs = [subprocess.Popen([sys.executable, __file__, 'robust_worker', n, *g], env=env) for g in groups]
    codes = [p.wait() for p in procs]
    print(f"\n{n} workers done in {(time.time() - t0) / 60:.0f} min, exit codes {codes}")
    robust_check()


def robust_worker(n, *jobs):
    from qbench import robust as R
    _threads(n)
    for j in jobs:
        m, c = json.loads(j)
        R.run_job(m, c, tag=f"[{m} {c}] ")


def robust_check():
    from qbench import robust as R
    df = R.rows()
    n, bad = R.check_reproduction(df)
    print(f"{len(df)} traced runs; {n} compared with the original test runs, {len(bad)} differ")
    for b in bad[:20]:
        print('  DIFFERS', b)


def lr_ext(n='5'):
    """A4 follow-up (validation only): lr 0.002 and 0.02 for the image test models of
    both sides, Stage 1 protocol. Then lr_ext_report."""
    import subprocess, time
    from qbench import robust as R
    t0 = time.time()
    env = {**os.environ, 'QBENCH_DEVICE': 'cpu'}
    procs = [subprocess.Popen([sys.executable, __file__, 'lr_ext_worker', n, m], env=env) for m in R.LR_MODELS]
    codes = [p.wait() for p in procs]
    print(f"workers done in {(time.time() - t0) / 60:.0f} min, exit codes {codes}")
    lr_ext_report()


def lr_ext_worker(n, *models):
    from qbench import robust as R
    _threads(n)
    for m in models:
        R.lr_ext_job(m, tag=f"[{m}] ")


def lr_ext_report():
    from qbench import robust as R
    b = R.lr_ext_board()
    b.to_csv(qbench.config.RESULTS_DIR / 'revision_a4_lr_ext.csv', index=False)
    print(b.to_string())


EXPERIMENTS = {'lr_ext': lr_ext, 'lr_ext_worker': lr_ext_worker, 'lr_ext_report': lr_ext_report,
               'robust': robust, 'robust_worker': robust_worker, 'robust_check': robust_check,
               'ensemble': ensemble, 'qnn_scan': qnn_scan,'qnn_scan_worker': qnn_scan_worker,
               'qnn_scan_report': qnn_scan_report, 'qnn_test': qnn_test,
               'qnn_test_worker': qnn_test_worker, 'qnn_test_report': qnn_test_report,
               'filter_test': filter_test, 'filter_test_worker': filter_test_worker,
               'filter_test_report': filter_test_report,
               'filter_scan': filter_scan, 'filter_scan_worker': filter_scan_worker,
               'classical_scan': classical_scan, 'classical_scan_report': classical_scan_report,
               'filter_scan_report': filter_scan_report, 'qnn_matched': qnn_matched, 'pl_check': pl_check, 'router': router, 'final': final, 'endtoend': endtoend,
               'qnn_tune': qnn_tune, 'qnn_worker': qnn_worker, 'qnn_report': qnn_report,
               'qnn_final': qnn_final, 'qnn_rebuild': qnn_rebuild, 'qnn_final_worker': qnn_final_worker,
               'qnn_final_report': qnn_final_report,
               'repro': repro, 'report': report, 'tune': tune, 'tune_report': tune_report,
               'pipeline': pipeline, 'confirm': confirm, 'confirm_report': confirm_report,
               'arch': arch, 'arch100': arch100, 'arch_worker': arch_worker,
               'arch_report': arch_report,
               'fqc_test': fqc_test}
if __name__ == '__main__':
    EXPERIMENTS[sys.argv[1]](*sys.argv[2:])
