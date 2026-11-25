"""One training loop for every model.

Model selection is on VALIDATION only: the epoch with the highest val
accuracy is kept (first one on ties), exactly as before. Test accuracy is
computed once, after selection, and only when phase='final'. In
phase='tune' the test set is never touched -- test_acc is left empty.

A recipe is a dict. Defaults reproduce the earlier notebooks:
    lr=0.005, epochs=20, batch_size=32, optimizer='adam', weight_decay=0,
    warmup_epochs=0, plus model options act / norm / frozen (see models.py)
"""
import hashlib
import json
import os
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from . import config
from .data import CropData, make_loaders
from .models import build_model, count_params
from .results import ResultStore

DEFAULT_RECIPE = {
    'lr': config.LR,
    'epochs': config.NUM_EPOCHS,
    'batch_size': config.BATCH_SIZE,
    'optimizer': 'adam',
    'weight_decay': 0.0,
    'warmup_epochs': 0,
}
MODEL_OPTS = ('act', 'norm', 'frozen')

# QBENCH_DEVICE=cpu|cuda overrides (lets two processes share CPU and GPU)
device = torch.device(os.environ.get('QBENCH_DEVICE')
                      or ('cuda' if torch.cuda.is_available() else 'cpu'))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def recipe_id(recipe):
    """Stable string for a recipe -- only the keys that differ from default."""
    diff = {k: v for k, v in sorted(recipe.items())
            if k in MODEL_OPTS or DEFAULT_RECIPE.get(k) != v}
    return json.dumps(diff, sort_keys=True) if diff else 'default'


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device)
        _, pred = torch.max(model(images).data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return 100.0 * correct / total


@torch.no_grad()
def predict(model, loader):
    """Per-image predictions, labels and softmax probabilities, in loader order."""
    model.eval()
    preds, ys, probs = [], [], []
    for images, labels in loader:
        out = model(images.to(device, dtype=torch.float32))
        preds.append(out.argmax(1).cpu().numpy())
        probs.append(torch.softmax(out, 1).cpu().numpy())
        ys.append(labels.numpy())
    return np.concatenate(preds), np.concatenate(ys), np.concatenate(probs)


def predictions_path(crop, model_name, rid, seed):
    h = hashlib.sha1(rid.encode()).hexdigest()[:8]
    return config.RESULTS_DIR / 'preds' / crop / f"{model_name.replace('/', '_')}__{h}__s{seed}.npz"


def _rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _set_rng_state(st):
    random.setstate(st[0])
    np.random.set_state(st[1])
    torch.set_rng_state(st[2])
    if st[3] is not None:
        torch.cuda.set_rng_state_all(st[3])


def fit(model, train_loader, val_loader, recipe, trace_loader=None):
    """Train; return (best_val, best_epoch, best_state, per-epoch history).

    trace_loader (added 2026-11-19, robustness check only): if given, the
    model is also evaluated on it after every epoch, with every random state
    restored afterwards, so training and selection are exactly unchanged.
    The per-epoch accuracy goes into history['trace_acc'], the softmax
    outputs into history['_trace_prob'] (not written to the curves CSV)."""
    params = [p for p in model.parameters() if p.requires_grad]
    if recipe['optimizer'] == 'adam':
        opt = torch.optim.Adam(params, lr=recipe['lr'], weight_decay=recipe['weight_decay'])
    elif recipe['optimizer'] == 'adamw':
        opt = torch.optim.AdamW(params, lr=recipe['lr'], weight_decay=recipe['weight_decay'])
    else:
        raise ValueError(f"unknown optimizer {recipe['optimizer']}")

    sched = None
    warm_steps = int(recipe['warmup_epochs'] * len(train_loader))
    if warm_steps > 0:
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda step: min(1.0, (step + 1) / warm_steps))

    criterion = nn.CrossEntropyLoss()
    best_val, best_epoch, best_state, history = -1.0, -1, None, []
    for epoch in range(recipe['epochs']):
        model.train()
        loss_sum = n = 0
        for images, labels in train_loader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device)
            opt.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            loss_sum += loss.item() * labels.size(0)
            n += labels.size(0)
        v = evaluate(model, val_loader)
        history.append({'epoch': epoch, 'train_loss': loss_sum / n, 'val_acc': v})
        if trace_loader is not None:
            st = _rng_state()
            tp, ty, tprob = predict(model, trace_loader)
            _set_rng_state(st)
            history[-1]['trace_acc'] = 100.0 * float((tp == ty).mean())
            history[-1]['_trace_prob'] = tprob
            history[-1]['_trace_y'] = ty
        if v > best_val:
            best_val, best_epoch, best_state = v, epoch, deepcopy(model.state_dict())
    return best_val, best_epoch, best_state, history


def trace_path(crop, model_name, rid, seed):
    """Per-epoch test outputs of a traced re-run (robustness check, 2026-11-19).
    Kept apart from predictions_path so the original test files are never touched."""
    h = hashlib.sha1(rid.encode()).hexdigest()[:8]
    return config.RESULTS_DIR / 'preds_trace' / crop / f"{model_name.replace('/', '_')}__{h}__s{seed}.npz"


def run_one(data, model_name, seed, recipe=None, phase='tune', trace_test=False):
    """Train one model on one crop with one seed. Returns (row, history).

    trace_test=True (phase 'final' only): also record test accuracy after every
    epoch (see fit), report the last-five-epoch test mean next to the usual
    best-validation-epoch number, and save the outputs to trace_path."""
    if phase not in ('tune', 'final'):
        raise ValueError("phase must be 'tune' or 'final'")
    if trace_test and phase != 'final':
        raise ValueError('trace_test needs phase="final"')
    r = {**DEFAULT_RECIPE, **(recipe or {})}
    model_opts = {k: r[k] for k in MODEL_OPTS if k in r}

    set_seed(seed)
    train_loader, val_loader, test_loader = make_loaders(data, seed, r['batch_size'])
    model = build_model(model_name, data.num_classes, seed=seed, **model_opts).to(device)
    counts = count_params(model)

    t0 = time.time()
    best_val, best_epoch, best_state, history = fit(
        model, train_loader, val_loader, r, trace_loader=test_loader if trace_test else None)
    model.load_state_dict(best_state)
    test_acc = float('nan')
    trace = {}
    if trace_test:
        probs = np.stack([h.pop('_trace_prob') for h in history])     # epochs x images x classes
        ys = [h.pop('_trace_y') for h in history]
        last5 = probs[-5:].mean(0)
        trace = {'test_last5': float(np.mean([h['trace_acc'] for h in history[-5:]])),
                 'test_last': history[-1]['trace_acc'],
                 'test_last5_probavg': 100.0 * float((last5.argmax(1) == ys[0]).mean())}
    if phase == 'final':                 # test once; keep every image's prediction
        pred, y, prob = predict(model, test_loader)
        test_acc = 100.0 * float((pred == y).mean())
        if trace_test:
            path = trace_path(data.crop, model_name, recipe_id(r), seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, pred=pred.astype(np.int16), y=y.astype(np.int16), split=data.split_hash,
                     prob=prob.astype(np.float32), prob_last5=last5.astype(np.float32),
                     acc_epochs=np.array([h['trace_acc'] for h in history], dtype=np.float32))
            path = None
        else:
            path = predictions_path(data.crop, model_name, recipe_id(r), seed)
    if phase == 'final' and path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, pred=pred.astype(np.int16), y=y.astype(np.int16), split=data.split_hash,
                 prob=prob.astype(np.float32))   # prob: for ensembles (added 2026-11-10)

    row = {'crop': data.crop, 'model': model_name, 'recipe': recipe_id(r),
           'seed': seed, 'phase': phase, **counts,
           'val_acc': best_val, 'best_epoch': best_epoch,
           'val_last5': float(np.mean([h['val_acc'] for h in history[-5:]])),
           'test_acc': test_acc, **trace,
           'seconds': round(time.time() - t0, 1), 'split': data.split_hash}
    if hasattr(model, 'extra'):          # model-specific diagnostics (e.g. a learned scale)
        row.update(model.extra())
    return row, history


def run(store_name, crops, models, seeds=None, recipe=None, phase='tune', verbose=True):
    """Every (crop, model, seed) combination, resumable. Returns a DataFrame.

    store_name : CSV name under RESULTS_DIR (no extension)
    models     : list of names, e.g. ['q/ry_cnot_B6', 'c/Conv2x2/amp']
    """
    seeds = config.SEEDS if seeds is None else seeds
    store = ResultStore(store_name)
    rid = recipe_id({**DEFAULT_RECIPE, **(recipe or {})})
    t_all = time.time()
    for crop in crops:
        todo = [(m, s) for m in models for s in seeds
                if not store.done(crop, m, rid, s, phase)]
        if not todo:
            if verbose:
                print(f"[{crop}] all {len(models) * len(seeds)} runs already done")
            continue
        data = CropData(crop)
        if verbose:
            print(f"\n##### {data}  recipe={rid}  phase={phase}")
        last = None
        for m, s in todo:
            if verbose and m != last:
                print(f"--- {m}")
                last = m
            row, hist = run_one(data, m, s, recipe, phase)
            store.add(row, hist)
            if verbose:
                tst = '' if phase == 'tune' else f" | TEST {row['test_acc']:6.2f}%"
                print(f"  seed {s:2d} | filt {row['filter_params']:2d} head "
                      f"{row['head_params']} | val {row['val_acc']:6.2f}% "
                      f"@ep{row['best_epoch']:2d}{tst} | {row['seconds']:5.1f}s")
    if verbose:
        print(f"\nwall time {(time.time() - t_all) / 60:.1f} min -> {store.path}")
    return store.df()
