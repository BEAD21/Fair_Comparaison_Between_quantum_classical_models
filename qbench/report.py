"""Turn a results CSV into one self-contained HTML page.

    from qbench.report import build
    build('repro', ref='q/ry_cnot_B6')      # -> work/reports/repro.html

Per crop: accuracy of every seed for every model (with mean and 95% CI),
the paired difference of each model against `ref` (by seed), and the
validation curves. Uses test_acc for phase='final' rows and val_acc for
phase='tune' rows (where test was never evaluated). No external libraries;
the page opens offline.
"""
import json

import numpy as np
import pandas as pd

from . import config
from .stats import ci95, paired

# categorical slots, fixed order (validated: light + dark, all-pairs for 3)
SLOTS_LIGHT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SLOTS_DARK = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767']


def _label(model, recipe):
    return model if recipe == 'default' else f'{model}  {recipe}'


def _payload(df, curves, ref, metric=None):
    df = df.copy()
    df['label'] = [_label(m, r) for m, r in zip(df['model'], df['recipe'])]
    order = list(dict.fromkeys(df['label']))            # first appearance = slot order
    if ref is not None and ref in order:
        order.remove(ref)
        order.insert(0, ref)
    ref = ref if ref in order else order[0]
    forced, crops = metric, []
    for crop in list(dict.fromkeys(df['crop'])):
        sub = df[df['crop'] == crop]
        metric = forced or ('test_acc' if sub['test_acc'].notna().all() else 'val_acc')
        models = []
        for lab in [l for l in order if l in set(sub['label'])]:
            g = sub[sub['label'] == lab].sort_values('seed')
            m, h = ci95(g[metric])
            entry = {
                'label': lab, 'slot': order.index(lab),
                'filter_params': int(g['filter_params'].iloc[0]),
                'total_params': int(g['total_params'].iloc[0]),
                'mean': m, 'n': len(g),
                # one seed -> no spread; JSON has no NaN
                'ci': None if pd.isna(h) else h,
                'sd': None if len(g) < 2 else float(g[metric].std(ddof=1)),
                'seeds': [{'seed': int(r.seed), 'value': float(getattr(r, metric)),
                           'val': float(r.val_acc), 'epoch': int(r.best_epoch)}
                          for r in g.itertuples()],
            }
            if lab != ref and ref in set(sub['label']):
                tmp = sub.assign(model=sub['label'], recipe='x')
                p = paired(tmp, ref, lab, crop, metric=metric)
                a = tmp[tmp['model'] == ref].set_index('seed')[metric]
                b = g.set_index('seed')[metric]
                common = a.index.intersection(b.index)
                entry['paired'] = {k: (None if pd.isna(v) else float(v)) for k, v in p.items()
                                   if k in ('diff', 'lo', 'hi', 'p_t', 'p_wilcoxon', 'n')}
                entry['paired']['seeds'] = [{'seed': int(s), 'value': float(a[s] - b[s])}
                                            for s in common]
            if curves is not None:
                c = curves[(curves['crop'] == crop) & (curves['label'] == lab)]
                c = c[c['seed'].isin(g['seed'])]
                if len(c):
                    mc = c.groupby('epoch')['val_acc'].mean()
                    entry['curve'] = [{'epoch': int(e), 'val': float(v)} for e, v in mc.items()]
            models.append(entry)
        crops.append({'crop': crop, 'metric': metric, 'models': models})
    return {'ref': ref, 'crops': crops}


def build(name, ref=None, title=None, out=None, metric=None):
    """Write RESULTS_DIR/../reports/<name>.html and return its path.
    metric: column to plot (default test_acc for final rows, else val_acc)."""
    df = pd.read_csv(config.RESULTS_DIR / f'{name}.csv')
    cpath = config.RESULTS_DIR / f'{name}_curves.csv'
    curves = pd.read_csv(cpath) if cpath.exists() else None
    if curves is not None:
        curves = curves.drop_duplicates(subset=['crop', 'model', 'recipe', 'seed', 'phase', 'epoch'],
                                        keep='last')
        curves['label'] = [_label(m, r) for m, r in zip(curves['model'], curves['recipe'])]
    data = _payload(df, curves, ref, metric)
    out = out or (config.WORK_DIR / 'reports' / f'{name}.html')
    out.parent.mkdir(parents=True, exist_ok=True)
    html = (_TEMPLATE
            .replace('__TITLE__', title or f'Results: {name}')
            .replace('__DATA__', json.dumps(data, allow_nan=False))
            .replace('__SLOTS_LIGHT__', ''.join(f'--s{i}:{c};' for i, c in enumerate(SLOTS_LIGHT)))
            .replace('__SLOTS_DARK__', ''.join(f'--s{i}:{c};' for i, c in enumerate(SLOTS_DARK))))
    out.write_text(html, encoding='utf-8')
    return out


FAMILY_SLOTS = {'quantum': 0, 'classical_raw': 1, 'classical_amp': 2}


def build_tuning(families, crops=None, top=8, out=None):
    """Tuning report: per crop, one row per family, one dot per trial
    (mean val accuracy over the tuning seeds); the tick marks the winner."""
    from html import escape
    from .tune import TUNE_SEEDS, leaderboard
    curves_all = {}
    for fam in families:
        cp = config.RESULTS_DIR / f'tune_{fam}_curves.csv'
        if cp.exists():
            curves_all[fam] = pd.read_csv(cp)
    if crops is None:
        crops = list(dict.fromkeys(pd.concat(
            [pd.read_csv(config.RESULTS_DIR / f'tune_{f}.csv') for f in families])['crop']))
    out_crops = []
    for crop in crops:
        models, tables = [], []
        for i, fam in enumerate(families):
            slot = FAMILY_SLOTS.get(fam, i)
            lb = leaderboard(fam, crop)
            if not len(lb):
                continue
            best = lb.iloc[0]
            entry = {
                'label': fam, 'slot': slot, 'filter_params': int(best['filter_params']),
                'total_params': int(best['filter_params']) + 258,
                'mean': float(best['val_last5']), 'ci': None, 'sd': float(lb['val_last5'].std(ddof=1)) if len(lb) > 1 else 0.0,
                'n': len(lb),
                'right': f"best {best['val_last5']:.2f}",
                'right2': f"{len(lb)} trials · median {lb['val_last5'].median():.2f}",
                'seeds': [{'seed': k, 'value': float(r.val_last5), 'val': float(r.val_last5), 'epoch': 0,
                           'tip': f"<b>{escape(r.model)}</b><br>{escape(r.recipe)}<br>last-5 val {r.val_last5:.2f}"
                                  f" · best-epoch val {r.val_mean:.2f} ± {r.val_sd:.2f}<br>filter {r.filter_params} params"}
                          for k, r in enumerate(lb.itertuples())],
            }
            entry['sublabel'] = f"winner: {best['model']}"
            c = curves_all.get(fam)
            if c is not None:
                c = c[(c['crop'] == crop) & (c['model'] == best['model']) &
                      (c['recipe'] == best['recipe']) & c['seed'].isin(TUNE_SEEDS)]
                c = c.drop_duplicates(subset=['seed', 'epoch'], keep='last')
                if len(c):
                    entry['curve'] = [{'epoch': int(e), 'val': float(v)}
                                      for e, v in c.groupby('epoch')['val_acc'].mean().items()]
            models.append(entry)
            rows = ''.join(
                f"<tr><td>{k + 1}</td><td>{escape(r.model)}</td><td>{escape(r.recipe)}</td>"
                f"<td>{r.filter_params}</td><td>{r.val_last5:.2f}</td><td>{r.val_mean:.2f}</td>"
                f"<td>{r.val_sd:.2f}</td><td>{r.best_epoch:.1f}</td></tr>"
                for k, r in enumerate(lb.head(top).itertuples()))
            tables.append(f"<details{' open' if i == 0 else ''}><summary>{fam}: top {top} of {len(lb)}</summary>"
                          f"<table><tr><th>#</th><th>model</th><th>recipe</th><th>filter</th>"
                          f"<th>last-5 val</th><th>best-epoch val</th><th>sd</th><th>best epoch</th></tr>{rows}</table></details>")
        out_crops.append({
            'crop': crop, 'metric': 'val_acc', 'models': models, 'tableHtml': ''.join(tables),
            'title1': '1. Every trial of the search',
            'subtitle1': f'Each dot is one configuration: validation accuracy averaged over its last 5 epochs '
                         f'and over seeds {TUNE_SEEDS} (the selection metric). The black tick marks the '
                         f'winner. Test set not used.',
            'title3': '2. Validation curve of each winner', 'subtitle3': 'mean over the tuning seeds',
        })
    data = {'ref': None, 'crops': out_crops,
            'intro': 'Phase 2 tuning. Every family gets the same number of trials and the same recipe '
                     'draws; only the filter differs. Hover a dot to see its configuration.'}
    # dotRows needs a sublabel field
    for C in out_crops:
        for m in C['models']:
            m.setdefault('sublabel', '')
    out = out or (config.WORK_DIR / 'reports' / 'tuning.html')
    out.parent.mkdir(parents=True, exist_ok=True)
    html = (_TEMPLATE.replace('__TITLE__', 'Phase 2: tuning')
            .replace('__DATA__', json.dumps(data, allow_nan=False))
            .replace('__SLOTS_LIGHT__', ''.join(f'--s{i}:{c};' for i, c in enumerate(SLOTS_LIGHT)))
            .replace('__SLOTS_DARK__', ''.join(f'--s{i}:{c};' for i, c in enumerate(SLOTS_DARK))))
    out.write_text(html, encoding='utf-8')
    return out


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;
 --muted:#898781;--grid:#e1e0d9;--border:rgba(11,11,11,.10);__SLOTS_LIGHT__}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
 --surface-1:#1a1a19;--text-primary:#f0efec;--text-secondary:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
 --border:rgba(255,255,255,.10);__SLOTS_DARK__}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--text-primary:#f0efec;
 --text-secondary:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--border:rgba(255,255,255,.10);__SLOTS_DARK__}
html,body{margin:0}
.viz-root{background:var(--surface-1);color:var(--text-primary);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;
 min-height:100vh;padding:28px 32px 60px;box-sizing:border-box}
h1{font-size:22px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 2px;padding-top:18px;border-top:1px solid var(--grid)}
h3{font-size:14px;margin:22px 0 2px}.sub{color:var(--text-secondary);margin:0 0 10px;max-width:820px}
.note{color:var(--muted);font-size:12px}
.row{display:flex;gap:10px;align-items:center;margin:8px 0 0}
button{font:inherit;font-size:12px;color:var(--text-secondary);background:transparent;border:1px solid var(--border);
 border-radius:6px;padding:3px 10px;cursor:pointer}
svg text{font:12px system-ui,-apple-system,"Segoe UI",sans-serif;fill:var(--text-secondary)}
svg .lbl{fill:var(--text-primary)}svg .mut{fill:var(--muted)}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:6px 0 2px;font-size:12px;color:var(--text-secondary)}
.legend span{display:inline-flex;align-items:center;gap:6px}.sw{width:10px;height:10px;border-radius:3px;display:inline-block}
table{border-collapse:collapse;font-size:13px;margin-top:8px;font-variant-numeric:tabular-nums}
th,td{padding:4px 12px 4px 0;text-align:right;border-bottom:1px solid var(--grid)}th{color:var(--text-secondary);font-weight:600}
td:first-child,th:first-child{text-align:left}
#tip{position:fixed;pointer-events:none;background:var(--surface-1);color:var(--text-primary);border:1px solid var(--border);
 box-shadow:0 2px 10px rgba(0,0,0,.15);border-radius:8px;padding:7px 10px;font-size:12px;display:none;z-index:9;
 font-variant-numeric:tabular-nums;white-space:nowrap}
details summary{cursor:pointer;color:var(--text-secondary);margin-top:10px}
</style></head>
<body><div class="viz-root" id="root">
<h1>__TITLE__</h1>
<p class="sub" id="intro"></p>
<div class="row"><button id="theme">Toggle dark / light</button></div>
<div id="out"></div><div id="tip"></div>
</div>
<script>
const D = __DATA__;
const NS = 'http://www.w3.org/2000/svg';
const tip = document.getElementById('tip');
const col = i => `var(--s${i % 8})`;
const f2 = v => (v == null || isNaN(v)) ? '–' : v.toFixed(2);
const fp = p => p == null ? '–' : (p < 0.001 ? '<0.001' : p.toFixed(3));
function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
function txt(parent, x, y, s, cls, anchor) { const t = el('text', {x, y, class: cls || '', 'text-anchor': anchor || 'start'}, parent);
  t.textContent = s; return t; }
function showTip(ev, html) { tip.innerHTML = html; tip.style.display = 'block';
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + w > innerWidth - 8) x = ev.clientX - w - 14; if (y + h > innerHeight - 8) y = ev.clientY - h - 14;
  tip.style.left = x + 'px'; tip.style.top = y + 'px'; }
function hideTip() { tip.style.display = 'none'; }
function niceTicks(lo, hi, n) { const span = hi - lo, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || 10 * mag;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6)); return out; }
function legend(parent, models) { const L = document.createElement('div'); L.className = 'legend';
  models.forEach(m => { const s = document.createElement('span');
    s.innerHTML = `<i class="sw" style="background:${col(m.slot)}"></i>${m.label}`; L.appendChild(s); });
  parent.appendChild(L); }

// ---------- dot plot: every seed, mean and 95% CI, one row per model
function dotRows(parent, rows, opts) {
  const W = 860, left = 250, right = 150, rowH = 60, top = 26, H = top + rows.length * rowH + 30;
  const svg = el('svg', {width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': opts.aria}, parent);
  const xs = rows.flatMap(r => r.points.map(p => p.value).concat([r.lo, r.hi])).filter(v => v != null);
  let lo = Math.min(...xs, opts.zero ? 0 : Infinity), hi = Math.max(...xs, opts.zero ? 0 : -Infinity);
  const pad = (hi - lo) * 0.06 || 1; lo -= pad; hi += pad;
  if (opts.cap100) hi = Math.min(hi, 100.5);
  const X = v => left + (v - lo) / (hi - lo) * (W - left - right);
  for (const t of niceTicks(lo, hi, 7)) {
    el('line', {x1: X(t), x2: X(t), y1: top - 6, y2: H - 26, stroke: 'var(--grid)', 'stroke-width': 1}, svg);
    txt(svg, X(t), H - 10, (opts.zero && t > 0 ? '+' : '') + t, 'mut', 'middle'); }
  if (opts.zero) { el('line', {x1: X(0), x2: X(0), y1: top - 6, y2: H - 26, stroke: 'var(--muted)', 'stroke-width': 1.5}, svg);
    txt(svg, X(0) - 8, top - 10, opts.negLabel, 'mut', 'end'); txt(svg, X(0) + 8, top - 10, opts.posLabel, 'mut', 'start'); }
  rows.forEach((r, i) => {
    const cy = top + i * rowH + rowH / 2;
    el('rect', {x: left - 236, y: cy - 6, width: 10, height: 10, rx: 3, fill: col(r.slot)}, svg);
    txt(svg, left - 220, cy + 4, r.label, 'lbl');
    if (r.sublabel) txt(svg, left - 220, cy + 18, r.sublabel, 'mut');
    if (r.lo != null) el('line', {x1: X(r.lo), x2: X(r.hi), y1: cy, y2: cy, stroke: 'var(--text-secondary)', 'stroke-width': 2, 'stroke-linecap': 'round'}, svg);
    // stack seeds that share a value so none is hidden behind another
    const same = {}; r.points.forEach(p => { const k = p.value.toFixed(3); same[k] = (same[k] || 0) + 1; });
    const seen = {};
    r.points.forEach(p => {
      const key = p.value.toFixed(3), j = (seen[key] = (seen[key] || 0) + 1) - 1;
      const jy = cy + (j - (same[key] - 1) / 2) * 7;
      const c = el('circle', {cx: X(p.value), cy: jy, r: 4.5, fill: col(r.slot), stroke: 'var(--surface-1)', 'stroke-width': 2}, svg);
      const hit = el('circle', {cx: X(p.value), cy: jy, r: 9, fill: 'transparent'}, svg);
      hit.addEventListener('mousemove', ev => { c.setAttribute('r', 6); showTip(ev, p.tip); });
      hit.addEventListener('mouseleave', () => { c.setAttribute('r', 4.5); hideTip(); }); });
    el('line', {x1: X(r.mean), x2: X(r.mean), y1: cy - 13, y2: cy + 13, stroke: 'var(--text-primary)', 'stroke-width': 2.5, 'stroke-linecap': 'round'}, svg);
    txt(svg, W - right + 14, cy + 4, r.right, 'lbl');
    if (r.right2) txt(svg, W - right + 14, cy + 18, r.right2, 'mut');
  });
}

// ---------- validation curves (overlay for <=3 models, small multiples beyond)
function curves(parent, models) {
  const ms = models.filter(m => m.curve); if (!ms.length) return;
  const groups = ms.length <= 3 ? [ms] : ms.map(m => [m]);
  const wrap = document.createElement('div'); wrap.style.cssText = 'display:flex;flex-wrap:wrap;gap:12px'; parent.appendChild(wrap);
  if (ms.length <= 3) legend(parent.insertBefore(document.createElement('div'), wrap), ms);
  const all = ms.flatMap(m => m.curve.map(c => c.val));
  let lo = Math.min(...all), hi = Math.max(...all); const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi = Math.min(100.5, hi + pad);
  groups.forEach(g => {
    const single = groups.length > 1, W = single ? 420 : 860, H = single ? 220 : 280, l = 46, r = single ? 16 : 190, t = single ? 26 : 14, b = 34;
    const svg = el('svg', {width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'validation accuracy per epoch'}, wrap);
    const E = Math.max(...g.flatMap(m => m.curve.map(c => c.epoch)));
    const X = e => l + e / Math.max(E, 1) * (W - l - r), Y = v => t + (hi - v) / (hi - lo) * (H - t - b);
    if (single) { el('rect', {x: 0, y: 4, width: 10, height: 10, rx: 3, fill: col(g[0].slot)}, svg); txt(svg, 16, 13, g[0].label, 'lbl'); }
    for (const v of niceTicks(lo, hi, 5)) { el('line', {x1: l, x2: W - r, y1: Y(v), y2: Y(v), stroke: 'var(--grid)'}, svg); txt(svg, l - 8, Y(v) + 4, v, 'mut', 'end'); }
    for (const e of niceTicks(0, E, 6)) txt(svg, X(e), H - 14, e, 'mut', 'middle');
    txt(svg, (l + W - r) / 2, H - 1, 'epoch', 'mut', 'middle');
    g.forEach(m => { el('path', {d: m.curve.map((c, k) => (k ? 'L' : 'M') + X(c.epoch) + ' ' + Y(c.val)).join(''),
      fill: 'none', stroke: col(m.slot), 'stroke-width': 2, 'stroke-linejoin': 'round'}, svg);
    });
    if (!single) {   // end labels, pushed apart so they never overlap
      const ends = g.map(m => { const c = m.curve[m.curve.length - 1]; return {m, x: X(c.epoch), y: Y(c.val), v: c.val}; })
                    .sort((a, b) => a.y - b.y);
      for (let k = 1; k < ends.length; k++) if (ends[k].y - ends[k - 1].y < 15) ends[k].y = ends[k - 1].y + 15;
      ends.forEach(e => txt(svg, e.x + 8, e.y + 4, `${e.m.label}: ${f2(e.v)}`, 'lbl')); }
    const cross = el('line', {y1: t, y2: H - b, stroke: 'var(--muted)', 'stroke-width': 1, visibility: 'hidden'}, svg);
    const dots = g.map(m => el('circle', {r: 4.5, fill: col(m.slot), stroke: 'var(--surface-1)', 'stroke-width': 2, visibility: 'hidden'}, svg));
    const hit = el('rect', {x: l, y: t, width: W - l - r, height: H - t - b, fill: 'transparent'}, svg);
    hit.addEventListener('mousemove', ev => { const bb = svg.getBoundingClientRect();
      const e = Math.round((ev.clientX - bb.left - l) / (W - l - r) * E); if (e < 0 || e > E) return;
      cross.setAttribute('x1', X(e)); cross.setAttribute('x2', X(e)); cross.setAttribute('visibility', 'visible');
      let html = `<b>epoch ${e}</b>`;
      g.forEach((m, k) => { const c = m.curve.find(c => c.epoch === e); if (!c) return;
        dots[k].setAttribute('cx', X(e)); dots[k].setAttribute('cy', Y(c.val)); dots[k].setAttribute('visibility', 'visible');
        html += `<br><i class="sw" style="background:${col(m.slot)}"></i> ${m.label}: ${f2(c.val)}%`; });
      showTip(ev, html); });
    hit.addEventListener('mouseleave', () => { cross.setAttribute('visibility', 'hidden'); dots.forEach(d => d.setAttribute('visibility', 'hidden')); hideTip(); });
  });
}

// ---------- page
const metricName = m => m === 'test_acc' ? 'test accuracy'
  : m === 'val_last5' ? 'validation accuracy, mean of last 5 epochs (confirmation round, test not evaluated)'
  : 'validation accuracy (tuning run, test not evaluated)';
document.getElementById('intro').textContent = D.intro ||
  `Each dot is one seed. The black tick is the mean, the grey bar its 95% confidence interval. ` +
  `Differences are paired by seed: positive means ${D.ref} did better. Hover any dot or curve for details.`;
const out = document.getElementById('out');
D.crops.forEach(C => {
  const sec = document.createElement('section'); out.appendChild(sec);
  sec.insertAdjacentHTML('beforeend', `<h2>${C.crop}</h2><h3>${C.title1 || '1. Accuracy per seed'}</h3>
    <p class="sub">${C.subtitle1 || metricName(C.metric) + ', one row per model'}</p>`);
  dotRows(sec, C.models.map(m => ({label: m.label, slot: m.slot, sublabel: m.sublabel || `filter ${m.filter_params} · total ${m.total_params} params`,
    mean: m.mean, lo: m.ci == null ? null : m.mean - m.ci, hi: m.ci == null ? null : m.mean + m.ci,
    right: m.right || `${f2(m.mean)} ± ${f2(m.ci)}`, right2: m.right2 || `sd ${f2(m.sd)} · n=${m.n}`,
    points: m.seeds.map(s => ({value: s.value, tip: s.tip || `<b>${m.label}</b><br>seed ${s.seed}: <b>${f2(s.value)}%</b><br>val ${f2(s.val)}% · best epoch ${s.epoch}`}))})),
    {aria: `${C.crop} accuracy per seed`, cap100: true});
  const pr = C.models.filter(m => m.paired);
  if (pr.length) {
    sec.insertAdjacentHTML('beforeend', `<h3>2. Paired difference: ${D.ref} minus each model</h3>
      <p class="sub">Same seed on both sides. If the grey bar crosses zero, the difference is not significant.</p>`);
    dotRows(sec, pr.map(m => ({label: m.label, slot: m.slot, mean: m.paired.diff, lo: m.paired.lo, hi: m.paired.hi,
      right: `${m.paired.diff >= 0 ? '+' : ''}${f2(m.paired.diff)} pp`, right2: `p=${fp(m.paired.p_t)} (t) · ${fp(m.paired.p_wilcoxon)} (W)`,
      points: m.paired.seeds.map(s => ({value: s.value, tip: `seed ${s.seed}: ${D.ref} ${s.value >= 0 ? 'ahead by' : 'behind by'} <b>${f2(Math.abs(s.value))} pp</b>`}))})),
      {aria: `${C.crop} paired differences`, zero: true, negLabel: '← other model better', posLabel: `${D.ref} better →`});
  }
  if (C.models.some(m => m.curve)) {
    sec.insertAdjacentHTML('beforeend', `<h3>${C.title3 || '3. Validation accuracy during training'}</h3><p class="sub">${C.subtitle3 || 'mean over seeds, per epoch'}</p>`);
    curves(sec, C.models);
  }
  if (C.tableHtml) { sec.insertAdjacentHTML('beforeend', C.tableHtml); return; }
  let t = `<details open><summary>Table</summary><table><tr><th>model</th><th>filter</th><th>total</th><th>mean</th><th>95% CI</th><th>sd</th><th>n</th>
    <th>diff vs ref</th><th>CI of diff</th><th>p (t)</th><th>p (Wilcoxon)</th></tr>`;
  C.models.forEach(m => { const p = m.paired;
    t += `<tr><td>${m.label}</td><td>${m.filter_params}</td><td>${m.total_params}</td><td>${f2(m.mean)}</td><td>± ${f2(m.ci)}</td><td>${f2(m.sd)}</td><td>${m.n}</td>
      <td>${p ? f2(p.diff) : 'ref'}</td><td>${p ? `[${f2(p.lo)}, ${f2(p.hi)}]` : ''}</td><td>${p ? fp(p.p_t) : ''}</td><td>${p ? fp(p.p_wilcoxon) : ''}</td></tr>`; });
  t += '</table></details>';
  const seeds = [...new Set(C.models.flatMap(m => m.seeds.map(s => s.seed)))].sort((a, b) => a - b);
  t += `<details><summary>Per-seed values</summary><table><tr><th>seed</th>${C.models.map(m => `<th>${m.label}</th>`).join('')}</tr>` +
    seeds.map(s => `<tr><td>${s}</td>${C.models.map(m => { const x = m.seeds.find(z => z.seed === s); return `<td>${x ? f2(x.value) : ''}</td>`; }).join('')}</tr>`).join('') +
    '</table></details>';
  sec.insertAdjacentHTML('beforeend', t);
});
document.getElementById('theme').onclick = () => { const r = document.documentElement;
  const dark = r.dataset.theme ? r.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  r.dataset.theme = dark ? 'light' : 'dark'; };
</script></body></html>
"""
