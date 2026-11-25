"""Result CSVs that survive Kaggle session resets.

Every finished run is appended to RESULTS_DIR/<name>.csv immediately, so a
crash loses at most the run in progress. Re-running the same experiment
skips (crop, model, recipe, seed, phase) combinations already on disk.

To carry results across sessions on Kaggle:
  1. at the end of a session: "Save Version" (Quick save is enough); the
     notebook's /kaggle/working becomes that version's output
  2. in the next session: Add Input -> your notebook's output, mounted as
     /kaggle/input/qbench-results (or set QBENCH_RESTORE)
  3. call results.restore() first -- it copies those CSVs back into place
"""
import shutil
import time

import pandas as pd

from . import config

KEY = ['crop', 'model', 'recipe', 'seed', 'phase']


def safe_write(write, path):
    """Call write(); if Windows says the file is locked (usually: open in
    Excel), wait and retry instead of crashing a long run."""
    warned = False
    while True:
        try:
            return write()
        except PermissionError:
            if not warned:
                print(f"\n  !! {path.name} is locked (open in Excel?). Close it -- "
                      f"the run continues automatically.", flush=True)
                warned = True
            time.sleep(5)


def restore(src=None, overwrite=False):
    """Copy result CSVs from a previous session's output into RESULTS_DIR."""
    src = config.RESTORE_DIR if src is None else src
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        print(f"restore: {src} not attached -- starting with no prior results")
        return []
    copied = []
    for f in sorted(src.rglob('*.csv')):
        dst = config.RESULTS_DIR / f.name
        if overwrite or not dst.exists():
            shutil.copy2(f, dst)
            copied.append(f.name)
    print(f"restore: copied {len(copied)} CSV(s) from {src}")
    return copied


class ResultStore:
    def __init__(self, name):
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.path = config.RESULTS_DIR / f'{name}.csv'
        self.curves_path = config.RESULTS_DIR / f'{name}_curves.csv'
        self.rows = pd.read_csv(self.path).to_dict('records') if self.path.exists() else []
        self._done = {tuple(r[k] for k in KEY) for r in self.rows}

    def done(self, crop, model, recipe, seed, phase):
        return (crop, model, recipe, seed, phase) in self._done

    def add(self, row, curve=None):
        self.rows.append(row)
        self._done.add(tuple(row[k] for k in KEY))
        safe_write(lambda: pd.DataFrame(self.rows).to_csv(self.path, index=False), self.path)
        if curve:
            c = pd.DataFrame(curve)
            for k in reversed(KEY):
                c.insert(0, k, row[k])
            safe_write(lambda: c.to_csv(self.curves_path, mode='a', index=False,
                                        header=not self.curves_path.exists()), self.curves_path)

    def df(self):
        return pd.DataFrame(self.rows)
