"""Paths and fixed hyper-parameters, in one place.

Defaults are the Kaggle layout. Override any path with an environment
variable before importing qbench, e.g.
    os.environ['QBENCH_DATA'] = '/kaggle/input/other/PlantVillage'
"""
import os
from pathlib import Path

# ---- paths -----------------------------------------------------------------
DATA_DIR = Path(os.environ.get(
    'QBENCH_DATA', '/kaggle/input/cropleaves/PlantVillage_Processed_700 images'))
WORK_DIR = Path(os.environ.get('QBENCH_WORK', '/kaggle/working'))
RESULTS_DIR = WORK_DIR / 'results'        # every CSV the pipeline writes
CACHE_DIR = WORK_DIR / 'cache'            # preprocessed .npz per crop

# A previous Kaggle run's /kaggle/working, attached as an input dataset.
# results.restore() copies its CSVs back so runs resume instead of restarting.
RESTORE_DIR = Path(os.environ.get('QBENCH_RESTORE', '/kaggle/input/qbench-results'))

# ---- data ------------------------------------------------------------------
# Corn removed 2026-11-09: Corn_healthy images have no black background (the
# other class does), so the share of near-black pixels alone classifies Corn
# validation 100% correctly. Its results stay on disk but are not used.
CROPS = ['Apple', 'Cherry', 'Grape', 'Peach', 'Pepper', 'Potato', 'Tomato']
EXCLUDED_CROPS = ['Corn']
MAIN_CROPS = ['Cherry', 'Corn']
IMG_SIZE = 32

# ---- training (the recipe every earlier experiment used) -------------------
SEEDS = list(range(10))
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 0.005
