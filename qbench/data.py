"""Data loading.

The split is fixed by the dataset's own train/validation/test folders, so
there is nothing random to re-draw. The first load of a crop writes a
manifest (file path, label, split) next to the results; that file is the
record of exactly which images were in each split.

Preprocessing is identical to the original notebooks: grayscale, INTER_AREA
resize to 32x32, scale to [0, 1].
"""
import hashlib

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from . import config
from .results import safe_write

SPLITS = ('train', 'validation', 'test')
CRLF, LF = chr(13) + chr(10), chr(10)


def create_dataframes(base_dir, crop_name):
    """Scan <base>/<crop>/<class>/<split>/* into one DataFrame per split.

    Same ordering as the original cell (shuffle with random_state=42), so
    batch order -- and therefore every seed's result -- is unchanged.
    """
    data = {s: [] for s in SPLITS}
    crop_dir = base_dir / crop_name
    if not crop_dir.is_dir():
        raise FileNotFoundError(f"crop directory not found: {crop_dir}")
    for class_dir in crop_dir.iterdir():
        if not class_dir.is_dir():
            continue
        for set_dir in class_dir.iterdir():
            if set_dir.name in data:
                for img_path in set_dir.glob('*'):
                    data[set_dir.name].append((str(img_path), class_dir.name))
    out = {}
    for s in SPLITS:
        df = pd.DataFrame(data[s], columns=['Filepaths', 'Labels'])
        out[s] = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return out['train'], out['test'], out['validation']


def _load_images(df, label_map, size):
    images, labels = [], []
    for path, label in zip(df['Filepaths'], df['Labels']):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            images.append(img / 255.0)
            labels.append(label_map[label])
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)


def _write_manifest(crop, dfs):
    rows = []
    for split, df in dfs.items():
        for path, label in zip(df['Filepaths'], df['Labels']):
            rows.append((split, label, path))
    text = pd.DataFrame(rows, columns=['split', 'label', 'path']).to_csv(index=False)
    path = config.RESULTS_DIR / f'split_manifest_{crop}.csv'
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # rewrite only if the split changed (it normally never does); reading a
    # file Excel has open is fine, writing to it is not
    try:
        unchanged = (path.exists() and
                     path.read_text(encoding='utf-8') == text.replace(CRLF, LF))
    except PermissionError:
        unchanged = False
    if not unchanged:
        safe_write(lambda: path.write_text(text, encoding='utf-8', newline=''), path)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


class CropData:
    """All three splits of one crop, held in memory as float32 arrays.

    Attributes: X_train, y_train, X_val, y_val, X_test, y_test,
                label_map, num_classes, split_hash
    """

    def __init__(self, crop, size=config.IMG_SIZE, use_cache=True):
        self.crop = crop
        cache = config.CACHE_DIR / f'{crop}_{size}.npz'
        train_df, test_df, valid_df = create_dataframes(config.DATA_DIR, crop)
        self.split_hash = _write_manifest(
            crop, {'train': train_df, 'validation': valid_df, 'test': test_df})
        self.label_map = {l: i for i, l in enumerate(sorted(train_df['Labels'].unique()))}
        self.num_classes = len(self.label_map)

        if use_cache and cache.exists():
            z = np.load(cache, allow_pickle=False)
            if str(z['split_hash']) == self.split_hash:
                for k in ('X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'):
                    setattr(self, k, z[k])
                return

        self.X_train, self.y_train = _load_images(train_df, self.label_map, size)
        self.X_val, self.y_val = _load_images(valid_df, self.label_map, size)
        self.X_test, self.y_test = _load_images(test_df, self.label_map, size)
        if use_cache:
            config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(cache, X_train=self.X_train, y_train=self.y_train,
                     X_val=self.X_val, y_val=self.y_val,
                     X_test=self.X_test, y_test=self.y_test,
                     split_hash=self.split_hash)

    def __repr__(self):
        return (f"CropData({self.crop}: train={len(self.y_train)} "
                f"val={len(self.y_val)} test={len(self.y_test)} "
                f"classes={self.label_map} split={self.split_hash})")


class PreloadedDataset(Dataset):
    def __init__(self, images, labels):
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def make_loaders(d, seed, batch_size=config.BATCH_SIZE):
    """Train loader shuffled by a generator seeded with `seed`; val/test fixed.

    num_workers=0: the data is already in memory, and the shuffle order is
    set by the generator, so this gives the same batches as the old
    num_workers=2 loaders, only faster.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    train = DataLoader(PreloadedDataset(d.X_train, d.y_train),
                       batch_size=batch_size, shuffle=True, generator=g)
    val = DataLoader(PreloadedDataset(d.X_val, d.y_val),
                     batch_size=batch_size, shuffle=False)
    test = DataLoader(PreloadedDataset(d.X_test, d.y_test),
                      batch_size=batch_size, shuffle=False)
    return train, val, test
