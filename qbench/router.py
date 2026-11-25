"""Stage 1 of the two-stage pipeline: ResNet-50 crop router.

Ported from crop-classifier.ipynb: ImageNet ResNet-50 with frozen weights,
global average pool, Dropout(0.5), Linear(2048, 8), Adam lr 0.001, 20 epochs,
batch 32, 224x224 colour. Same train/validation/test images as stage 2.

Three differences from the notebook, all on purpose:
  - PyTorch weights (IMAGENET1K_V1) instead of Keras; no torchvision needed
  - images get the ImageNet normalisation the weights were trained with (the
    notebook fed raw 0-255 pixels, without keras preprocess_input)
  - the epoch is chosen on validation, as for every other model (the notebook
    reported its last epoch)

The backbone is frozen, so its 2048 features are computed once per crop and
cached; training a head on them takes seconds.

Model name: rn/head (num_classes from the data).
    RouterData(8)   crop router, labels = crop index in config.CROPS
    RouterData(16)  one-stage reference, labels = 2 * crop index + disease
"""
import hashlib

import cv2
import numpy as np
import torch
import torch.nn as nn

from . import config
from .data import create_dataframes, _write_manifest
from .qnn import FeatNet

WEIGHTS_URL = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RECIPE = {'lr': 0.001, 'epochs': 20}


class _Bottleneck(nn.Module):
    def __init__(self, inp, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or inp != planes * 4:
            self.downsample = nn.Sequential(nn.Conv2d(inp, planes * 4, 1, stride, bias=False),
                                            nn.BatchNorm2d(planes * 4))

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        return self.relu(y + (x if self.downsample is None else self.downsample(x)))


class ResNet50(nn.Module):
    """Backbone only; state_dict names match the torchvision checkpoint."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._layer(64, 64, 3, 1)
        self.layer2 = self._layer(256, 128, 4, 2)
        self.layer3 = self._layer(512, 256, 6, 2)
        self.layer4 = self._layer(1024, 512, 3, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    @staticmethod
    def _layer(inp, planes, n, stride):
        blocks = [_Bottleneck(inp, planes, stride)]
        blocks += [_Bottleneck(planes * 4, planes) for _ in range(n - 1)]
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.avgpool(x).flatten(1)


def backbone(device, pretrained=True):
    net = ResNet50()
    if pretrained:
        sd = torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location='cpu', check_hash=True)
        sd = {k: v for k, v in sd.items() if not k.startswith('fc.')}
        net.load_state_dict(sd, strict=True)
    return net.to(device).eval()


def _read(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f'unreadable image: {path}')   # would break the per-image alignment
    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
    return ((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)


@torch.no_grad()
def _extract(net, df, device, batch=64):
    paths, out = list(df['Filepaths']), []
    for i in range(0, len(paths), batch):
        x = torch.from_numpy(np.stack([_read(p) for p in paths[i:i + batch]])).to(device)
        out.append(net(x).cpu().numpy())
    return np.concatenate(out)


def crop_features(crop, net, device):
    """{F_train, y_train, F_val, y_val, F_test, y_test} of one crop, cached.
    Image order = CropData order, y = its binary disease label."""
    train_df, test_df, valid_df = create_dataframes(config.DATA_DIR, crop)
    h = _write_manifest(crop, {'train': train_df, 'validation': valid_df, 'test': test_df})
    cache = config.CACHE_DIR / f'{crop}_resnet50.npz'
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z['split_hash']) == h:
            return {k: z[k] for k in z.files if k != 'split_hash'}, h
    lm = {l: i for i, l in enumerate(sorted(train_df['Labels'].unique()))}
    if len(lm) != 2:
        raise ValueError(f'{crop}: expected 2 classes, got {lm}')
    out = {}
    for s, df in (('train', train_df), ('val', valid_df), ('test', test_df)):
        out[f'F_{s}'] = _extract(net, df, device)
        out[f'y_{s}'] = df['Labels'].map(lm).to_numpy(dtype=np.int64)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, split_hash=h, **out)
    return out, h


class RouterData:
    """All crops pooled, in config.CROPS order. Works with make_loaders / run_one."""

    def __init__(self, num_classes=len(config.CROPS), pretrained=True):
        from .train import device
        crops = list(config.CROPS)
        if num_classes not in (len(crops), 2 * len(crops)):
            raise ValueError(f'num_classes must be {len(crops)} (router) or {2 * len(crops)} (one-stage reference)')
        self.crop, self.crops, self.num_classes = f'ALL{num_classes}', crops, num_classes
        net = backbone(device, pretrained)
        parts, hashes = [], []
        for c in crops:
            p, h = crop_features(c, net, device)
            parts.append(p)
            hashes.append(h)
        self.split_hash = hashlib.sha1(''.join(hashes).encode()).hexdigest()[:12]
        for attr in ('train', 'val', 'test'):
            X = np.concatenate([p[f'F_{attr}'] for p in parts]).astype(np.float32)
            y = np.concatenate([(2 * i + p[f'y_{attr}'] if num_classes == 2 * len(crops)
                                 else np.full(len(p[f'y_{attr}']), i))
                                for i, p in enumerate(parts)]).astype(np.int64)
            setattr(self, f'X_{attr}', X)
            setattr(self, f'y_{attr}', y)

    def __repr__(self):
        return (f"RouterData({self.crop}: train={len(self.y_train)} val={len(self.y_val)} "
                f"test={len(self.y_test)} split={self.split_hash})")


def build_router_model(name, num_classes):
    if name != 'rn/head':
        raise ValueError(f"bad router model name '{name}'")
    return FeatNet(nn.Dropout(0.5), nn.Linear(2048, num_classes))


def run_router(seeds=config.SEEDS, pretrained=True, store_name='router'):
    """Router and one-stage reference, each seed trains one head. Test once."""
    from .results import ResultStore
    from .train import DEFAULT_RECIPE, recipe_id, run_one
    store = ResultStore(store_name)
    rid = recipe_id({**DEFAULT_RECIPE, **RECIPE})
    for n in (len(config.CROPS), 2 * len(config.CROPS)):
        data = RouterData(n, pretrained)
        print(data, flush=True)
        for s in seeds:
            if store.done(data.crop, 'rn/head', rid, s, 'final'):
                continue
            row, hist = run_one(data, 'rn/head', s, RECIPE, 'final')
            store.add(row, hist)
            print(f"{data.crop} seed {s} | val {row['val_acc']:6.2f} @ep{row['best_epoch']:2d} "
                  f"| TEST {row['test_acc']:6.2f} | {row['seconds']:5.1f}s", flush=True)
    return store.df()
