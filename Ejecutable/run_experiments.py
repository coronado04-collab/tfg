"""
Experiment Runner — Research Differentiation Improvements
==========================================================
This script orchestrates all four improvement experiments:

  1. K-Fold Cross-Validation  (plate-aware, replaces single split)
  2. Pretrained Backbones     (ResNet-18/34/50, VGG-11/16, SlimResNet)
  3. Alternative Clustering   (cosine MeanShift, Spectral cosine, Spectral Gaussian)
  4. Enhanced Contrastive Loss (plate + class labels for Siamese training)

Each experiment can be run independently. Results are saved to OUTPUT_DIR.

Usage (from Colab or command line):
    %run run_experiments.py              # runs all experiments
    python run_experiments.py --exp 2    # run only experiment 2
"""

import os
import sys
import json
import random
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from collections import defaultdict
from tqdm import tqdm

# ── Local imports (same directory) ──
from kfold_cross_validation import PlateAwareKFold, aggregate_fold_results, print_cv_summary
from pretrained_backbones import build_pretrained_model, get_pretrained_transform, SUPPORTED_BACKBONES
from clustering_methods import cluster_with_method, compare_clustering_methods, CLUSTERING_METHODS
from enhanced_contrastive_loss import (
    EnhancedContrastiveLoss,
    generate_enhanced_pairs,
    EnhancedSiameseDataset,
    train_enhanced_siamese,
)


# ==============================================================================
# CONFIG — update these paths for your environment
# ==============================================================================
# For Google Colab:
#   WORK_DIR   = '/content/data_lab'
#   OUTPUT_DIR = '/content/drive/MyDrive/TFG/resultados_entrenamiento'
# For local:
#   WORK_DIR   = '../Data'
#   OUTPUT_DIR = './results'

WORK_DIR   = os.environ.get('TFG_WORK_DIR',   '../Data')
OUTPUT_DIR = os.environ.get('TFG_OUTPUT_DIR',  './results')

METADATA_PATH = os.path.join(WORK_DIR, 'metadata.json')
DATASET_PATH  = os.path.join(WORK_DIR, 'Dataset')

GLOBAL_SEED = 42
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training hyperparams
BATCH_SIZE          = 64
NUM_EPOCHS_CNN      = 200
NUM_EPOCHS_SIAMESE  = 50
N_FOLDS             = 5

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)


# ==============================================================================
# Helper: import notebook classes (ColonyDataset, SingleColonyCNN, etc.)
# These are defined in the notebook; we provide minimal re-definitions here
# so the script can run standalone.
# ==============================================================================

# ── AdaptivePadding (from notebook cell 6) ──
class AdaptivePadding:
    def __init__(self, margin=20, fill=0):
        self.margin = margin
        self.fill   = fill

    def __call__(self, img):
        w, h     = img.size
        max_side = max(w, h)
        pad_w    = max_side - w + self.margin
        pad_h    = max_side - h + self.margin
        padding  = (pad_w // 2, pad_h // 2,
                    pad_w - pad_w // 2, pad_h - pad_h // 2)
        return transforms.functional.pad(img, padding, fill=self.fill)


# ── ColonyDataset (from notebook cell 8) ──
from PIL import Image

class ColonyDataset(torch.utils.data.Dataset):
    def __init__(self, metadata, root_dir, species_to_idx, transform=None):
        self.metadata       = metadata
        self.root_dir       = root_dir
        self.species_to_idx = species_to_idx
        self.transform      = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item  = self.metadata[idx]
        path  = os.path.join(self.root_dir, item['filename'])
        img   = Image.open(path).convert('RGB')
        label = self.species_to_idx[item['species']]
        if self.transform:
            img = self.transform(img)
        return img, label


# ── SingleColonyCNN (from notebook cell 6) ──
import torch.nn.functional as F

class SingleColonyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(3,   20,  kernel_size=5)
        self.conv2   = nn.Conv2d(20,  50,  kernel_size=5)
        self.conv3   = nn.Conv2d(50,  100, kernel_size=4)
        self.conv4   = nn.Conv2d(100, 200, kernel_size=4)
        self.lrn     = nn.LocalResponseNorm(size=5)
        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.75)
        self.fc1     = nn.Linear(200 * 5 * 5, 500)
        self.fc2     = nn.Linear(500, 32)

    def _conv_block(self, x, conv_layer):
        x = conv_layer(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.lrn(x)
        x = self.pool(x)
        return x

    def forward(self, x, _check_shape=False):
        x = self._conv_block(x, self.conv1)
        x = self._conv_block(x, self.conv2)
        x = self._conv_block(x, self.conv3)
        x = self._conv_block(x, self.conv4)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        return self.fc2(x)

    def get_pIDv(self, x):
        self.eval()
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)


# ── SiameseCNN (from notebook cell 22) ──
class SiameseCNN(nn.Module):
    def __init__(self, embedding_dim=15):
        super().__init__()
        self.conv1     = nn.Conv2d(3, 20, kernel_size=5)
        self.conv2     = nn.Conv2d(20, 50, kernel_size=5)
        self.prelu_c1  = nn.PReLU()
        self.prelu_c2  = nn.PReLU()
        self.prelu_fc1 = nn.PReLU()
        self.prelu_fc2 = nn.PReLU()
        self.pool      = nn.MaxPool2d(kernel_size=2, stride=1)
        self.fc1       = nn.Linear(50 * 119 * 119, 500)
        self.fc2       = nn.Linear(500, embedding_dim)
        self.dropout   = nn.Dropout(0.5)

    def forward_one(self, x):
        x = self.pool(self.prelu_c1(self.conv1(x)))
        x = self.prelu_c2(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.prelu_fc1(self.fc1(x)))
        x = self.prelu_fc2(self.fc2(x))
        return x

    def forward(self, x1, x2):
        return self.forward_one(x1), self.forward_one(x2)


# ── Weight init ──
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# ── LR Scheduler ──
class PaperLRScheduler:
    def __init__(self, optimizer, initial_lr=0.01,
                 decay_rate=0.9999, halve_at=50000):
        self.optimizer  = optimizer
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.halve_at   = halve_at
        self.iteration  = 0
        self.halved     = False
        for pg in optimizer.param_groups:
            pg['lr'] = initial_lr

    def step(self):
        self.iteration += 1
        lr = self.initial_lr * (self.decay_rate ** self.iteration)
        if self.iteration >= self.halve_at and not self.halved:
            lr /= 2.0
            self.halved = True
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr


# ── Inference helpers ──
TRANSFORM_INFERENCE = transforms.Compose([
    AdaptivePadding(margin=20),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def get_pIDv_batch(image_paths, model, device, batch_size=32):
    model.eval()
    all_pIDv = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        imgs  = torch.stack([
            TRANSFORM_INFERENCE(Image.open(p).convert('RGB')) for p in batch
        ]).to(device)
        with torch.no_grad():
            pIDv = torch.softmax(model(imgs), dim=1).cpu().numpy()
        all_pIDv.append(pIDv)
    return np.vstack(all_pIDv)


def get_embeddings_batch(image_paths, siamese_cnn, device, batch_size=32):
    siamese_cnn.eval()
    all_emb = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        imgs  = torch.stack([
            TRANSFORM_INFERENCE(Image.open(p).convert('RGB')) for p in batch
        ]).to(device)
        with torch.no_grad():
            emb = siamese_cnn.forward_one(imgs).cpu().numpy()
        all_emb.append(emb)
    return np.vstack(all_emb)


def smooth_pIDv(pIDv_matrix, cluster_labels):
    sIDv = np.zeros_like(pIDv_matrix)
    for cid in np.unique(cluster_labels):
        mask       = (cluster_labels == cid)
        mean_pIDv  = pIDv_matrix[mask].mean(axis=0)
        mean_pIDv /= mean_pIDv.sum() + 1e-9
        sIDv[mask] = mean_pIDv
    return sIDv


# ==============================================================================
# EXPERIMENT 1: K-Fold Cross-Validation
# ==============================================================================
def run_experiment_1_kfold(n_folds=N_FOLDS):
    """
    Run K-fold plate-aware cross-validation with the original SingleColonyCNN.
    Reports per-fold and aggregate metrics.
    """
    print('\n' + '=' * 70)
    print(f'EXPERIMENT 1: {n_folds}-Fold Plate-Aware Cross-Validation')
    print('=' * 70)

    exp_dir = os.path.join(OUTPUT_DIR, 'exp1_kfold')
    os.makedirs(exp_dir, exist_ok=True)

    # Load metadata
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    samples = list(meta['patch_list'].values())
    species = sorted(set(s['species'] for s in samples))
    sp2idx  = {sp: i for i, sp in enumerate(species)}

    t_train = transforms.Compose([
        AdaptivePadding(20), transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])
    t_val = transforms.Compose([
        AdaptivePadding(20), transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    full_ds = ColonyDataset(samples, DATASET_PATH, sp2idx, transform=None)

    kfold = PlateAwareKFold(n_splits=n_folds, test_ratio=0.15, seed=GLOBAL_SEED)
    fold_results = []

    for fold, (tr_idx, va_idx, te_idx, tr_pl, va_pl, te_pl) in enumerate(
        kfold.split(full_ds)
    ):
        print(f'\n--- Fold {fold + 1}/{n_folds} ---')
        fold_dir = os.path.join(exp_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        # Create subsets with appropriate transforms
        train_ds = ColonyDataset(
            [samples[i] for i in tr_idx], DATASET_PATH, sp2idx, t_train)
        val_ds = ColonyDataset(
            [samples[i] for i in va_idx], DATASET_PATH, sp2idx, t_val)
        test_ds = ColonyDataset(
            [samples[i] for i in te_idx], DATASET_PATH, sp2idx, t_val)

        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=2)
        test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=2)

        # Train
        model = SingleColonyCNN()
        model.apply(init_weights)
        model = model.to(DEVICE)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.SGD(model.parameters(), lr=0.01,
                               momentum=0.9, weight_decay=0.0005)
        scheduler = PaperLRScheduler(optimizer)

        best_val_acc = 0.0
        for epoch in range(NUM_EPOCHS_CNN):
            model.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(imgs), labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Validate
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    preds = model(imgs).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total   += labels.size(0)
            val_acc = 100.0 * correct / total

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'idx_to_species': {i: sp for sp, i in sp2idx.items()},
                }, os.path.join(fold_dir, 'best_model.pth'))

            if (epoch + 1) % 50 == 0:
                print(f'    Epoch {epoch+1}/{NUM_EPOCHS_CNN} | '
                      f'val acc: {val_acc:.2f}% (best: {best_val_acc:.2f}%)')

        # Test
        ckpt = torch.load(os.path.join(fold_dir, 'best_model.pth'),
                           map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        test_acc = 100.0 * correct / total

        print(f'  Fold {fold+1} | val: {best_val_acc:.2f}% | test: {test_acc:.2f}%')
        fold_results.append({
            'val_acc':  best_val_acc,
            'test_acc': test_acc,
        })

        # Free GPU
        del model
        torch.cuda.empty_cache()

    summary = aggregate_fold_results(fold_results)
    print_cv_summary(summary)

    with open(os.path.join(exp_dir, 'cv_results.json'), 'w') as f:
        # Convert numpy to native Python for JSON serialization
        json_summary = {}
        for k, v in summary.items():
            json_summary[k] = {
                'mean': float(v['mean']),
                'std':  float(v['std']),
                'per_fold': [float(x) for x in v['per_fold']],
            }
        json.dump(json_summary, f, indent=2)

    return summary


# ==============================================================================
# EXPERIMENT 2: Pretrained Backbones
# ==============================================================================
def run_experiment_2_backbones(backbones=None):
    """
    Train and evaluate pretrained backbones as Single Colony CNN replacements.
    """
    if backbones is None:
        backbones = ['resnet18', 'resnet34', 'resnet50', 'vgg11_bn', 'slim_resnet']

    print('\n' + '=' * 70)
    print(f'EXPERIMENT 2: Pretrained Backbones — {backbones}')
    print('=' * 70)

    exp_dir = os.path.join(OUTPUT_DIR, 'exp2_backbones')
    os.makedirs(exp_dir, exist_ok=True)

    # Load metadata
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    samples = list(meta['patch_list'].values())
    species = sorted(set(s['species'] for s in samples))
    sp2idx  = {sp: i for i, sp in enumerate(species)}
    n_classes = len(species)

    results = {}

    for backbone_name in backbones:
        print(f'\n--- Backbone: {backbone_name} ---')
        bb_dir = os.path.join(exp_dir, backbone_name)
        os.makedirs(bb_dir, exist_ok=True)

        # Get transforms (with ImageNet normalization for pretrained models)
        use_pretrained = backbone_name != 'slim_resnet'
        if use_pretrained:
            t_train = get_pretrained_transform(backbone_name, is_train=True)
            t_val   = get_pretrained_transform(backbone_name, is_train=False)
        else:
            t_train = transforms.Compose([
                AdaptivePadding(20), transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ])
            t_val = transforms.Compose([
                AdaptivePadding(20), transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])

        # Use single split for backbone comparison (same split as original)
        full_ds = ColonyDataset(samples, DATASET_PATH, sp2idx, transform=None)
        from kfold_cross_validation import PlateAwareKFold
        # Use just the first fold to get the standard split
        kfold = PlateAwareKFold(n_splits=5, test_ratio=0.15, seed=GLOBAL_SEED)
        tr_idx, va_idx, te_idx, _, _, _ = next(iter(kfold.split(full_ds)))

        train_ds = ColonyDataset(
            [samples[i] for i in tr_idx], DATASET_PATH, sp2idx, t_train)
        val_ds = ColonyDataset(
            [samples[i] for i in va_idx], DATASET_PATH, sp2idx, t_val)
        test_ds = ColonyDataset(
            [samples[i] for i in te_idx], DATASET_PATH, sp2idx, t_val)

        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=2)
        test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=2)

        # Build model
        model = build_pretrained_model(
            backbone_name, n_classes=n_classes,
            pretrained=use_pretrained, dropout=0.5,
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        # Use smaller LR for pretrained (fine-tuning)
        lr = 0.001 if use_pretrained else 0.01
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)

        best_val_acc = 0.0
        history = {'train_losses': [], 'val_losses': [],
                   'train_accs': [], 'val_accs': []}

        for epoch in range(NUM_EPOCHS_CNN):
            # Train
            model.train()
            t_loss, correct, total = 0.0, 0, 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                t_loss  += loss.item()
                correct += (out.argmax(1) == labels).sum().item()
                total   += labels.size(0)

            # Validate
            model.eval()
            v_loss, vc, vt = 0.0, 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    out = model(imgs)
                    v_loss += criterion(out, labels).item()
                    vc     += (out.argmax(1) == labels).sum().item()
                    vt     += labels.size(0)

            train_acc = 100.0 * correct / total
            val_acc   = 100.0 * vc / vt
            history['train_losses'].append(t_loss / len(train_loader))
            history['val_losses'].append(v_loss / len(val_loader))
            history['train_accs'].append(train_acc)
            history['val_accs'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'backbone': backbone_name,
                    'idx_to_species': {i: sp for sp, i in sp2idx.items()},
                }, os.path.join(bb_dir, 'best_model.pth'))

            if (epoch + 1) % 50 == 0:
                print(f'    Epoch {epoch+1}/{NUM_EPOCHS_CNN} | '
                      f'val: {val_acc:.2f}% (best: {best_val_acc:.2f}%)')

        # Test
        ckpt = torch.load(os.path.join(bb_dir, 'best_model.pth'),
                           map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                correct += (model(imgs).argmax(1) == labels).sum().item()
                total   += labels.size(0)
        test_acc = 100.0 * correct / total

        print(f'  {backbone_name} | val: {best_val_acc:.2f}% | test: {test_acc:.2f}%')
        results[backbone_name] = {
            'val_acc':  best_val_acc,
            'test_acc': test_acc,
            'history':  history,
        }

        with open(os.path.join(bb_dir, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)

        del model
        torch.cuda.empty_cache()

    # Summary table
    print('\n' + '=' * 50)
    print(f'{"Backbone":20s} {"Val Acc":>10s} {"Test Acc":>10s}')
    print('-' * 50)
    for name, res in results.items():
        print(f'{name:20s} {res["val_acc"]:10.2f}% {res["test_acc"]:10.2f}%')
    print('=' * 50)

    with open(os.path.join(exp_dir, 'backbone_results.json'), 'w') as f:
        json.dump({k: {'val_acc': v['val_acc'], 'test_acc': v['test_acc']}
                   for k, v in results.items()}, f, indent=2)

    return results


# ==============================================================================
# EXPERIMENT 3: Clustering Methods Comparison
# ==============================================================================
def run_experiment_3_clustering(model_path=None, siamese_path=None):
    """
    Compare clustering methods on validation-set embeddings.
    Requires trained SingleColonyCNN and SiameseCNN models.
    """
    print('\n' + '=' * 70)
    print('EXPERIMENT 3: Clustering Methods Comparison')
    print('=' * 70)

    exp_dir = os.path.join(OUTPUT_DIR, 'exp3_clustering')
    os.makedirs(exp_dir, exist_ok=True)

    if model_path is None:
        model_path = os.path.join(OUTPUT_DIR, 'best_single_colony_model.pth')
    if siamese_path is None:
        siamese_path = os.path.join(OUTPUT_DIR, 'best_siamese_model.pth')

    # Load models
    ckpt_cnn = torch.load(model_path, map_location=DEVICE)
    idx2sp   = ckpt_cnn['idx_to_species']
    sp2idx   = {v: k for k, v in idx2sp.items()}

    single_cnn = SingleColonyCNN().to(DEVICE)
    single_cnn.load_state_dict(ckpt_cnn['model_state_dict'])
    single_cnn.eval()

    ckpt_siam   = torch.load(siamese_path, map_location=DEVICE)
    siamese_cnn = SiameseCNN(embedding_dim=15).to(DEVICE)
    siamese_cnn.load_state_dict(ckpt_siam)
    siamese_cnn.eval()

    # Load metadata and get validation plates
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    samples = list(meta['patch_list'].values())
    full_ds = ColonyDataset(samples, DATASET_PATH, sp2idx, transform=None)

    kfold = PlateAwareKFold(n_splits=5, test_ratio=0.15, seed=GLOBAL_SEED)
    _, va_idx, te_idx, _, va_plates, te_plates = next(iter(kfold.split(full_ds)))

    # Use test plates for evaluation
    plate_to_colonies = defaultdict(list)
    for fname, info in meta['patch_list'].items():
        if info['plate_n'] in te_plates and info['species'] in sp2idx:
            plate_to_colonies[info['plate_n']].append({
                'path':   os.path.join(DATASET_PATH, info['filename']),
                'gt_idx': sp2idx[info['species']],
            })

    # Run all clustering methods per plate
    methods = list(CLUSTERING_METHODS.keys())
    method_results = {m: {'correct_l3': 0, 'total': 0} for m in methods}
    method_results['baseline_l2'] = {'correct': 0, 'total': 0}

    print(f'\nEvaluating {len(methods)} methods on {len(plate_to_colonies)} test plates...')

    for pid, cols in tqdm(plate_to_colonies.items(), desc='Test plates'):
        if len(cols) < 2:
            continue

        paths   = [c['path'] for c in cols]
        gt      = np.array([c['gt_idx'] for c in cols])
        pIDv    = get_pIDv_batch(paths, single_cnn, DEVICE)
        embs    = get_embeddings_batch(paths, siamese_cnn, DEVICE)
        l2_pred = np.argmax(pIDv, axis=1)

        method_results['baseline_l2']['correct'] += (l2_pred == gt).sum()
        method_results['baseline_l2']['total']   += len(gt)

        for method in methods:
            try:
                labels = cluster_with_method(embs, method=method)
                sIDv   = smooth_pIDv(pIDv, labels)
                l3_pred = np.argmax(sIDv, axis=1)
                method_results[method]['correct_l3'] += (l3_pred == gt).sum()
                method_results[method]['total']      += len(gt)
            except Exception as e:
                print(f'    {method} failed on plate {pid}: {e}')

    # Print results
    print('\n' + '=' * 60)
    print(f'{"Method":30s} {"Top-1 Acc":>10s}')
    print('-' * 60)

    bl = method_results['baseline_l2']
    l2_acc = 100.0 * bl['correct'] / max(bl['total'], 1)
    print(f'{"L2 Baseline (no clustering)":30s} {l2_acc:10.2f}%')

    summary = {'baseline_l2': l2_acc}
    for method in methods:
        r = method_results[method]
        acc = 100.0 * r['correct_l3'] / max(r['total'], 1)
        delta = acc - l2_acc
        print(f'{CLUSTERING_METHODS[method]:30s} {acc:10.2f}% ({delta:+.2f})')
        summary[method] = acc

    print('=' * 60)

    with open(os.path.join(exp_dir, 'clustering_comparison.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


# ==============================================================================
# EXPERIMENT 4: Enhanced Contrastive Loss
# ==============================================================================
def run_experiment_4_enhanced_siamese(alphas=None):
    """
    Train Siamese CNN with enhanced contrastive loss at different alpha values.
    """
    if alphas is None:
        alphas = [1.0, 0.7, 0.5, 0.3]  # 1.0 = original, others = enhanced

    print('\n' + '=' * 70)
    print(f'EXPERIMENT 4: Enhanced Contrastive Loss — alphas={alphas}')
    print('=' * 70)

    exp_dir = os.path.join(OUTPUT_DIR, 'exp4_enhanced_siamese')
    os.makedirs(exp_dir, exist_ok=True)

    # Load metadata
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    samples = list(meta['patch_list'].values())
    species = sorted(set(s['species'] for s in samples))
    sp2idx  = {sp: i for i, sp in enumerate(species)}

    full_ds = ColonyDataset(samples, DATASET_PATH, sp2idx, transform=None)
    kfold = PlateAwareKFold(n_splits=5, test_ratio=0.15, seed=GLOBAL_SEED)
    tr_idx, va_idx, te_idx, tr_plates, va_plates, te_plates = next(
        iter(kfold.split(full_ds)))

    transform = transforms.Compose([
        AdaptivePadding(20), transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    results = {}

    for alpha in alphas:
        print(f'\n--- Alpha = {alpha} ---')
        alpha_dir = os.path.join(exp_dir, f'alpha_{alpha}')
        os.makedirs(alpha_dir, exist_ok=True)

        # Generate enhanced pairs
        pairs, plate_labels, class_labels = generate_enhanced_pairs(
            METADATA_PATH, max_pairs_per_plate=50,
            cross_plate_ratio=0.3 if alpha < 1.0 else 0.0,
            restrict_to_plates=tr_plates,
        )
        va_pairs, va_pl, va_cl = generate_enhanced_pairs(
            METADATA_PATH, max_pairs_per_plate=50,
            cross_plate_ratio=0.3 if alpha < 1.0 else 0.0,
            restrict_to_plates=va_plates,
        )

        tr_ds = EnhancedSiameseDataset(
            pairs, plate_labels, class_labels, DATASET_PATH, transform)
        va_ds = EnhancedSiameseDataset(
            va_pairs, va_pl, va_cl, DATASET_PATH, transform)

        tr_loader = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, num_workers=2)
        va_loader = DataLoader(va_ds, BATCH_SIZE, shuffle=False, num_workers=2)

        model = SiameseCNN(embedding_dim=15)
        model.apply(init_weights)

        history = train_enhanced_siamese(
            model, tr_loader, va_loader,
            num_epochs=NUM_EPOCHS_SIAMESE,
            device=DEVICE,
            save_path=os.path.join(alpha_dir, 'best_siamese.pth'),
            checkpoint_path=os.path.join(alpha_dir, 'siamese_ckpt.pth'),
            margin=1.0, alpha=alpha,
        )

        results[str(alpha)] = {
            'final_train_loss': history['train_losses'][-1],
            'final_val_loss':   history['val_losses'][-1],
            'final_val_acc':    history['val_accs'][-1],
        }

        with open(os.path.join(alpha_dir, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)

        del model
        torch.cuda.empty_cache()

    # Summary
    print('\n' + '=' * 60)
    print(f'{"Alpha":>8s} {"Train Loss":>12s} {"Val Loss":>12s} {"Val Acc":>10s}')
    print('-' * 60)
    for alpha_str, res in results.items():
        print(f'{alpha_str:>8s} {res["final_train_loss"]:12.4f} '
              f'{res["final_val_loss"]:12.4f} {res["final_val_acc"]:10.2f}%')
    print('=' * 60)

    with open(os.path.join(exp_dir, 'enhanced_siamese_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Run research differentiation experiments')
    parser.add_argument('--exp', type=int, default=0,
                        help='Experiment number (1-4). 0 = run all.')
    parser.add_argument('--n-folds', type=int, default=N_FOLDS,
                        help=f'Number of CV folds (default {N_FOLDS})')
    parser.add_argument('--backbones', nargs='+', default=None,
                        help='Backbone names for exp 2')
    parser.add_argument('--alphas', nargs='+', type=float, default=None,
                        help='Alpha values for exp 4')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'Device: {DEVICE}')
    print(f'Output: {OUTPUT_DIR}')

    if args.exp in (0, 1):
        run_experiment_1_kfold(n_folds=args.n_folds)
    if args.exp in (0, 2):
        run_experiment_2_backbones(backbones=args.backbones)
    if args.exp in (0, 3):
        run_experiment_3_clustering()
    if args.exp in (0, 4):
        run_experiment_4_enhanced_siamese(alphas=args.alphas)

    print('\nAll experiments complete.')


if __name__ == '__main__':
    main()
