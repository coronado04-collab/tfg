"""
Enhanced Contrastive Loss with Same-Class Factor
==================================================
Extends the original contrastive loss to incorporate species class information
in addition to plate-level identity.

Original paper:
    - Genuine pairs:  same plate AND same species  (label=1)
    - Impostor pairs: different species             (label=0)

Enhancement:
    - Add a secondary signal: whether pairs share the same species class,
      even across different plates. This teaches the Siamese network to
      group colonies by species, not just by plate identity.

Usage:
    from enhanced_contrastive_loss import (
        EnhancedContrastiveLoss,
        generate_enhanced_pairs,
        EnhancedSiameseDataset,
    )
"""

import os
import json
import random
import itertools
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image


# ==============================================================================
# Enhanced Contrastive Loss
# ==============================================================================
class EnhancedContrastiveLoss(nn.Module):
    """
    Multi-factor contrastive loss combining plate identity and species class.

    Loss = alpha * L_plate + (1 - alpha) * L_class

    where:
        L_plate = original contrastive loss (same plate + species vs different species)
        L_class = contrastive loss on species class (same species vs different species,
                  regardless of plate)

    Parameters
    ----------
    margin : float
        Contrastive loss margin (default 1.0, same as original).
    alpha : float
        Weight for the plate-level loss (default 0.5).
        alpha=1.0 recovers the original loss.
        alpha=0.0 uses only the class-level loss.
    """

    def __init__(self, margin=1.0, alpha=0.5):
        super().__init__()
        self.margin = margin
        self.alpha  = alpha

    def forward(self, out1, out2, plate_label, class_label):
        """
        Parameters
        ----------
        out1, out2 : Tensor [B, D]
            Embeddings for each branch.
        plate_label : Tensor [B]
            1.0 if same plate AND same species, 0.0 otherwise.
        class_label : Tensor [B]
            1.0 if same species (any plate), 0.0 if different species.
        """
        d = torch.sqrt(torch.sum((out1 - out2) ** 2, dim=1) + 1e-8)

        # Plate-level loss (original formulation)
        loss_plate = self._contrastive(d, plate_label)

        # Class-level loss
        loss_class = self._contrastive(d, class_label)

        return self.alpha * loss_plate + (1.0 - self.alpha) * loss_class

    def _contrastive(self, d, label):
        """Standard contrastive loss given distances and binary labels."""
        pos = label * d ** 2
        neg = (1 - label) * torch.clamp(self.margin - d, min=0.0) ** 2
        return torch.mean(pos + neg) / 2.0


# ==============================================================================
# Backward-compatible wrapper: use with original 3-tuple batches
# ==============================================================================
class OriginalContrastiveLoss(nn.Module):
    """Original contrastive loss (unchanged from notebook, for reference)."""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        d   = torch.sqrt(torch.sum((out1 - out2) ** 2, dim=1) + 1e-8)
        pos = label * d ** 2
        neg = (1 - label) * torch.clamp(self.margin - d, min=0.0) ** 2
        return torch.mean(pos + neg) / 2.0


# ==============================================================================
# Enhanced Pair Generation
# ==============================================================================
def generate_enhanced_pairs(json_path, max_pairs_per_plate=50,
                             cross_plate_ratio=0.3,
                             restrict_to_plates=None):
    """
    Generate pairs with both plate-level and class-level labels.

    In addition to the original genuine/impostor pairs, this function
    generates cross-plate same-species pairs: colonies that share the
    same species but come from different plates.

    Parameters
    ----------
    json_path : str
        Path to metadata.json.
    max_pairs_per_plate : int
        Max genuine pairs per plate (same as original).
    cross_plate_ratio : float
        Fraction of additional cross-plate same-species pairs to add,
        relative to the number of genuine pairs.
    restrict_to_plates : set or None
        Restrict sampling to these plates only.

    Returns
    -------
    pairs : list of (filename1, filename2)
    plate_labels : list of float
        1.0 = same plate + same species, 0.0 otherwise.
    class_labels : list of float
        1.0 = same species (any plate), 0.0 = different species.
    """
    print(f'Generating enhanced pairs (max {max_pairs_per_plate} genuine/plate, '
          f'cross-plate ratio={cross_plate_ratio})...')

    with open(json_path, 'r') as f:
        data = json.load(f)
    samples = list(data['patch_list'].values())

    if restrict_to_plates is not None:
        samples = [s for s in samples if s['plate_n'] in restrict_to_plates]
        print(f'  Restricted to {len(restrict_to_plates)} plates '
              f'→ {len(samples)} colonies')

    plates         = defaultdict(list)
    species_groups = defaultdict(list)
    for s in samples:
        plates[s['plate_n']].append(s)
        species_groups[s['species']].append(s)

    # ── 1. Genuine pairs: same plate AND same species (plate_label=1, class_label=1)
    genuine_pairs = []
    for p_name, p_samples in plates.items():
        possible = [
            (p1['filename'], p2['filename'])
            for p1, p2 in itertools.combinations(p_samples, 2)
            if p1['species'] == p2['species']
        ]
        if len(possible) > max_pairs_per_plate:
            possible = random.sample(possible, max_pairs_per_plate)
        genuine_pairs.extend(possible)

    genuine_plate_labels = [1.0] * len(genuine_pairs)
    genuine_class_labels = [1.0] * len(genuine_pairs)
    print(f'  Genuine (same plate+species): {len(genuine_pairs)}')

    # ── 2. Cross-plate same-species pairs (plate_label=0, class_label=1)
    n_cross = int(len(genuine_pairs) * cross_plate_ratio)
    cross_pairs = []
    species_with_multi_plates = {
        sp: cols for sp, cols in species_groups.items()
        if len(set(c['plate_n'] for c in cols)) > 1
    }

    if species_with_multi_plates and n_cross > 0:
        sp_names_multi = list(species_with_multi_plates.keys())
        sp_counts_multi = [len(species_with_multi_plates[sp])
                           for sp in sp_names_multi]

        attempts = 0
        while len(cross_pairs) < n_cross and attempts < n_cross * 10:
            attempts += 1
            sp = random.choices(sp_names_multi, weights=sp_counts_multi, k=1)[0]
            cols = species_with_multi_plates[sp]
            c1, c2 = random.sample(cols, 2)
            if c1['plate_n'] != c2['plate_n']:
                cross_pairs.append((c1['filename'], c2['filename']))

    cross_plate_labels = [0.0] * len(cross_pairs)  # different plate
    cross_class_labels = [1.0] * len(cross_pairs)  # same species
    print(f'  Cross-plate same-species:     {len(cross_pairs)}')

    # ── 3. Impostor pairs: different species (plate_label=0, class_label=0)
    sp_names  = list(species_groups.keys())
    sp_counts = [len(species_groups[sp]) for sp in sp_names]

    target_impostor = len(genuine_pairs) + len(cross_pairs)
    impostor_pairs  = []
    while len(impostor_pairs) < target_impostor:
        sp1, sp2 = random.choices(sp_names, weights=sp_counts, k=2)
        if sp1 == sp2:
            continue
        img1 = random.choice(species_groups[sp1])['filename']
        img2 = random.choice(species_groups[sp2])['filename']
        impostor_pairs.append((img1, img2))

    impostor_plate_labels = [0.0] * len(impostor_pairs)
    impostor_class_labels = [0.0] * len(impostor_pairs)
    print(f'  Impostor (diff species):      {len(impostor_pairs)}')

    # ── Combine and shuffle
    all_pairs        = genuine_pairs + cross_pairs + impostor_pairs
    all_plate_labels = genuine_plate_labels + cross_plate_labels + impostor_plate_labels
    all_class_labels = genuine_class_labels + cross_class_labels + impostor_class_labels

    combined = list(zip(all_pairs, all_plate_labels, all_class_labels))
    random.shuffle(combined)
    pairs, plate_labels, class_labels = zip(*combined)

    n_pos_class = sum(1 for cl in class_labels if cl == 1.0)
    n_neg_class = sum(1 for cl in class_labels if cl == 0.0)
    print(f'  Total: {len(pairs)} pairs  '
          f'(class+ : {n_pos_class}, class- : {n_neg_class})')

    return list(pairs), list(plate_labels), list(class_labels)


# ==============================================================================
# Enhanced Siamese Dataset
# ==============================================================================
class EnhancedSiameseDataset(Dataset):
    """
    Dataset that returns (img1, img2, plate_label, class_label).

    Compatible with EnhancedContrastiveLoss.
    """

    def __init__(self, pairs, plate_labels, class_labels,
                 root_dir, transform=None):
        self.pairs        = pairs
        self.plate_labels = plate_labels
        self.class_labels = class_labels
        self.root_dir     = root_dir
        self.transform    = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        f1, f2 = self.pairs[idx]
        pl     = self.plate_labels[idx]
        cl     = self.class_labels[idx]
        img1   = Image.open(os.path.join(self.root_dir, f1)).convert('RGB')
        img2   = Image.open(os.path.join(self.root_dir, f2)).convert('RGB')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2,
                torch.tensor(pl, dtype=torch.float32),
                torch.tensor(cl, dtype=torch.float32))


# ==============================================================================
# Enhanced Siamese Training Loop
# ==============================================================================
def train_enhanced_siamese(model, train_loader, val_loader, num_epochs,
                            device, save_path, checkpoint_path,
                            margin=1.0, alpha=0.5, lr=0.001,
                            weight_decay=0.0005):
    """
    Training loop for the Siamese CNN with enhanced contrastive loss.

    Parameters
    ----------
    model : SiameseCNN
    train_loader, val_loader : DataLoader
        Must yield (img1, img2, plate_label, class_label) batches.
    alpha : float
        Weight for plate-level loss vs class-level loss.
    """
    from tqdm import tqdm

    model     = model.to(device)
    criterion = EnhancedContrastiveLoss(margin=margin, alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)

    start_epoch, best_val_loss = 0, float('inf')
    history = {'train_losses': [], 'val_losses': [],
               'train_accs':   [], 'val_accs':   []}

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        history       = ckpt.get('history', history)
        print(f'Resuming enhanced Siamese from epoch {start_epoch}')
    else:
        print('Starting enhanced Siamese from scratch.')

    for epoch in range(start_epoch, num_epochs):
        print(f'  [best so far: {best_val_loss:.4f}]')

        # ── Train ──
        model.train()
        t_loss, correct, total = 0.0, 0, 0
        for img1, img2, pl, cl in tqdm(train_loader,
                                        desc=f'Epoch {epoch+1}/{num_epochs}'):
            img1 = img1.to(device)
            img2 = img2.to(device)
            pl   = pl.to(device)
            cl   = cl.to(device)

            optimizer.zero_grad()
            o1, o2 = model(img1, img2)
            loss = criterion(o1, o2, pl, cl)
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            d = torch.sqrt(torch.sum((o1 - o2) ** 2, dim=1) + 1e-8)
            # Use class_label for accuracy (more meaningful than plate_label)
            correct += ((d < 0.5).float() == cl).sum().item()
            total   += cl.size(0)

        # ── Val ──
        model.eval()
        v_loss, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for img1, img2, pl, cl in val_loader:
                img1 = img1.to(device)
                img2 = img2.to(device)
                pl   = pl.to(device)
                cl   = cl.to(device)
                o1, o2 = model(img1, img2)
                loss = criterion(o1, o2, pl, cl)
                v_loss += loss.item()
                d2 = torch.sum((o1 - o2) ** 2, dim=1)
                vc += ((d2 < 0.5).float() == cl).sum().item()
                vt += cl.size(0)

        avg_tl = t_loss / max(len(train_loader), 1)
        avg_vl = v_loss / max(len(val_loader), 1)
        t_acc  = 100.0 * correct / max(total, 1)
        v_acc  = 100.0 * vc / max(vt, 1)

        history['train_losses'].append(avg_tl)
        history['val_losses'].append(avg_vl)
        history['train_accs'].append(t_acc)
        history['val_accs'].append(v_acc)

        print(f'  Train loss: {avg_tl:.4f} | acc: {t_acc:.1f}%')
        print(f'  Val   loss: {avg_vl:.4f} | acc: {v_acc:.1f}%')

        # Save checkpoint
        torch.save({
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch':                epoch,
            'best_val_loss':        best_val_loss,
            'history':              history,
        }, checkpoint_path)

        # Save best model
        if avg_vl < best_val_loss:
            best_val_loss = avg_vl
            torch.save(model.state_dict(), save_path)
            print(f'  ★ New best val loss: {best_val_loss:.4f}')

    return history
