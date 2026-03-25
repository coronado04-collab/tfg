"""
K-Fold Plate-Aware Cross-Validation
====================================
Replaces the single random train/val/test split with K-fold cross-validation
at the plate level, ensuring no data leakage across folds.

Usage:
    from kfold_cross_validation import PlateAwareKFold

    kfold = PlateAwareKFold(n_splits=5, seed=42)
    for fold, (tr_idx, va_idx, te_idx, tr_plates, va_plates, te_plates) in enumerate(
        kfold.split(full_dataset)
    ):
        train_subset = Subset(full_dataset, tr_idx)
        val_subset   = Subset(full_dataset, va_idx)
        test_subset  = Subset(full_dataset, te_idx)
        # ... train and evaluate ...
"""

import random
import numpy as np
from collections import defaultdict


class PlateAwareKFold:
    """
    K-Fold cross-validation that splits at the plate level.

    Each fold uses one partition as test, one as validation, and the rest
    as training — ensuring zero plate overlap between splits.

    Parameters
    ----------
    n_splits : int
        Number of folds (default 5).
    test_ratio : float
        Fraction of plates held out as a fixed test set before folding.
        If 0.0, all plates participate in cross-validation.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_splits=5, test_ratio=0.15, seed=42):
        self.n_splits   = n_splits
        self.test_ratio = test_ratio
        self.seed       = seed

    def split(self, dataset):
        """
        Yields (train_indices, val_indices, test_indices,
                train_plates, val_plates, test_plates)
        for each fold.

        Parameters
        ----------
        dataset : ColonyDataset
            Must have a `.metadata` attribute (list of dicts with 'plate_n').
        """
        rng = random.Random(self.seed)

        # Group indices by plate
        plate_to_indices = defaultdict(list)
        for i, item in enumerate(dataset.metadata):
            plate_to_indices[item['plate_n']].append(i)

        plates = list(plate_to_indices.keys())
        rng.shuffle(plates)

        # Hold out a fixed test set if requested
        n_test = int(len(plates) * self.test_ratio)
        if n_test > 0:
            test_plates_fixed = set(plates[:n_test])
            cv_plates = plates[n_test:]
        else:
            test_plates_fixed = set()
            cv_plates = plates[:]

        te_idx_fixed = [i for p in test_plates_fixed
                        for i in plate_to_indices[p]]

        # Split remaining plates into K folds
        fold_size = len(cv_plates) // self.n_splits
        folds = []
        for k in range(self.n_splits):
            start = k * fold_size
            if k == self.n_splits - 1:
                end = len(cv_plates)  # last fold gets remainder
            else:
                end = start + fold_size
            folds.append(cv_plates[start:end])

        for k in range(self.n_splits):
            val_plates_k   = set(folds[k])
            train_plates_k = set()
            for j in range(self.n_splits):
                if j != k:
                    train_plates_k.update(folds[j])

            tr_idx = [i for p in train_plates_k for i in plate_to_indices[p]]
            va_idx = [i for p in val_plates_k   for i in plate_to_indices[p]]

            # If we have a fixed test set, use it; otherwise use val fold as test too
            if test_plates_fixed:
                te_idx    = te_idx_fixed
                te_plates = test_plates_fixed
            else:
                te_idx    = va_idx
                te_plates = val_plates_k

            # Sanity checks
            assert len(set(tr_idx) & set(va_idx)) == 0, \
                f'Fold {k}: train/val plate overlap!'
            if test_plates_fixed:
                assert len(set(tr_idx) & set(te_idx)) == 0, \
                    f'Fold {k}: train/test plate overlap!'

            n_total = len(tr_idx) + len(va_idx) + (
                len(te_idx) if test_plates_fixed else 0)
            print(f'Fold {k+1}/{self.n_splits}:')
            print(f'  Train plates: {len(train_plates_k):4d}  '
                  f'→ {len(tr_idx):6d} colonies')
            print(f'  Val   plates: {len(val_plates_k):4d}  '
                  f'→ {len(va_idx):6d} colonies')
            if test_plates_fixed:
                print(f'  Test  plates: {len(te_plates):4d}  '
                      f'→ {len(te_idx):6d} colonies (fixed)')
            print()

            yield (tr_idx, va_idx, te_idx,
                   train_plates_k, val_plates_k, te_plates)

    def __repr__(self):
        return (f'PlateAwareKFold(n_splits={self.n_splits}, '
                f'test_ratio={self.test_ratio}, seed={self.seed})')


def aggregate_fold_results(fold_results):
    """
    Aggregate metrics from multiple folds.

    Parameters
    ----------
    fold_results : list of dict
        Each dict should contain metric names as keys and float values.
        E.g. [{'top1_l2': 0.80, 'top1_l3': 0.85, 'v_measure': 0.70}, ...]

    Returns
    -------
    dict with 'mean' and 'std' for each metric, plus per-fold values.
    """
    if not fold_results:
        return {}

    metrics = list(fold_results[0].keys())
    summary = {}
    for m in metrics:
        vals = [fr[m] for fr in fold_results if m in fr]
        summary[m] = {
            'mean':     np.mean(vals),
            'std':      np.std(vals),
            'per_fold': vals,
        }
    return summary


def print_cv_summary(summary):
    """Pretty-print cross-validation results."""
    print('=' * 60)
    print('CROSS-VALIDATION SUMMARY')
    print('=' * 60)
    for metric, stats in summary.items():
        print(f'  {metric:20s}: {stats["mean"]:.4f} ± {stats["std"]:.4f}')
        folds_str = ', '.join(f'{v:.4f}' for v in stats['per_fold'])
        print(f'  {"":20s}  [{folds_str}]')
    print('=' * 60)
