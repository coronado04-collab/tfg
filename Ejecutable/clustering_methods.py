"""
Alternative Clustering Methods for Level 3
============================================
Replaces / augments Mean Shift with:
  1. Spectral Clustering with cosine distance matrix
  2. Mean Shift with cosine distance (via cosine-transformed embeddings)
  3. Gaussian (RBF) Kernel-based Spectral Clustering

All functions follow the same interface as `cluster_with_meanshift`:
    Input:  embeddings [M, D]
    Output: cluster_labels [M] (integer array)

Usage:
    from clustering_methods import (
        cluster_spectral_cosine,
        cluster_meanshift_cosine,
        cluster_spectral_gaussian,
        cluster_with_method,       # unified dispatcher
    )

    labels = cluster_with_method(embeddings, method='spectral_cosine', n_clusters=3)
"""

import numpy as np
from sklearn.cluster import MeanShift, SpectralClustering, estimate_bandwidth
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.preprocessing import normalize


# ==============================================================================
# 1. Spectral Clustering with Cosine Distance
# ==============================================================================
def cluster_spectral_cosine(embeddings, n_clusters=None, max_clusters=10):
    """
    Spectral Clustering using a cosine similarity affinity matrix.

    Parameters
    ----------
    embeddings : ndarray [M, D]
        Colony embeddings from the Siamese CNN.
    n_clusters : int or None
        Number of clusters. If None, automatically selected via
        the eigengap heuristic on the graph Laplacian.
    max_clusters : int
        Maximum number of clusters to consider for auto-selection.

    Returns
    -------
    labels : ndarray [M] of int
    """
    M = len(embeddings)
    if M <= 1:
        return np.zeros(M, dtype=int)

    # Build cosine similarity matrix (values in [-1, 1])
    sim = cosine_similarity(embeddings)
    # Shift to [0, 1] for a valid affinity matrix
    affinity = (sim + 1.0) / 2.0
    np.fill_diagonal(affinity, 1.0)

    if n_clusters is None:
        n_clusters = _estimate_n_clusters_eigengap(affinity, max_clusters)

    n_clusters = min(n_clusters, M)

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42,
        n_init=10,
    )
    labels = sc.fit_predict(affinity)

    n = len(np.unique(labels))
    print(f'    Spectral (cosine): {M} colonies → {n} cluster(s)')
    return labels


# ==============================================================================
# 2. Mean Shift with Cosine Distance
# ==============================================================================
def cluster_meanshift_cosine(embeddings, bandwidth=None, quantile=0.3):
    """
    Mean Shift on L2-normalized embeddings, effectively using cosine distance.

    Normalizing embeddings to unit length maps them onto the hypersphere.
    Euclidean distance on the unit sphere is monotonically related to
    cosine distance: ||a - b||² = 2(1 - cos(a, b)).

    Parameters
    ----------
    embeddings : ndarray [M, D]
    bandwidth : float or None
        If None, estimated automatically.
    quantile : float
        Quantile for bandwidth estimation.

    Returns
    -------
    labels : ndarray [M] of int
    """
    M = len(embeddings)
    if M <= 1:
        return np.zeros(M, dtype=int)

    # L2-normalize → unit sphere → Euclidean ≈ cosine
    normed = normalize(embeddings, norm='l2')

    if bandwidth is None:
        bandwidth = estimate_bandwidth(
            normed, quantile=quantile,
            n_samples=min(M, 500),
        )
        bandwidth = max(bandwidth, 1e-3)

    for bin_seeding in (True, False):
        try:
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding,
                           cluster_all=True)
            ms.fit(normed)
            break
        except ValueError:
            if not bin_seeding:
                print(f'    MeanShift (cosine): failed at bw={bandwidth:.4f}, '
                      f'returning 1 cluster')
                return np.zeros(M, dtype=int)

    n = len(np.unique(ms.labels_))
    print(f'    MeanShift (cosine): {M} colonies → {n} cluster(s) '
          f'(bw={bandwidth:.4f})')
    return ms.labels_


# ==============================================================================
# 3. Spectral Clustering with Gaussian (RBF) Kernel
# ==============================================================================
def cluster_spectral_gaussian(embeddings, n_clusters=None, gamma=None,
                               max_clusters=10):
    """
    Spectral Clustering using a Gaussian (RBF) kernel affinity matrix.

    K(x, y) = exp(-gamma * ||x - y||²)

    Parameters
    ----------
    embeddings : ndarray [M, D]
    n_clusters : int or None
        If None, auto-selected via eigengap.
    gamma : float or None
        RBF kernel parameter. If None, uses 1 / (D * Var(X)).
    max_clusters : int
        Max clusters for auto-selection.

    Returns
    -------
    labels : ndarray [M] of int
    """
    M = len(embeddings)
    if M <= 1:
        return np.zeros(M, dtype=int)

    # Build RBF affinity matrix
    if gamma is None:
        gamma = 1.0 / (embeddings.shape[1] * embeddings.var() + 1e-9)

    affinity = rbf_kernel(embeddings, gamma=gamma)
    np.fill_diagonal(affinity, 1.0)

    if n_clusters is None:
        n_clusters = _estimate_n_clusters_eigengap(affinity, max_clusters)

    n_clusters = min(n_clusters, M)

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42,
        n_init=10,
    )
    labels = sc.fit_predict(affinity)

    n = len(np.unique(labels))
    print(f'    Spectral (Gaussian): {M} colonies → {n} cluster(s) '
          f'(gamma={gamma:.4f})')
    return labels


# ==============================================================================
# Eigengap Heuristic for automatic n_clusters estimation
# ==============================================================================
def _estimate_n_clusters_eigengap(affinity, max_k=10):
    """
    Estimate number of clusters from the eigengap of the normalized Laplacian.

    The largest gap between consecutive eigenvalues (sorted ascending)
    suggests the number of clusters.
    """
    M = affinity.shape[0]
    max_k = min(max_k, M)

    # Degree matrix and normalized Laplacian
    D = np.diag(affinity.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(affinity.sum(axis=1) + 1e-10))
    L = np.eye(M) - D_inv_sqrt @ affinity @ D_inv_sqrt

    # Eigenvalues (symmetric → real)
    eigenvalues = np.sort(np.real(np.linalg.eigvalsh(L)))

    # Find largest gap in the first max_k eigenvalues
    gaps = np.diff(eigenvalues[:max_k + 1])
    if len(gaps) == 0:
        return 1

    # Skip the trivial first eigenvalue (≈0), look at gaps starting from index 0
    best_k = int(np.argmax(gaps[1:max_k])) + 2  # +2: 1-indexed, skip first gap
    best_k = max(2, min(best_k, M))

    return best_k


# ==============================================================================
# Unified Dispatcher
# ==============================================================================
CLUSTERING_METHODS = {
    'meanshift':          'Original Mean Shift (Euclidean)',
    'meanshift_cosine':   'Mean Shift on L2-normed embeddings (cosine)',
    'spectral_cosine':    'Spectral Clustering with cosine affinity',
    'spectral_gaussian':  'Spectral Clustering with Gaussian (RBF) kernel',
}


def cluster_with_method(embeddings, method='meanshift', **kwargs):
    """
    Unified clustering dispatcher.

    Parameters
    ----------
    embeddings : ndarray [M, D]
    method : str
        One of: 'meanshift', 'meanshift_cosine', 'spectral_cosine',
        'spectral_gaussian'.
    **kwargs : dict
        Passed to the chosen clustering function.

    Returns
    -------
    labels : ndarray [M] of int
    """
    if method == 'meanshift':
        return _cluster_meanshift_original(embeddings, **kwargs)
    elif method == 'meanshift_cosine':
        return cluster_meanshift_cosine(embeddings, **kwargs)
    elif method == 'spectral_cosine':
        return cluster_spectral_cosine(embeddings, **kwargs)
    elif method == 'spectral_gaussian':
        return cluster_spectral_gaussian(embeddings, **kwargs)
    else:
        raise ValueError(
            f'Unknown method: {method}. '
            f'Supported: {list(CLUSTERING_METHODS.keys())}')


def _cluster_meanshift_original(embeddings, bandwidth=None, quantile=0.3):
    """Original Mean Shift (same as notebook's cluster_with_meanshift)."""
    M = len(embeddings)
    if M <= 1:
        return np.zeros(M, dtype=int)

    if bandwidth is None:
        bandwidth = estimate_bandwidth(
            embeddings, quantile=quantile,
            n_samples=min(M, 500),
        )
        bandwidth = max(bandwidth, 1e-3)

    for bin_seeding in (True, False):
        try:
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding,
                           cluster_all=True)
            ms.fit(embeddings)
            break
        except ValueError:
            if not bin_seeding:
                print(f'    MeanShift: failed at bw={bandwidth:.4f}, '
                      f'returning 1 cluster')
                return np.zeros(M, dtype=int)

    n = len(np.unique(ms.labels_))
    print(f'    MeanShift: {M} colonies → {n} cluster(s) (bw={bandwidth:.4f})')
    return ms.labels_


def compare_clustering_methods(embeddings, gt_labels, methods=None, **kwargs):
    """
    Run multiple clustering methods on the same embeddings and compare.

    Parameters
    ----------
    embeddings : ndarray [M, D]
    gt_labels : ndarray [M] of int
        Ground truth species indices for quality metrics.
    methods : list of str or None
        Methods to compare. None = all supported methods.
    **kwargs : dict
        Extra params forwarded to each method.

    Returns
    -------
    results : dict  {method_name: {'labels': ndarray, 'n_clusters': int,
                                    'homogeneity': float, 'completeness': float,
                                    'v_measure': float}}
    """
    from sklearn.metrics import homogeneity_completeness_v_measure

    if methods is None:
        methods = list(CLUSTERING_METHODS.keys())

    results = {}
    for method in methods:
        print(f'\n--- {CLUSTERING_METHODS.get(method, method)} ---')
        labels = cluster_with_method(embeddings, method=method, **kwargs)
        h, c, v = homogeneity_completeness_v_measure(gt_labels, labels)
        results[method] = {
            'labels':       labels,
            'n_clusters':   len(np.unique(labels)),
            'homogeneity':  h,
            'completeness': c,
            'v_measure':    v,
        }
        print(f'    H={h:.3f}  C={c:.3f}  V={v:.3f}  '
              f'(k={results[method]["n_clusters"]})')

    return results
