"""
utils.py

General-purpose utilities used throughout the segmentation module:
- Safe division that avoids divide-by-zero errors
- Directory creation
- Lightweight logger
- JSON read/write convenience
- Z-score utilities
"""

import os
import json
import numpy as np
from typing import Any, Dict


# -------------------------------------------------------------
# Safe Division
# -------------------------------------------------------------

def safe_div(a, b):
    """Safe elementwise division. Returns 0 where division is not possible."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # Handle NaN and inf values in b, suppress warnings for invalid divisions
    with np.errstate(divide='ignore', invalid='ignore'):
        b_valid = np.isfinite(b) & (b > 0)
        result = np.zeros_like(a, dtype=float)
        np.divide(a, b, out=result, where=b_valid)
    return result


# -------------------------------------------------------------
# Directory Helpers
# -------------------------------------------------------------

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# -------------------------------------------------------------
# Lightweight Logger
# -------------------------------------------------------------

def log(msg: str):
    """Simple logger for internal module debugging."""
    print(f"[SEG] {msg}")


# -------------------------------------------------------------
# JSON Helpers
# -------------------------------------------------------------

def write_json(path: str, data: Dict[str, Any]):
    """Write dictionary as JSON with pretty formatting."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def read_json(path: str) -> Dict[str, Any]:
    """Read JSON file, return dictionary."""
    with open(path, "r") as f:
        return json.load(f)


# -------------------------------------------------------------
# Z-Score Utilities
# -------------------------------------------------------------

def compute_cluster_zscores(X_scaled, labels, feature_names):
    """
    Compute mean z-scores per cluster.

    Parameters:
    -----------
    X_scaled: np.ndarray
        Scaled feature matrix (StandardScaler output)
    labels: array-like
        Cluster labels
    feature_names: list
        Names of the features in X_scaled's columns

    Returns:
    --------
    dict:
        cluster_id -> {feature_name: zscore_mean}
    """
    clusters = {}
    df = np.column_stack([X_scaled, labels])
    labels_unique = np.unique(labels)

    for cl in labels_unique:
        mask = (labels == cl)
        cluster_mean = X_scaled[mask].mean(axis=0)
        clusters[int(cl)] = dict(zip(feature_names, cluster_mean))

    return clusters


def top_n_features(zscores: Dict[str, float], n: int = 5):
    """
    Given a dict of feature_name -> zscore, return:
        - top positive n
        - top negative n
    """
    sorted_items = sorted(zscores.items(), key=lambda x: x[1], reverse=True)
    top_pos = sorted_items[:n]
    top_neg = sorted_items[-n:]
    return top_pos, top_neg


def compute_segment_scores(X_scaled: np.ndarray, segment_centroids: Dict) -> np.ndarray:
    """
    Compute similarity scores for each ZCTA to all segments.
    
    Parameters:
    -----------
    X_scaled: np.ndarray
        Scaled feature matrix (n_samples, n_features)
    segment_centroids: dict
        Dictionary mapping segment_id (int or tuple) -> centroid array
        Supports both single-level (int keys) and two-level (tuple keys) segmentation
        
    Returns:
    --------
    scores: np.ndarray
        Array of shape (n_samples, n_segments) with similarity scores
        Scores are computed as negative Euclidean distance (higher = more similar)
    segment_keys: list
        List of segment_id values (int or tuple) in the same order as scores columns
    """
    n_samples = X_scaled.shape[0]
    segment_keys = list(segment_centroids.keys())
    n_segments = len(segment_keys)
    
    scores = np.zeros((n_samples, n_segments))
    
    for i, segment_id in enumerate(segment_keys):
        centroid = segment_centroids[segment_id]
        # Compute negative Euclidean distance (higher = more similar)
        distances = np.linalg.norm(X_scaled - centroid, axis=1)
        # Convert distance to similarity score (inverse distance, normalized)
        # Using negative distance so higher scores = more similar
        scores[:, i] = -distances
    
    return scores, segment_keys


# -------------------------------------------------------------
# Micro-Cluster K Auto-Selection
# -------------------------------------------------------------

def auto_select_micro_k(n_items: int, cfg) -> int:
    """
    Auto-select K for micro clustering based on macro-group size.

    Logic:
        if n < threshold_small: k_small
        if threshold_small <= n < threshold_large: k_medium
        if n >= threshold_large: k_large
    
    Important: K must never exceed n_items (K-means requirement).
    """
    if n_items < cfg.micro_threshold_small:
        k = cfg.micro_k_small
    elif n_items < cfg.micro_threshold_large:
        k = cfg.micro_k_medium
    else:
        k = cfg.micro_k_large
    
    # Ensure K never exceeds the number of items
    k = min(k, n_items)
    
    # Ensure at least 1 cluster (though this shouldn't happen in practice)
    k = max(1, k)
    
    return k

