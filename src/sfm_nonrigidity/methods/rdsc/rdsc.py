"""Inference-only reference kernels; see docs/methods.md.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def robust_affine_residuals(
    positions: np.ndarray, residuals: np.ndarray, *, iterations: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Remove a robust affine nuisance field without using deformation truth."""
    # A coordinate-wise median is not rotation equivariant and therefore made
    # the normalized score depend on the arbitrary SfM world orientation.
    # The centroid is exactly translation/rotation/scale equivariant; q90 below
    # still limits the influence of isolated support outliers on the scale.
    center = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center, axis=1)
    scale = float(np.quantile(distances, 0.9))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("Degenerate reconstructed point support")
    design = np.column_stack((np.ones(len(positions)), (positions - center) / scale))
    weights = np.ones(len(positions), dtype=np.float64)
    coefficients = np.zeros((4, 3), dtype=np.float64)
    for _ in range(iterations):
        root_w = np.sqrt(weights)[:, None]
        coefficients = np.linalg.lstsq(design * root_w, residuals * root_w, rcond=None)[
            0
        ]
        errors = np.linalg.norm(residuals - design @ coefficients, axis=1)
        robust_scale = 1.4826 * float(np.median(np.abs(errors - np.median(errors))))
        robust_scale = max(robust_scale, np.finfo(float).eps * scale)
        cutoff = 1.345 * robust_scale
        updated = np.minimum(1.0, cutoff / np.maximum(errors, np.finfo(float).tiny))
        if np.max(np.abs(updated - weights)) < 1e-8:
            weights = updated
            break
        weights = updated
    return residuals - design @ coefficients, weights


def _neighbor_indices(positions: np.ndarray, neighbors: int) -> np.ndarray:
    k = min(max(2, neighbors + 1), len(positions))
    return cKDTree(positions).query(positions, k=k, workers=1)[1]


def detection_scores(
    positions: np.ndarray,
    residuals: np.ndarray,
    *,
    neighbors: int,
    shuffle_count: int,
    seed: int,
) -> dict[str, float]:
    """Compute scores using reconstructed geometry only."""
    center = np.mean(positions, axis=0)
    spatial_scale = float(np.quantile(np.linalg.norm(positions - center, axis=1), 0.9))
    if spatial_scale <= 1e-12:
        raise ValueError("Degenerate spatial scale")
    raw_norm = np.linalg.norm(residuals, axis=1) / spatial_scale
    affine_residual, weights = robust_affine_residuals(positions, residuals)
    affine_norm = np.linalg.norm(affine_residual, axis=1) / spatial_scale
    indices = _neighbor_indices(positions, neighbors)
    coherent = affine_residual[indices].mean(axis=1)
    coherent_q95 = float(
        np.quantile(np.linalg.norm(coherent, axis=1), 0.95) / spatial_scale
    )

    rng = np.random.default_rng(seed)
    shuffled_q95 = np.empty(shuffle_count, dtype=np.float64)
    for index in range(shuffle_count):
        shuffled = affine_residual[rng.permutation(len(affine_residual))]
        shuffled_mean = shuffled[indices].mean(axis=1)
        shuffled_q95[index] = (
            np.quantile(np.linalg.norm(shuffled_mean, axis=1), 0.95) / spatial_scale
        )
    shuffle_reference = float(np.median(shuffled_q95))
    coherence_excess = coherent_q95 / max(shuffle_reference, np.finfo(float).tiny)
    return {
        "raw_median": float(np.median(raw_norm)),
        "raw_q95": float(np.quantile(raw_norm, 0.95)),
        "affine_residual_q95": float(np.quantile(affine_norm, 0.95)),
        "coherent_affine_q95": coherent_q95,
        "coherence_excess": coherence_excess,
        "rdsc": coherent_q95 * coherence_excess,
        "spatial_scale": spatial_scale,
        "robust_inlier_fraction": float(np.mean(weights >= 0.999999)),
    }
