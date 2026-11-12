"""Reusable multi-reference tangent coordinates for SPD matrices."""

from __future__ import annotations

import numpy as np

from .local_geometry import (
    aligned_local_tangent_coordinates,
    matrix_exp_symmetric,
    matrix_log_spd,
    select_number_of_centers,
    riemannian_kmeans,
)


def fit_multi_reference_tangent(
    matrices: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Fit adaptive SPD references and return aligned tangent coordinates."""
    n_centers, scores = select_number_of_centers(
        matrices, random_state=random_state
    )
    centers, assignments, distances = riemannian_kmeans(
        matrices, n_centers, random_state=random_state
    )
    log_matrices = np.asarray([matrix_log_spd(matrix) for matrix in matrices])
    global_center = matrix_exp_symmetric(log_matrices.mean(axis=0))
    coordinates = aligned_local_tangent_coordinates(
        matrices, centers, assignments, global_center
    )
    summary = {
        "n_centers": int(n_centers),
        "silhouette_scores": scores,
        "mean_center_distance": float(np.mean(distances)),
    }
    return coordinates, centers, assignments, summary


def transform_multi_reference_tangent(
    matrices: np.ndarray,
    centers: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign new SPD matrices to fitted references and align their coordinates."""
    log_matrices = np.asarray([matrix_log_spd(matrix) for matrix in matrices])
    log_centers = np.asarray([matrix_log_spd(center) for center in centers])
    distances = np.linalg.norm(
        log_matrices[:, None] - log_centers[None, :], axis=(2, 3)
    )
    assignments = np.argmin(distances, axis=1)
    coordinates = aligned_local_tangent_coordinates(
        matrices, centers, assignments, target
    )
    return coordinates, assignments
