"""Local multi-chart geometry for SPD connectomes."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score


def _spectral_function(matrix: np.ndarray, function) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    return (vectors * function(values)) @ vectors.T


def matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric square root of an SPD matrix."""
    return _spectral_function(matrix, np.sqrt)


def matrix_inv_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Return the inverse symmetric square root of an SPD matrix."""
    return _spectral_function(matrix, lambda values: 1.0 / np.sqrt(values))


def matrix_exp_symmetric(matrix: np.ndarray) -> np.ndarray:
    """Exponentiate a symmetric matrix through its eigendecomposition."""
    return _spectral_function(matrix, np.exp)


def matrix_log_spd(matrix: np.ndarray) -> np.ndarray:
    """Take the principal logarithm of an SPD matrix."""
    return _spectral_function(matrix, np.log)


def log_map(base: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Map an SPD point to the affine-invariant tangent space at ``base``."""
    base_sqrt = matrix_sqrt(base)
    base_inv_sqrt = matrix_inv_sqrt(base)
    relative = base_inv_sqrt @ point @ base_inv_sqrt
    return base_sqrt @ matrix_log_spd(relative) @ base_sqrt


def exp_map(base: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """Map a tangent vector at ``base`` back to the SPD manifold."""
    base_sqrt = matrix_sqrt(base)
    base_inv_sqrt = matrix_inv_sqrt(base)
    relative = base_inv_sqrt @ tangent @ base_inv_sqrt
    return base_sqrt @ matrix_exp_symmetric(relative) @ base_sqrt


def parallel_transport(
    tangent: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Transport a tangent vector from ``source`` to ``target``."""
    target_sqrt = matrix_sqrt(target)
    target_inv_sqrt = matrix_inv_sqrt(target)
    relative = target_inv_sqrt @ source @ target_inv_sqrt
    transport = (
        target_sqrt
        @ matrix_inv_sqrt(relative)
        @ target_inv_sqrt
    )
    return transport @ tangent @ transport.T


def aligned_local_tangent_coordinates(
    matrices: np.ndarray,
    centers: np.ndarray,
    assignments: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Transport local log vectors to one common tangent space."""
    rows, cols = np.triu_indices(matrices.shape[-1], k=1)
    features = np.zeros((len(matrices), len(rows)), dtype=np.float64)
    for index, (matrix, assignment) in enumerate(zip(matrices, assignments)):
        local_tangent = log_map(centers[assignment], matrix)
        common_tangent = parallel_transport(
            local_tangent, centers[assignment], target
        )
        center_offset = log_map(target, centers[assignment])
        features[index] = (center_offset + common_tangent)[rows, cols]
    return features


def affine_invariant_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Compute the affine-invariant Riemannian distance between two SPD matrices."""
    relative = matrix_inv_sqrt(first) @ second @ matrix_inv_sqrt(first)
    eigenvalues = np.linalg.eigvalsh((relative + relative.T) / 2.0)
    return float(np.linalg.norm(np.log(eigenvalues)))


def pairwise_affine_invariant_distances(matrices: np.ndarray) -> np.ndarray:
    """Compute the full pairwise affine-invariant distance matrix."""
    distances = np.zeros((len(matrices), len(matrices)), dtype=np.float64)
    for first in range(len(matrices)):
        for second in range(first + 1, len(matrices)):
            distance = affine_invariant_distance(matrices[first], matrices[second])
            distances[first, second] = distances[second, first] = distance
    return distances


def riemannian_mean(
    matrices: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Compute the affine-invariant Karcher mean by iterative updates."""
    if len(matrices) == 0:
        raise ValueError("Cannot compute a mean of an empty cluster.")
    mean = matrices[0].copy()
    for _ in range(max_iterations):
        update = sum(log_map(mean, matrix) for matrix in matrices) / len(matrices)
        if np.linalg.norm(update, ord="fro") <= tolerance:
            break
        mean = exp_map(mean, update)
    return (mean + mean.T) / 2.0


def _initial_centers(
    distance_matrix: np.ndarray,
    n_centers: int,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    centers = [int(rng.integers(len(distance_matrix)))]
    while len(centers) < n_centers:
        nearest = np.min(distance_matrix[:, centers], axis=1)
        nearest[centers] = -1
        centers.append(int(np.argmax(nearest)))
    return np.asarray(centers, dtype=np.int64)


def riemannian_kmeans(
    matrices: np.ndarray,
    n_centers: int,
    max_iterations: int = 30,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster SPD matrices using affine-invariant distances and means."""
    if n_centers < 1 or n_centers > len(matrices):
        raise ValueError("The number of centers must be between 1 and n_samples.")
    if n_centers == 1:
        center = riemannian_mean(matrices)
        assignments = np.zeros(len(matrices), dtype=np.int64)
        distances = np.asarray(
            [affine_invariant_distance(matrix, center) for matrix in matrices]
        )
        return np.asarray([center]), assignments, distances
    distances = pairwise_affine_invariant_distances(matrices)
    center_indices = _initial_centers(distances, n_centers, random_state)
    centers = matrices[center_indices].copy()

    for _ in range(max_iterations):
        assignments = np.argmin(
            np.asarray(
                [
                    [affine_invariant_distance(matrix, center) for center in centers]
                    for matrix in matrices
                ]
            ),
            axis=1,
        )
        updated = []
        for cluster in range(n_centers):
            members = matrices[assignments == cluster]
            if len(members) == 0:
                updated.append(centers[cluster])
            else:
                updated.append(riemannian_mean(members))
        updated = np.asarray(updated)
        shift = max(
            affine_invariant_distance(old, new)
            for old, new in zip(centers, updated)
        )
        centers = updated
        if shift <= 1e-7:
            break

    assignments = np.argmin(
        np.asarray(
            [
                [affine_invariant_distance(matrix, center) for center in centers]
                for matrix in matrices
            ]
        ),
        axis=1,
    )
    center_distances = np.asarray(
        [
            affine_invariant_distance(matrix, centers[assignment])
            for matrix, assignment in zip(matrices, assignments)
        ]
    )
    return centers, assignments, center_distances


def select_number_of_centers(
    matrices: np.ndarray,
    candidates: range = range(1, 7),
    random_state: int = 42,
) -> tuple[int, dict[int, float]]:
    """Select local charts only when the data supports multiple neighborhoods."""
    distances = pairwise_affine_invariant_distances(matrices)
    _, one_assignments, one_distances = riemannian_kmeans(
        matrices, 1, random_state=random_state
    )
    baseline_dispersion = float(np.mean(one_distances))
    scores: dict[int, float] = {1: 0.0}
    for n_centers in candidates:
        if n_centers == 1:
            continue
        if n_centers >= len(matrices):
            continue
        _, assignments, center_distances = riemannian_kmeans(
            matrices, n_centers, random_state=random_state
        )
        minimum_size = max(5, len(matrices) // (n_centers * 4))
        if np.min(np.bincount(assignments, minlength=n_centers)) < minimum_size:
            continue
        silhouette = float(
            silhouette_score(distances, assignments, metric="precomputed")
        )
        improvement = (baseline_dispersion - float(np.mean(center_distances))) / max(
            baseline_dispersion, np.finfo(float).eps
        )
        if silhouette > 0.10 and improvement > 0.05:
            scores[n_centers] = silhouette
    valid_multiple = [number for number in scores if number > 1]
    if not valid_multiple:
        return 1, scores
    return max(valid_multiple, key=scores.get), scores


def local_tangent_coordinates(
    matrices: np.ndarray,
    centers: np.ndarray,
    assignments: np.ndarray,
) -> np.ndarray:
    """Return strict-upper-triangle tangent coordinates in each local chart."""
    rows, cols = np.triu_indices(matrices.shape[-1], k=1)
    features = np.zeros((len(matrices), len(rows)), dtype=np.float64)
    for index, (matrix, assignment) in enumerate(zip(matrices, assignments)):
        features[index] = log_map(centers[assignment], matrix)[rows, cols]
    return features


def local_relative_tangent_coordinates(
    matrices: np.ndarray,
    centers: np.ndarray,
    assignments: np.ndarray,
) -> np.ndarray:
    """Return local log coordinates in a common whitened symmetric basis."""
    rows, cols = np.triu_indices(matrices.shape[-1], k=1)
    features = np.zeros((len(matrices), len(rows)), dtype=np.float64)
    for index, (matrix, assignment) in enumerate(zip(matrices, assignments)):
        base_inv_sqrt = matrix_inv_sqrt(centers[assignment])
        relative = base_inv_sqrt @ matrix @ base_inv_sqrt
        features[index] = matrix_log_spd(relative)[rows, cols]
    return features


def local_reconstruction_errors(
    matrices: np.ndarray,
    centers: np.ndarray,
    assignments: np.ndarray,
) -> np.ndarray:
    """Measure tangent-chart round-trip error in Frobenius norm."""
    errors = []
    for matrix, assignment in zip(matrices, assignments):
        tangent = log_map(centers[assignment], matrix)
        reconstruction = exp_map(centers[assignment], tangent)
        errors.append(np.linalg.norm(matrix - reconstruction, ord="fro"))
    return np.asarray(errors)


def summarize_local_geometry(
    matrices: np.ndarray,
    random_state: int = 42,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Fit local charts and return summary, centers, assignments, and features."""
    n_centers, silhouette_scores = select_number_of_centers(
        matrices, random_state=random_state
    )
    centers, assignments, center_distances = riemannian_kmeans(
        matrices, n_centers, random_state=random_state
    )
    tangent_features = local_tangent_coordinates(matrices, centers, assignments)
    reconstruction_errors = local_reconstruction_errors(
        matrices, centers, assignments
    )
    pairwise = pairwise_affine_invariant_distances(matrices)
    summary = {
        "n_centers": n_centers,
        "silhouette_scores": silhouette_scores,
        "mean_center_distance": float(center_distances.mean()),
        "max_center_distance": float(center_distances.max()),
        "mean_reconstruction_error": float(reconstruction_errors.mean()),
        "max_reconstruction_error": float(reconstruction_errors.max()),
        "mean_pairwise_distance": float(pairwise[np.triu_indices_from(pairwise, k=1)].mean()),
    }
    return summary, centers, assignments, tangent_features
