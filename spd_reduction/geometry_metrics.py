"""Local geometry and neighborhood preservation metrics for SPD dimensionality reduction under AIRM."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

FLOOR = 1e-12
MEAN_EPS = 1e-6


def random_spd_matrix(dim: int, cond: float, rng: np.random.Generator) -> np.ndarray:
    """Generate an SPD matrix with a target condition number, strictly positive definite."""
    q_mat, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    eigenvalues = np.logspace(0, np.log10(cond), dim)
    matrix = (q_mat * eigenvalues) @ q_mat.T
    matrix = 0.5 * (matrix + matrix.T)
    return matrix + FLOOR * np.eye(dim)


def generate_multicluster_spd_dataset(
    n_per_cluster: int = 30,
    dim: int = 28,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic SPD matrices across 6 clusters with condition numbers 1e0 to 1e6."""
    rng = np.random.default_rng(seed)
    conds = [1e0, 1e1, 1e2, 1e3, 1e4, 1e6]
    matrices: list[np.ndarray] = []
    labels: list[int] = []

    for cluster_id, cond in enumerate(conds):
        base = random_spd_matrix(dim, cond, rng)
        for _ in range(n_per_cluster):
            noise = rng.standard_normal((dim, dim)) * (0.02 + 0.01 * cluster_id)
            sample = base + (noise @ noise.T) / dim
            sample = 0.5 * (sample + sample.T)
            eigenvalues, eigenvectors = np.linalg.eigh(sample)
            eigenvalues = np.clip(eigenvalues, FLOOR, None)
            sample = (eigenvectors * eigenvalues) @ eigenvectors.T
            matrices.append(sample)
            labels.append(cluster_id)

    return np.stack(matrices), np.array(labels, dtype=int)


def spd_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Symmetric matrix square root with numerical eigenvalue floor."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, FLOOR, None)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T


def spd_inv_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Inverse symmetric matrix square root with numerical eigenvalue floor."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, FLOOR, None)
    return (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T


def spd_log(matrix: np.ndarray) -> np.ndarray:
    """Matrix logarithm with numerical eigenvalue floor."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, FLOOR, None)
    return (eigenvectors * np.log(eigenvalues)) @ eigenvectors.T


def airm_dist(first: np.ndarray, second: np.ndarray) -> float:
    """Affine-Invariant Riemannian distance, with Log-Euclidean fallback."""
    first_inv_sqrt = spd_inv_sqrt(first)
    relative = first_inv_sqrt @ second @ first_inv_sqrt
    relative = 0.5 * (relative + relative.T)
    eigenvalues, _ = np.linalg.eigh(relative)
    eigenvalues = np.clip(eigenvalues, FLOOR, None)
    dist = float(np.linalg.norm(np.log(eigenvalues)))
    if not np.isfinite(dist):
        dist = float(np.linalg.norm(spd_log(first) - spd_log(second)))
    return dist


def pairwise_airm(matrices: np.ndarray) -> np.ndarray:
    """Compute pairwise AIRM distance matrix."""
    n_samples = len(matrices)
    distances = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distances[i, j] = distances[j, i] = airm_dist(matrices[i], matrices[j])
    return distances


def frechet_mean_regularized(
    matrices: np.ndarray,
    tol: float = 1e-9,
    max_iter: int = 60,
) -> np.ndarray:
    """Regularized affine-invariant Karcher/Fréchet mean."""
    dim = matrices.shape[1]
    mean = matrices.mean(axis=0) + MEAN_EPS * np.eye(dim)
    for _ in range(max_iter):
        mean_sqrt = spd_sqrt(mean)
        mean_inv = spd_inv_sqrt(mean)
        tangent_mean = np.mean([spd_log(mean_inv @ mat @ mean_inv) for mat in matrices], axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(tangent_mean)
        mean = mean_sqrt @ (eigenvectors * np.exp(eigenvalues)) @ eigenvectors.T @ mean_sqrt
        mean = 0.5 * (mean + mean.T) + MEAN_EPS * np.eye(dim)
        if np.linalg.norm(tangent_mean) < tol:
            break
    return mean


def vectorize_sym_isometric(matrix: np.ndarray) -> np.ndarray:
    """Frobenius-isometric vectorization of a symmetric matrix.

    - Diagonal entries are preserved as-is.
    - Off-diagonal entries are multiplied by sqrt(2).
    Ensures <A, B>_Frobenius = <vec(A), vec(B)>_Euclidean.
    """
    dim = matrix.shape[-1]
    upper_idx = np.triu_indices(dim)
    vector = matrix[..., upper_idx[0], upper_idx[1]].copy()
    off_diagonal = upper_idx[0] != upper_idx[1]
    vector[..., off_diagonal] *= np.sqrt(2.0)
    return vector


def vectorize_batch_isometric(matrices: np.ndarray) -> np.ndarray:
    """Batch Frobenius-isometric vectorization."""
    return np.stack([vectorize_sym_isometric(matrix) for matrix in matrices])


def classical_pca_multidim(
    matrices: np.ndarray,
    dims: list[int],
) -> dict[int, np.ndarray]:
    """Classical PCA on Frobenius-isometric vectorized symmetric matrices."""
    vectors = vectorize_batch_isometric(matrices)
    mean_vec = vectors.mean(axis=0)
    centered = vectors - mean_vec
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return {dim: centered @ vt[:dim].T for dim in dims}


def tangent_pca_multidim(
    matrices: np.ndarray,
    dims: list[int],
) -> dict[int, np.ndarray]:
    """Tangent PCA on AIRM tangent vectors at regularized Fréchet mean."""
    n_samples = len(matrices)
    mean_mat = frechet_mean_regularized(matrices)
    mean_inv = spd_inv_sqrt(mean_mat)
    tangents = np.stack([spd_log(mean_inv @ matrices[i] @ mean_inv) for i in range(n_samples)])
    vectors = vectorize_batch_isometric(tangents)
    mean_vec = vectors.mean(axis=0)
    centered = vectors - mean_vec
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return {dim: centered @ vt[:dim].T for dim in dims}


def knn_from_dist(dist_matrix: np.ndarray, k: int) -> np.ndarray:
    """Extract k nearest neighbors indices from a distance matrix."""
    dist_copy = dist_matrix.copy()
    np.fill_diagonal(dist_copy, np.inf)
    return np.argsort(dist_copy, axis=1)[:, :k]


def knn_from_points(points: np.ndarray, k: int) -> np.ndarray:
    """Extract k nearest neighbors from embedded coordinate points."""
    return knn_from_dist(cdist(points, points), k)


def neighbor_overlap_per_point(knn_a: np.ndarray, knn_b: np.ndarray) -> np.ndarray:
    """Fraction of shared neighbors per sample."""
    n_samples, k = knn_a.shape
    return np.array([len(set(knn_a[i]) & set(knn_b[i])) / k for i in range(n_samples)])


def np_at_k(dist_high: np.ndarray, points_low: np.ndarray, k: int) -> float:
    """Neighborhood Preservation at k (NP@k): fraction of AIRM neighbors preserved."""
    knn_h = knn_from_dist(dist_high, k)
    knn_l = knn_from_points(points_low, k)
    return float(neighbor_overlap_per_point(knn_h, knn_l).mean())


def trustworthiness(dist_high: np.ndarray, points_low: np.ndarray, k: int) -> float:
    """Trustworthiness: penalizes false neighbors (intruders) in reduced space."""
    n_samples = dist_high.shape[0]
    knn_h = knn_from_dist(dist_high, k)
    dist_low = cdist(points_low, points_low)
    ranks_low = np.argsort(np.argsort(dist_low, axis=1), axis=1)

    penalty = 0.0
    for i in range(n_samples):
        for rank_idx, j in enumerate(knn_h[i]):
            penalty += ranks_low[i, j] - rank_idx

    denom = n_samples * k * (2 * n_samples - 3 * k - 1) / 2.0
    return 1.0 - 2.0 / max(denom, 1e-12) * penalty


def continuity(dist_high: np.ndarray, points_low: np.ndarray, k: int) -> float:
    """Continuity: penalizes missing neighbors from high-dimensional space."""
    n_samples = dist_high.shape[0]
    dist_low = cdist(points_low, points_low)
    knn_l = knn_from_dist(dist_low, k)
    ranks_high = np.argsort(np.argsort(dist_high, axis=1), axis=1)

    penalty = 0.0
    for i in range(n_samples):
        for rank_idx, j in enumerate(knn_l[i]):
            penalty += ranks_high[i, j] - rank_idx

    denom = n_samples * k * (2 * n_samples - 3 * k - 1) / 2.0
    return 1.0 - 2.0 / max(denom, 1e-12) * penalty


def distance_distortion(
    dist_high: np.ndarray,
    points_low: np.ndarray,
    k: int,
) -> dict[str, float]:
    """Local and global relative distance distortion and Spearman rank correlation."""
    n_samples = dist_high.shape[0]
    dist_low = cdist(points_low, points_low)

    # Local distortion (k-NN)
    knn_h = knn_from_dist(dist_high, k)
    loc_ratios: list[float] = []
    for i in range(n_samples):
        for j in knn_h[i]:
            if dist_high[i, j] > FLOOR:
                loc_ratios.append(abs(dist_low[i, j] / dist_high[i, j] - 1.0))

    loc_arr = np.array(loc_ratios)

    # Global distortion (all pairs)
    upper_idx = np.triu_indices(n_samples, k=1)
    dh = dist_high[upper_idx]
    dl = dist_low[upper_idx]
    mask = dh > FLOOR
    glob_ratios = np.abs(dl[mask] / dh[mask] - 1.0)

    # Spearman rank correlation
    rho, _ = spearmanr(dh, dl)

    return {
        "local_mean": float(loc_arr.mean()),
        "local_median": float(np.median(loc_arr)),
        "global_mean": float(glob_ratios.mean()),
        "global_median": float(np.median(glob_ratios)),
        "spearman": float(rho),
    }


def local_reconstruction_error(
    matrices: np.ndarray,
    dist_high: np.ndarray,
    points_low: np.ndarray,
    k: int,
) -> float:
    """LLE-style reconstruction error using Riemannian tangent space weights."""
    n_samples = len(matrices)
    knn_h = knn_from_dist(dist_high, k)
    z_bar = points_low.mean(axis=0)
    errors: list[float] = []

    for i in range(n_samples):
        neighbors = knn_h[i]
        xi_inv = spd_inv_sqrt(matrices[i])
        tangent_neighbors = np.stack(
            [spd_log(xi_inv @ matrices[j] @ xi_inv) for j in neighbors]
        )
        v_neighbors = vectorize_batch_isometric(tangent_neighbors)

        gram = v_neighbors @ v_neighbors.T
        gram += 1e-8 * (np.trace(gram) / k) * np.eye(k)
        weights = np.linalg.solve(gram, np.ones(k))
        w_sum = weights.sum()
        weights = weights / w_sum if abs(w_sum) > 1e-12 else np.ones(k) / k

        reconstructed_z = weights @ points_low[neighbors]
        denom = np.linalg.norm(points_low[i] - z_bar) ** 2
        num = np.linalg.norm(points_low[i] - reconstructed_z) ** 2
        errors.append(num / max(denom, 1e-12))

    return float(np.mean(errors))
