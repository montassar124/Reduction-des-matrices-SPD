"""Numerical regularization and stability methods for SPD matrices near the cone boundary."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def generate_connectome_data(
    n_samples_per_class: int = 500,
    dim: int = 28,
    noise_level: float = 1e-6,
    class_shift: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates SPD matrices resembling brain connectomes.

    - Diagonal is forced to 1.0 (correlation matrices).
    - Uses a latent factor model (rank 5) to create network structure.
    - noise_level controls proximity to the convex cone border:
      small noise -> eigenvalues close to 0 (Border);
      large noise -> eigenvalues large (Inside).
    """
    rng = np.random.default_rng(seed)
    matrices: list[np.ndarray] = []
    labels: list[int] = []

    for class_id in range(2):
        for _ in range(n_samples_per_class):
            factors = rng.standard_normal((dim, 5)) * (1.0 + class_shift * class_id)
            unique_noise = rng.uniform(0.1, 1.0, size=dim) * noise_level

            covariance = factors @ factors.T + np.diag(unique_noise)
            diag_sqrt = np.sqrt(np.diag(covariance))
            correlation = covariance / np.outer(diag_sqrt, diag_sqrt)

            correlation = 0.5 * (correlation + correlation.T)
            correlation += np.eye(dim) * 1e-10  # tiny jitter for strict positive definiteness

            matrices.append(correlation)
            labels.append(class_id)

    return np.stack(matrices), np.array(labels, dtype=int)


def batch_spd_sqrt(matrices: np.ndarray, floor: float = 1e-15) -> np.ndarray:
    """Compute symmetric matrix square roots for a batch of SPD matrices."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    eigenvalues = np.clip(eigenvalues, floor, None)
    return np.einsum("nij,nj,nkj->nik", eigenvectors, np.sqrt(eigenvalues), eigenvectors)


def batch_spd_inv_sqrt(matrices: np.ndarray, floor: float = 1e-15) -> np.ndarray:
    """Compute inverse symmetric matrix square roots for a batch of SPD matrices."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    eigenvalues = np.clip(eigenvalues, floor, None)
    return np.einsum("nij,nj,nkj->nik", eigenvectors, 1.0 / np.sqrt(eigenvalues), eigenvectors)


def batch_spd_log(matrices: np.ndarray, floor: float = 1e-15) -> np.ndarray:
    """Compute matrix logarithms for a batch of SPD matrices."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    eigenvalues = np.clip(eigenvalues, floor, None)
    return np.einsum("nij,nj,nkj->nik", eigenvectors, np.log(eigenvalues), eigenvectors)


def diag_loading(matrices: np.ndarray, eps: float) -> np.ndarray:
    """M1: Diagonal loading C_i <- C_i + eps * I."""
    identity = np.eye(matrices.shape[-1], dtype=matrices.dtype)
    return matrices + eps * identity


def eig_flooring(matrices: np.ndarray, tau: float) -> np.ndarray:
    """M2: Eigenvalue flooring at tau followed by spectral reconstruction."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    eigenvalues = np.maximum(eigenvalues, tau)
    return np.einsum("nij,nj,nkj->nik", eigenvectors, eigenvalues, eigenvectors)


def spd_frechet_mean(
    matrices: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 30,
    floor: float = 1e-15,
) -> np.ndarray:
    """Compute the Riemannian Karcher/Fréchet mean using AIRM updates."""
    mean = matrices.mean(axis=0)
    for _ in range(max_iter):
        mean_sqrt = batch_spd_sqrt(mean[None], floor)[0]
        mean_inv_sqrt = batch_spd_inv_sqrt(mean[None], floor)[0]
        relative = mean_inv_sqrt @ matrices @ mean_inv_sqrt
        tangent_mean = batch_spd_log(relative, floor).mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(tangent_mean)
        mean = mean_sqrt @ (eigenvectors * np.exp(eigenvalues)) @ eigenvectors.T @ mean_sqrt
        mean = 0.5 * (mean + mean.T)
        if np.linalg.norm(tangent_mean) < tol:
            break
    return mean


def fit_tangent_pca(
    spd_mats: np.ndarray,
    k: int = 2,
    mean_eps: float = 0.0,
    floor: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Fit tangent PCA on SPD matrices.

    If mean_eps > 0, applies M3: regularize only the Fréchet mean with + mean_eps * I.
    """
    mean = spd_frechet_mean(spd_mats, floor=floor)
    if mean_eps > 0.0:
        mean = mean + mean_eps * np.eye(mean.shape[0])

    mean_inv_sqrt = batch_spd_inv_sqrt(mean[None], floor)[0]
    relative = mean_inv_sqrt @ spd_mats @ mean_inv_sqrt
    tangents = batch_spd_log(relative, floor)

    dim = tangents.shape[1]
    upper_idx = np.triu_indices(dim)
    tangent_vectors = tangents[:, upper_idx[0], upper_idx[1]]
    t_mean = tangent_vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(tangent_vectors - t_mean, full_matrices=False)
    return mean, vt[:k], t_mean, upper_idx


def transform_tangent(
    spd_mats: np.ndarray,
    mean_spd: np.ndarray,
    comps: np.ndarray,
    t_mean: np.ndarray,
    upper_idx: tuple[np.ndarray, np.ndarray],
    floor: float = 1e-15,
) -> np.ndarray:
    """Project new SPD matrices onto fitted tangent PCA components."""
    mean_inv_sqrt = batch_spd_inv_sqrt(mean_spd[None], floor)[0]
    relative = mean_inv_sqrt @ spd_mats @ mean_inv_sqrt
    tangents = batch_spd_log(relative, floor)
    tangent_vectors = tangents[:, upper_idx[0], upper_idx[1]]
    return (tangent_vectors - t_mean) @ comps.T


def fit_pca_spd(spd_mats: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Fit classical PCA on flattened symmetric matrices."""
    n_samples = spd_mats.shape[0]
    vectors = spd_mats.reshape(n_samples, -1)
    mean_vector = vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(vectors - mean_vector, full_matrices=False)
    return mean_vector, vt[:k]


def transform_pca_spd(
    spd_mats: np.ndarray,
    mean_vector: np.ndarray,
    comps: np.ndarray,
) -> np.ndarray:
    """Transform SPD matrices using classical PCA components."""
    n_samples = spd_mats.shape[0]
    vectors = spd_mats.reshape(n_samples, -1)
    return (vectors - mean_vector) @ comps.T


def evaluate_regularization_methods(
    noise_level: float,
    class_shift: float,
    eps: float = 1e-3,
    tau: float = 1e-3,
    n_runs: int = 10,
    seed: int = 42,
    n_samples_per_class: int = 500,
    dim: int = 28,
) -> dict[str, list[float]]:
    """Evaluate Baseline, M1 DiagLoad, M2 EigFloor, and M3 MeanReg over repeated splits."""
    matrices, labels = generate_connectome_data(
        n_samples_per_class=n_samples_per_class,
        dim=dim,
        noise_level=noise_level,
        class_shift=class_shift,
        seed=seed,
    )
    results: dict[str, list[float]] = {
        "Baseline": [],
        "M1 DiagLoad": [],
        "M2 EigFloor": [],
        "M3 MeanReg": [],
    }

    for run in range(n_runs):
        idx_train, idx_test = train_test_split(
            np.arange(len(labels)),
            test_size=0.2,
            random_state=run,
            stratify=labels,
        )
        x_train, y_train = matrices[idx_train], labels[idx_train]
        x_test, y_test = matrices[idx_test], labels[idx_test]

        def _evaluate_variant(name: str, x_tr_used: np.ndarray, x_te_used: np.ndarray, **kwargs):
            try:
                mean_spd, comps, t_mean, upper_idx = fit_tangent_pca(x_tr_used, k=2, **kwargs)
                z_train = transform_tangent(x_tr_used, mean_spd, comps, t_mean, upper_idx)
                z_test = transform_tangent(x_te_used, mean_spd, comps, t_mean, upper_idx)
                pred = KNeighborsClassifier(n_neighbors=5).fit(z_train, y_train).predict(z_test)
                results[name].append(float(accuracy_score(y_test, pred) * 100.0))
            except Exception as exc:
                results[name].append(0.0)
                print(f"[{name}] run {run} failed: {exc}")

        # Baseline
        _evaluate_variant("Baseline", x_train, x_test)

        # M1: Diagonal loading on all matrices
        _evaluate_variant("M1 DiagLoad", diag_loading(x_train, eps), diag_loading(x_test, eps))

        # M2: Eigenvalue flooring on all matrices
        _evaluate_variant("M2 EigFloor", eig_flooring(x_train, tau), eig_flooring(x_test, tau))

        # M3: Only Fréchet mean regularized
        _evaluate_variant("M3 MeanReg", x_train, x_test, mean_eps=eps)

    return results
