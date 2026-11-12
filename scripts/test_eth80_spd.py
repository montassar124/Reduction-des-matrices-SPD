"""Build SPD descriptors from ETH-80 and compare reductions."""

from collections import defaultdict
from pathlib import Path
import re

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import torch


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ETH-80"
RESULTS = ROOT / "eth80_spd_results.csv"
IMAGE_SIZE = (20, 20)
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_eth80() -> tuple[list[np.ndarray], np.ndarray]:
    image_pattern = re.compile(
        r"^([a-zA-Z]+)(\d+)-(\d+)-\d+\.png$", re.IGNORECASE
    )
    sets: list[np.ndarray] = []
    labels: list[str] = []
    for category_dir in sorted(DATA.iterdir()):
        if not category_dir.is_dir():
            continue
        groups: defaultdict[str, list[Path]] = defaultdict(list)
        for image_path in category_dir.rglob("*.png"):
            match = image_pattern.match(image_path.name)
            if match:
                groups[f"{match.group(1)}{match.group(2)}"].append(image_path)
        for instance, paths in sorted(groups.items()):
            if len(paths) < 10:
                continue
            images = [
                np.asarray(Image.open(path).convert("L").resize(IMAGE_SIZE), dtype=float).ravel()
                for path in sorted(paths)
            ]
            sets.append(np.stack(images))
            labels.append(category_dir.name)
    if len(sets) != 80:
        raise ValueError(f"Expected 80 ETH-80 objects, found {len(sets)}")
    return sets, np.asarray(labels)


def gaussian_spd_descriptor(image_set: np.ndarray) -> np.ndarray:
    mean = image_set.mean(axis=0)
    covariance = np.cov(image_set.T)
    covariance += 1e-3 * np.trace(covariance) * np.eye(image_set.shape[1])
    descriptor = np.zeros((image_set.shape[1] + 1, image_set.shape[1] + 1))
    descriptor[:-1, :-1] = covariance + np.outer(mean, mean)
    descriptor[:-1, -1] = mean
    descriptor[-1, :-1] = mean
    descriptor[-1, -1] = 1.0
    eigenvalues = np.linalg.eigvalsh(descriptor)
    if eigenvalues.min() <= 0:
        raise ValueError("ETH-80 descriptor is not SPD")
    return descriptor


def gpu_log_features(matrices: np.ndarray) -> torch.Tensor:
    """Compute batched SPD logarithms on CUDA."""
    values, vectors = torch.linalg.eigh(
        torch.as_tensor(matrices, dtype=torch.float64, device=DEVICE)
    )
    values = values.clamp_min(1e-12).log()
    return (vectors * values.unsqueeze(-2)) @ vectors.transpose(-1, -2)


def gpu_pca(train: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train.mean(dim=0)
    centered_train = train - mean
    centered_test = test - mean
    _, _, components = torch.linalg.svd(centered_train, full_matrices=False)
    basis = components[:2].T
    return centered_train @ basis, centered_test @ basis


def neighbor_preservation(
    original: torch.Tensor, embedding: torch.Tensor, neighbors: int = 6
) -> float:
    """Measure the percentage of original training neighbors retained."""
    count = original.shape[0]
    k = min(neighbors, count - 1)
    original_distances = torch.cdist(original, original)
    reduced_distances = torch.cdist(embedding, embedding)
    original_neighbors = torch.argsort(original_distances, dim=1)[:, 1 : k + 1]
    reduced_neighbors = torch.argsort(reduced_distances, dim=1)[:, 1 : k + 1]
    overlap = (
        original_neighbors.unsqueeze(2) == reduced_neighbors.unsqueeze(1)
    ).any(dim=2).float().sum(dim=1) / k
    return float(overlap.mean().item())


def gpu_lie_lle(
    train_matrices: np.ndarray, test_matrices: np.ndarray, neighbors: int = 6
) -> tuple[torch.Tensor, torch.Tensor]:
    train_logs = gpu_log_features(train_matrices).flatten(start_dim=1)
    test_logs = gpu_log_features(test_matrices).flatten(start_dim=1)
    k = min(neighbors, len(train_logs) - 1)
    distances = torch.cdist(train_logs, train_logs)
    weights = torch.zeros(
        (len(train_logs), len(train_logs)), dtype=torch.float64, device=DEVICE
    )
    for row in range(len(train_logs)):
        indices = torch.argsort(distances[row])[1 : k + 1]
        local = train_logs[indices] - train_logs[row]
        gram = local @ local.T
        gram += 1e-3 * torch.trace(gram).clamp_min(1e-12) / k * torch.eye(
            k, dtype=torch.float64, device=DEVICE
        )
        local_weights = torch.linalg.solve(
            gram, torch.ones(k, dtype=torch.float64, device=DEVICE)
        )
        weights[row, indices] = local_weights / local_weights.sum()
    identity = torch.eye(len(train_logs), dtype=torch.float64, device=DEVICE)
    gram = (identity - weights) @ (identity - weights).T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    embedding = eigenvectors[:, 1:3]
    test_distances = torch.cdist(test_logs, train_logs)
    test_embedding = torch.empty(
        (len(test_logs), 2), dtype=torch.float64, device=DEVICE
    )
    for row in range(len(test_logs)):
        indices = torch.argsort(test_distances[row])[:k]
        local = train_logs[indices] - test_logs[row]
        local_gram = local @ local.T
        local_gram += 1e-3 * torch.trace(local_gram).clamp_min(1e-12) / k * torch.eye(
            k, dtype=torch.float64, device=DEVICE
        )
        local_weights = torch.linalg.solve(
            local_gram, torch.ones(k, dtype=torch.float64, device=DEVICE)
        )
        test_embedding[row] = (
            local_weights / local_weights.sum()
        ) @ embedding[indices]
    return embedding, test_embedding


def gpu_multi_reference(
    train_matrices: np.ndarray, test_matrices: np.ndarray, references: int = 2
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Use multiple log-Euclidean tangent references fitted on training data."""
    train_logs = gpu_log_features(train_matrices)
    test_logs = gpu_log_features(test_matrices)
    train_flat = train_logs.flatten(start_dim=1)
    test_flat = test_logs.flatten(start_dim=1)
    clusterer = KMeans(n_clusters=references, n_init=5, random_state=SEED)
    assignments = clusterer.fit_predict(train_flat.detach().cpu().numpy())
    centers = torch.as_tensor(
        clusterer.cluster_centers_.reshape(references, *train_logs.shape[1:]),
        dtype=torch.float64,
        device=DEVICE,
    )
    center_distances = torch.cdist(train_flat, centers.flatten(start_dim=1))
    test_assignments = torch.argmin(
        torch.cdist(test_flat, centers.flatten(start_dim=1)), dim=1
    )
    train_local = train_logs - centers[torch.as_tensor(assignments, device=DEVICE)]
    test_local = test_logs - centers[test_assignments]
    global_log = train_logs.mean(dim=0)
    rows, cols = torch.triu_indices(
        train_logs.shape[-1], train_logs.shape[-1], offset=1, device=DEVICE
    )
    train_features = (train_local + global_log)[..., rows, cols]
    test_features = (test_local + global_log)[..., rows, cols]
    train_embedding, test_embedding = gpu_pca(train_features, test_features)
    return train_embedding, test_embedding, int(references)


def evaluate(
    matrices: np.ndarray, labels: np.ndarray, method: str, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    test: list[int] = []
    for category in np.unique(labels):
        indices = np.flatnonzero(labels == category)
        rng.shuffle(indices)
        train.extend(indices[:5])
        test.extend(indices[5:])
    train_indices = np.asarray(train)
    test_indices = np.asarray(test)
    train_features = torch.as_tensor(
        matrices[train_indices].reshape(len(train), -1),
        dtype=torch.float64,
        device=DEVICE,
    )
    test_features = torch.as_tensor(
        matrices[test_indices].reshape(len(test), -1),
        dtype=torch.float64,
        device=DEVICE,
    )
    original_features = train_features
    if method == "Lie-LLE":
        original_features = gpu_log_features(
            matrices[train_indices]
        ).flatten(start_dim=1)
        train_embedding, test_embedding = gpu_lie_lle(
            matrices[train_indices], matrices[test_indices]
        )
    elif method == "Repères multiples":
        original_features = gpu_log_features(
            matrices[train_indices]
        ).flatten(start_dim=1)
        train_embedding, test_embedding, _ = gpu_multi_reference(
            matrices[train_indices], matrices[test_indices]
        )
    else:
        if method == "PCA tangentielle":
            log_matrices = gpu_log_features(matrices)
            train_features = log_matrices[train_indices].flatten(start_dim=1)
            test_features = log_matrices[test_indices].flatten(start_dim=1)
            original_features = train_features
        train_embedding, test_embedding = gpu_pca(train_features, test_features)
    classifier = KNeighborsClassifier(n_neighbors=1)
    train_embedding_cpu = train_embedding.detach().cpu().numpy()
    test_embedding_cpu = test_embedding.detach().cpu().numpy()
    classifier.fit(train_embedding_cpu, labels[train_indices])
    accuracy = float(
        accuracy_score(labels[test_indices], classifier.predict(test_embedding_cpu))
    )
    preservation = neighbor_preservation(original_features, train_embedding)
    return accuracy, preservation


def main() -> None:
    image_sets, labels = load_eth80()
    matrices = np.asarray([gaussian_spd_descriptor(images) for images in image_sets])
    eigenvalues = np.asarray([np.linalg.eigvalsh(matrix)[0] for matrix in matrices])
    print(
        f"ETH-80: {len(matrices)} SPD matrices, shape {matrices.shape[1:]}, "
        f"minimum eigenvalue {eigenvalues.min():.3e}, device={DEVICE}"
    )
    rows = []
    for method in (
        "PCA classique",
        "PCA tangentielle",
        "Repères multiples",
        "Lie-LLE",
    ):
        scores = [
            evaluate(matrices, labels, method, seed) for seed in range(10)
        ]
        accuracies = [score[0] for score in scores]
        preservation = [score[1] for score in scores]
        rows.append(
            {
                "Method": method,
                "Mean accuracy": np.mean(accuracies),
                "Std": np.std(accuracies),
                "Neighbor preservation": np.mean(preservation),
                "Train per class": 5,
                "Test per class": 5,
                "Matrices": len(matrices),
            }
        )
    import pandas as pd

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS, index=False)
    print(results.round(4).to_string(index=False))
    print(f"Saved: {RESULTS}")


if __name__ == "__main__":
    main()
