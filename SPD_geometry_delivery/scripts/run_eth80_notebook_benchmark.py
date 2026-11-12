"""Reproduce the attached ETH-80 notebook benchmark exactly."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from scripts.test_eth80_spd import load_eth80

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eth80_notebook_benchmark.csv"


def neighbor_preservation(
    original: np.ndarray, embedding: np.ndarray, neighbors: int = 6
) -> float:
    """Return the percentage of original neighbours retained after reduction."""
    original = np.asarray(original).reshape(len(original), -1)
    embedding = np.asarray(embedding).reshape(len(embedding), -1)
    k = min(neighbors, len(original) - 1)
    if k <= 0:
        return 1.0
    original_distances = np.linalg.norm(
        original[:, None] - original[None, :], axis=2
    )
    embedding_distances = np.linalg.norm(
        embedding[:, None] - embedding[None, :], axis=2
    )
    original_neighbors = np.argsort(original_distances, axis=1)[:, 1 : k + 1]
    embedding_neighbors = np.argsort(embedding_distances, axis=1)[:, 1 : k + 1]
    retained = [
        len(set(expected).intersection(actual)) / k
        for expected, actual in zip(original_neighbors, embedding_neighbors)
    ]
    return float(np.mean(retained))


def descriptor(images: np.ndarray) -> np.ndarray:
    mean = images.mean(axis=0)
    covariance = np.cov(images.T)
    covariance += 1e-3 * np.trace(covariance) * np.eye(images.shape[1])
    covariance_eigenvalues = np.linalg.eigvalsh(covariance)
    log_scale = -np.sum(np.log(np.maximum(covariance_eigenvalues, 1e-300)))
    log_scale /= images.shape[1] + 1
    scale = np.exp(np.clip(log_scale, -700.0, 700.0))
    matrix = np.zeros((images.shape[1] + 1, images.shape[1] + 1))
    matrix[:-1, :-1] = covariance + np.outer(mean, mean)
    matrix[:-1, -1] = mean
    matrix[-1, :-1] = mean
    matrix[-1, -1] = 1.0
    return scale * (matrix + matrix.T) / 2.0


def log_spd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    return (vectors * np.log(np.maximum(values, 1e-15))) @ vectors.T


def lie_lle(logs: np.ndarray, k: int = 6, dimension: int = 30) -> np.ndarray:
    count = len(logs)
    dimension = min(dimension, logs.shape[1] ** 2, count - 2)
    flat = logs.reshape(count, -1)
    distances = np.linalg.norm(flat[:, None] - flat[None, :], axis=2)
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="precomputed").fit(
        distances
    ).kneighbors(distances, return_distance=False)[:, 1:]
    weights = np.zeros((count, count))
    for row in range(count):
        local = flat[neighbors[row]] - flat[row]
        gram = local @ local.T
        gram += 1e-3 * max(np.trace(gram) / k, 1e-12) * np.eye(k)
        try:
            values = np.linalg.solve(gram, np.ones(k))
        except np.linalg.LinAlgError:
            values = np.linalg.lstsq(gram, np.ones(k), rcond=None)[0]
        weights[row, neighbors[row]] = values / values.sum()
    objective = (np.eye(count) - weights) @ (np.eye(count) - weights).T
    _, vectors = np.linalg.eigh(objective)
    coefficients = vectors[:, 1 : dimension + 1]
    basis = flat.T @ coefficients
    left_vectors, _, _ = np.linalg.svd(basis, full_matrices=False)
    projection = np.zeros((logs.shape[1], dimension))
    for column in range(dimension):
        symmetric = (left_vectors[:, column].reshape(logs.shape[1], -1))
        symmetric = (symmetric + symmetric.T) / 2.0
        values, vectors = np.linalg.eigh(symmetric)
        projection[:, column] = vectors[:, -1] * np.sign(values[-1])
    projection, _ = np.linalg.qr(projection)
    return np.asarray(
        [
            _matrix_exp(
                projection.T @ log_matrix @ projection
            )
            for log_matrix in logs
        ]
    )


def _matrix_exp(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    return (vectors * np.exp(values)) @ vectors.T


def multi_reference_features(logs: np.ndarray, references: int = 2) -> np.ndarray:
    flat = logs.reshape(len(logs), -1)
    clusters = KMeans(n_clusters=references, n_init=10, random_state=42)
    assignments = clusters.fit_predict(flat)
    centers = clusters.cluster_centers_.reshape(references, logs.shape[1], logs.shape[2])
    global_center = logs.mean(axis=0)
    local = logs - centers[assignments] + global_center
    rows, cols = np.triu_indices(logs.shape[1], k=1)
    return local[:, rows, cols]


def one_nn_euclidean(train: np.ndarray, train_labels: np.ndarray,
                     test: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(test[:, None, :] - train[None, :, :], axis=2)
    return train_labels[np.argmin(distances, axis=1)]


def one_nn_log_spd(train: np.ndarray, train_labels: np.ndarray,
                   test: np.ndarray) -> np.ndarray:
    train_logs = np.asarray([log_spd(matrix) for matrix in train])
    predictions = []
    for matrix in test:
        distances = np.linalg.norm(train_logs - log_spd(matrix), axis=(1, 2))
        predictions.append(train_labels[np.argmin(distances)])
    return np.asarray(predictions)


def log_euclidean_1nn(train: np.ndarray, train_labels: np.ndarray,
                       test: np.ndarray) -> np.ndarray:
    train_logs = np.asarray([log_spd(matrix) for matrix in train])
    predictions = []
    for matrix in test:
        log_matrix = log_spd(matrix)
        distances = np.linalg.norm(train_logs - log_matrix, axis=(1, 2))
        predictions.append(train_labels[np.argmin(distances)])
    return np.asarray(predictions)


def main() -> None:
    image_sets, labels = load_eth80()
    matrices = np.asarray([descriptor(images) for images in image_sets])
    logs = np.asarray([log_spd(matrix) for matrix in matrices])
    rows = []
    rng = np.random.default_rng(42)
    for repeat in range(10):
        train, test = [], []
        for category in np.unique(labels):
            indices = np.flatnonzero(labels == category)
            rng.shuffle(indices)
            train.extend(indices[:5])
            test.extend(indices[5:])
        train = np.asarray(train)
        test = np.asarray(test)
        all_indices = np.concatenate([train, test])
        all_logs = logs[all_indices]
        all_matrices = matrices[all_indices]
        split = len(train)
        pca_classic = PCA(n_components=30, random_state=42).fit_transform(
            all_matrices.reshape(len(all_indices), -1)
        )
        pca_tangent = PCA(n_components=30, random_state=42).fit_transform(
            all_logs.reshape(len(all_indices), -1)
        )
        multi = PCA(n_components=30, random_state=42).fit_transform(
            multi_reference_features(all_logs)
        )
        lie = lie_lle(all_logs, k=6, dimension=30)
        predictions = {
            "PCA classique": one_nn_euclidean(
                pca_classic[:split], labels[train], pca_classic[split:]
            ),
            "PCA tangentielle": one_nn_euclidean(
                pca_tangent[:split], labels[train], pca_tangent[split:]
            ),
            "Repères multiples": one_nn_euclidean(
                multi[:split], labels[train], multi[split:]
            ),
            "Lie-LLE notebook": one_nn_log_spd(
                lie[:split], labels[train], lie[split:]
            ),
        }
        source_features = {
            "PCA classique": all_matrices.reshape(len(all_indices), -1),
            "PCA tangentielle": all_logs.reshape(len(all_indices), -1),
            "Repères multiples": multi_reference_features(all_logs),
            "Lie-LLE notebook": all_logs.reshape(len(all_indices), -1),
        }
        embeddings = {
            "PCA classique": pca_classic,
            "PCA tangentielle": pca_tangent,
            "Repères multiples": multi,
            "Lie-LLE notebook": lie,
        }
        for method, prediction in predictions.items():
            rows.append(
                {"Method": method, "Repeat": repeat,
                 "Accuracy": np.mean(prediction == labels[test]),
                 "Neighbor preservation": neighbor_preservation(
                     source_features[method], embeddings[method]
                 )}
            )
        print(
            f"repeat {repeat + 1:02d}/10 | "
            + " | ".join(
                f"{method}: {rows[-4 + index]['Accuracy'] * 100:.2f}%"
                for index, method in enumerate(predictions)
            )
        )
    results = pd.DataFrame(rows)
    summary = results.groupby("Method").agg(
        Accuracy=("Accuracy", "mean"),
        Accuracy_std=("Accuracy", "std"),
        Neighbor_preservation=("Neighbor preservation", "mean"),
    )
    summary.to_csv(RESULTS)
    display = summary.rename(
        columns={
            "Accuracy": "Accuracy (%)",
            "Accuracy_std": "Accuracy std (%)",
            "Neighbor_preservation": "Neighbours preserved (%)",
        }
    ).mul(100)
    print("\nETH-80 RESULTS (friend notebook protocol)")
    print(display.round(2).to_string())
    print(f"Saved: {RESULTS}")


if __name__ == "__main__":
    main()
