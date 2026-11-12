"""Compare reductions on spherical log-SPD data with class overlap."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from spd_reduction.local_geometry import (
    aligned_local_tangent_coordinates,
    matrix_exp_symmetric,
    matrix_log_spd,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RNG = np.random.default_rng(42)


def generate_case(
    first: tuple[float, float],
    second: tuple[float, float],
    samples_per_class: int = 1000,
    dimension: int = 5,
    sphere_radius: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    matrices = []
    labels = []
    for label, (lower, upper) in enumerate((first, second)):
        for _ in range(samples_per_class):
            random_matrix = RNG.uniform(
                lower, upper, size=(dimension, dimension)
            )
            symmetric = (random_matrix + random_matrix.T) / 2.0
            norm = np.linalg.norm(symmetric, ord="fro")
            if norm == 0.0:
                raise ValueError("Cannot project a zero log-matrix onto a sphere")
            symmetric = sphere_radius * symmetric / norm
            matrices.append(matrix_exp_symmetric(symmetric))
            labels.append(label)
    return np.asarray(matrices), np.asarray(labels)


def logs(matrices: np.ndarray) -> np.ndarray:
    return np.asarray([matrix_log_spd(matrix) for matrix in matrices])


def upper(matrices: np.ndarray) -> np.ndarray:
    rows, cols = np.triu_indices(matrices.shape[1], k=1)
    return matrices[:, rows, cols]


def neighbor_preservation(
    original: np.ndarray, embedding: np.ndarray, neighbors: int
) -> float:
    """Return the mean percentage of original neighbors retained."""
    count = len(original)
    if count < 2:
        return 1.0
    k = min(neighbors, count - 1)
    original_distances = np.linalg.norm(
        original[:, None, :] - original[None, :, :], axis=2
    )
    embedding_distances = np.linalg.norm(
        embedding[:, None, :] - embedding[None, :, :], axis=2
    )
    original_neighbors = np.argsort(original_distances, axis=1)[:, 1 : k + 1]
    embedding_neighbors = np.argsort(embedding_distances, axis=1)[:, 1 : k + 1]
    retained = [
        len(set(expected).intersection(actual)) / k
        for expected, actual in zip(original_neighbors, embedding_neighbors)
    ]
    return float(np.mean(retained))


class LieLLE:
    """Log-Euclidean Lie-LLE based on the supplied notebook formulation."""

    def __init__(
        self,
        n_neighbors: int = 6,
        n_components: int = 2,
        regularization: float = 1e-3,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.regularization = regularization

    @staticmethod
    def _log_features(matrices: np.ndarray) -> np.ndarray:
        return logs(matrices).reshape(len(matrices), -1)

    def _weights(
        self, reference: np.ndarray, candidates: np.ndarray, exclude_self: bool
    ) -> np.ndarray:
        distances = np.linalg.norm(
            reference[:, None, :] - candidates[None, :, :], axis=2
        )
        weights = np.zeros((len(reference), len(candidates)))
        for row, distance_row in enumerate(distances):
            order = np.argsort(distance_row)
            if exclude_self:
                order = order[order != row]
            neighbors = order[: self.n_neighbors]
            local = candidates[neighbors] - reference[row]
            gram = local @ local.T
            scale = float(np.trace(gram)) / len(neighbors)
            gram += self.regularization * max(scale, 1e-12) * np.eye(
                len(neighbors)
            )
            try:
                local_weights = np.linalg.solve(
                    gram, np.ones(len(neighbors))
                )
            except np.linalg.LinAlgError:
                local_weights = np.ones(len(neighbors))
            total = float(local_weights.sum())
            if abs(total) < 1e-12:
                local_weights = np.ones(len(neighbors)) / len(neighbors)
            else:
                local_weights /= total
            weights[row, neighbors] = local_weights
        return weights

    def fit_transform(self, matrices: np.ndarray) -> np.ndarray:
        self.train_features_ = self._log_features(matrices)
        self.n_neighbors = min(self.n_neighbors, len(matrices) - 1)
        weights = self._weights(
            self.train_features_, self.train_features_, exclude_self=True
        )
        identity = np.eye(len(matrices))
        gram = (identity - weights) @ (identity - weights).T
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)
        self.embedding_ = eigenvectors[:, order[1 : self.n_components + 1]]
        return self.embedding_

    def transform(self, matrices: np.ndarray) -> np.ndarray:
        features = self._log_features(matrices)
        weights = self._weights(features, self.train_features_, exclude_self=False)
        return weights @ self.embedding_


def log_euclidean_center(matrices: np.ndarray) -> np.ndarray:
    return matrix_exp_symmetric(logs(matrices).mean(axis=0))


def fit_adaptive_charts(
    train_matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_logs = logs(train_matrices)
    train_features = upper(train_logs)
    best_k = 1
    best_score = 0.0
    best_labels = np.zeros(len(train_matrices), dtype=int)
    for candidate in range(2, 7):
        labels = KMeans(
            n_clusters=candidate, n_init=5, random_state=42
        ).fit_predict(train_features)
        sizes = np.bincount(labels, minlength=candidate)
        if sizes.min() < max(20, len(train_matrices) // (candidate * 5)):
            continue
        score = silhouette_score(
            train_features, labels, sample_size=400, random_state=42
        )
        if score > best_score and score > 0.10:
            best_k = candidate
            best_score = score
            best_labels = labels
    global_center = matrix_exp_symmetric(train_logs.mean(axis=0))
    if best_k == 1:
        return np.asarray([global_center]), best_labels, np.array([0.0])
    centers = np.asarray(
        [
            matrix_exp_symmetric(
                train_logs[best_labels == cluster].mean(axis=0)
            )
            for cluster in range(best_k)
        ]
    )
    return centers, best_labels, np.array([best_score])


def assign(matrices: np.ndarray, centers: np.ndarray) -> np.ndarray:
    matrix_logs = logs(matrices)
    center_logs = logs(centers)
    distances = np.linalg.norm(
        matrix_logs[:, None] - center_logs[None, :], axis=(2, 3)
    )
    return np.argmin(distances, axis=1)


def evaluate_method(
    matrices: np.ndarray,
    labels: np.ndarray,
    method: str,
    neighbors: int = 6,
) -> tuple[float, float, list[int], float]:
    scores = []
    preservation_scores = []
    reference_counts = []
    for seed in range(10):
        train, test = train_test_split(
            np.arange(len(labels)),
            test_size=0.5,
            random_state=seed,
            stratify=labels,
        )
        train_matrices = matrices[train]
        test_matrices = matrices[test]
        train_logs = logs(train_matrices)
        test_logs = logs(test_matrices)
        if method == "PCA classique":
            train_features = train_matrices.reshape(len(train), -1)
            test_features = test_matrices.reshape(len(test), -1)
        elif method == "PCA tangent":
            train_features = upper(train_logs)
            test_features = upper(test_logs)
        elif method == "Lie-LLE":
            train_features = train_matrices
            test_features = test_matrices
        elif method == "Reperes adaptatifs":
            centers, train_assignments, _ = fit_adaptive_charts(train_matrices)
            test_assignments = assign(test_matrices, centers)
            reference_counts.append(len(centers))
            global_center = matrix_exp_symmetric(train_logs.mean(axis=0))
            train_features = aligned_local_tangent_coordinates(
                train_matrices, centers, train_assignments, global_center
            )
            test_features = aligned_local_tangent_coordinates(
                test_matrices, centers, test_assignments, global_center
            )
        if method == "Lie-LLE":
            reducer = LieLLE(n_neighbors=neighbors, n_components=2)
            train_embedding = reducer.fit_transform(train_features)
            test_embedding = reducer.transform(test_features)
        else:
            scaler = StandardScaler().fit(train_features)
            train_scaled = scaler.transform(train_features)
            test_scaled = scaler.transform(test_features)
            reducer = PCA(n_components=2, random_state=42)
            train_embedding = reducer.fit_transform(train_scaled)
            test_embedding = reducer.transform(test_scaled)
        classifier = KNeighborsClassifier(n_neighbors=5)
        classifier.fit(train_embedding, labels[train])
        scores.append(
            accuracy_score(labels[test], classifier.predict(test_embedding))
        )
        preservation_scores.append(
            neighbor_preservation(
                reducer.train_features_
                if method == "Lie-LLE"
                else train_features,
                train_embedding,
                neighbors,
            )
        )
    return (
        float(np.mean(scores)),
        float(np.std(scores)),
        reference_counts,
        float(np.mean(preservation_scores)),
    )


def main() -> None:
    cases = {
        "Case 2 - moderate overlap": ((0.0, 1.0), (0.25, 1.25)),
        "Case 3 - strong overlap": ((0.0, 1.0), (0.125, 1.125)),
    }
    rows = []
    for name, bounds in cases.items():
        matrices, labels = generate_case(*bounds)
        for method in (
            "PCA classique",
            "PCA tangent",
            "Reperes adaptatifs",
            "Lie-LLE",
        ):
            mean, std, reference_counts, preservation = evaluate_method(
                matrices, labels, method
            )
            rows.append(
                {
                    "Case": name,
                    "Method": method,
                    "Mean accuracy": mean,
                    "Std": std,
                    "Matrices": len(labels),
                    "References mean": np.mean(reference_counts)
                    if reference_counts
                    else 1,
                    "Neighbor preservation": preservation,
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "paper_spd_reductions_comparison.csv", index=False)
    display = results.copy()
    for column in ("Mean accuracy", "Std", "Neighbor preservation"):
        display[column] *= 100
    display = display.rename(
        columns={
            "Mean accuracy": "Accuracy (%)",
            "Std": "Accuracy std (%)",
            "Neighbor preservation": "Neighbours preserved (%)",
        }
    )
    print("\nPAPER-STYLE SYNTHETIC SPD RESULTS")
    print(display.round(2).to_string(index=False))
    chart = results.pivot(index="Case", columns="Method", values="Mean accuracy")
    axis = chart.plot.bar(
        figsize=(12, 5.5),
        color=["#19b5fe", "#8e7dff", "#ff9f43", "#00c49a"],
        edgecolor="#16324f",
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy moyenne")
    axis.set_title("DonnÃ©es du papier : comparaison SPD")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=15)
    axis.legend(title="")
    FIGURES.mkdir(exist_ok=True)
    axis.get_figure().savefig(
        FIGURES / "13_paper_spd_reductions_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(axis.get_figure())
    print(f"Saved: {ROOT / 'paper_spd_reductions_comparison.csv'}")


if __name__ == "__main__":
    main()
