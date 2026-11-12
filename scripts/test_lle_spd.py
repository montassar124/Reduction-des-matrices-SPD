"""Compare PCA, tangent PCA, and LLE on small SPD matrices and MNIST."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from spd_reduction.local_geometry import matrix_exp_symmetric, matrix_log_spd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RNG = np.random.default_rng(42)


def symmetric_features(matrices: np.ndarray) -> np.ndarray:
    rows, cols = np.triu_indices(matrices.shape[1], k=1)
    return matrices[:, rows, cols]


def log_features(matrices: np.ndarray) -> np.ndarray:
    return symmetric_features(
        np.asarray([matrix_log_spd(matrix) for matrix in matrices])
    )


def make_spd_case(
    samples: int = 500,
    dimension: int = 3,
    clusters: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    centers = []
    for _ in range(clusters):
        random_matrix = RNG.normal(0.0, 1.3, size=(dimension, dimension))
        centers.append(matrix_exp_symmetric((random_matrix + random_matrix.T) / 2.0))
    matrices = []
    labels = []
    for index in range(samples):
        label = index % clusters
        noise = RNG.normal(0.0, 0.18, size=(dimension, dimension))
        noise = (noise + noise.T) / 2.0
        matrices.append(matrix_exp_symmetric(matrix_log_spd(centers[label]) + noise))
        labels.append(label % 2)
    return np.asarray(matrices), np.asarray(labels)


def mnist_to_spd(images: np.ndarray) -> np.ndarray:
    """Convert each image to a regularized 3x3 weighted feature covariance."""
    coordinates = np.indices((28, 28)).reshape(2, -1).T.astype(float)
    coordinates -= coordinates.mean(axis=0)
    matrices = []
    for image in images:
        weights = image.astype(float) + 1e-3
        features = np.column_stack(
            [coordinates, image.astype(float).reshape(-1) / 255.0]
        )
        centered = features - np.average(features, axis=0, weights=weights)
        covariance = (centered * weights[:, None]).T @ centered / weights.sum()
        scale = max(float(np.trace(covariance)), 1.0)
        matrices.append(covariance + (1e-3 * scale) * np.eye(3))
    return np.asarray(matrices)


def fit_embedding(
    train_features: np.ndarray,
    test_features: np.ndarray,
    method: str,
    neighbors: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(train_features)
    train_scaled = scaler.transform(train_features)
    test_scaled = scaler.transform(test_features)
    if method == "PCA":
        model = PCA(n_components=2, random_state=42)
    else:
        model = LocallyLinearEmbedding(
            n_neighbors=neighbors,
            n_components=2,
            method="standard",
            eigen_solver="arpack",
            random_state=42,
        )
    train_embedding = model.fit_transform(train_scaled)
    test_embedding = model.transform(test_scaled)
    return train_embedding, test_embedding


def evaluate_dataset(
    name: str,
    matrices: np.ndarray,
    labels: np.ndarray,
    neighbors: int = 12,
) -> pd.DataFrame:
    rows = []
    for seed in range(5):
        train, test = train_test_split(
            np.arange(len(labels)),
            test_size=0.5,
            random_state=seed,
            stratify=labels,
        )
        features = log_features(matrices)
        for method in ("PCA", "LLE"):
            train_embedding, test_embedding = fit_embedding(
                features[train], features[test], method, neighbors
            )
            classifier = KNeighborsClassifier(n_neighbors=5)
            classifier.fit(train_embedding, labels[train])
            rows.append(
                {
                    "Dataset": name,
                    "Method": method,
                    "Seed": seed,
                    "Accuracy": accuracy_score(
                        labels[test], classifier.predict(test_embedding)
                    ),
                    "Matrices": len(labels),
                    "Dimension": matrices.shape[1],
                    "Neighbors": neighbors if method == "LLE" else 0,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    spd_matrices, spd_labels = make_spd_case()
    results = [evaluate_dataset("SPD synthétique 3x3", spd_matrices, spd_labels)]

    try:
        mnist = fetch_openml(
            "mnist_784", version=1, as_frame=False
        )
        images = mnist.data[:500].reshape(-1, 28, 28)
        labels = np.asarray(mnist.target[:500], dtype=int) % 2
        mnist_matrices = mnist_to_spd(images)
        results.append(evaluate_dataset("MNIST -> SPD 3x3", mnist_matrices, labels))
    except Exception as error:
        print(f"MNIST non disponible: {error}")

    table = pd.concat(results, ignore_index=True)
    table.to_csv(ROOT / "lle_spd_results.csv", index=False)
    summary = table.groupby(["Dataset", "Method"])["Accuracy"].agg(["mean", "std"])
    print(summary.round(4).to_string())

    plot = summary["mean"].unstack()
    axis = plot.plot.bar(
        figsize=(9, 5), color=["#19b5fe", "#ff9f43"], edgecolor="#16324f"
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy moyenne")
    axis.set_title("PCA versus LLE sur des représentations SPD")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=15)
    axis.legend(title="")
    FIGURES.mkdir(exist_ok=True)
    axis.get_figure().savefig(
        FIGURES / "11_lle_spd_comparison.png", dpi=180, bbox_inches="tight"
    )
    plt.close(axis.get_figure())
    print(f"Saved: {ROOT / 'lle_spd_results.csv'}")


if __name__ == "__main__":
    main()
