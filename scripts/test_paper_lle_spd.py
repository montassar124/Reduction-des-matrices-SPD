"""Reproduce the paper's synthetic SPD protocol with PCA and LLE."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def generate_paper_case(
    lower_first: float,
    upper_first: float,
    lower_second: float,
    upper_second: float,
    samples_per_class: int = 1000,
    dimension: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate SPD data using the paper's symmetrized-uniform protocol."""
    matrices = []
    labels = []
    for label, (lower, upper) in enumerate(
        [
            (lower_first, upper_first),
            (lower_second, upper_second),
        ]
    ):
        for _ in range(samples_per_class):
            random_matrix = RNG.uniform(
                lower, upper, size=(dimension, dimension)
            )
            symmetric = (random_matrix + random_matrix.T) / 2.0
            matrices.append(matrix_exp_symmetric(symmetric))
            labels.append(label)
    return np.asarray(matrices), np.asarray(labels)


def log_upper_features(matrices: np.ndarray) -> np.ndarray:
    rows, cols = np.triu_indices(matrices.shape[1], k=1)
    return np.asarray(
        [matrix_log_spd(matrix)[rows, cols] for matrix in matrices]
    )


def evaluate(
    features: np.ndarray,
    labels: np.ndarray,
    method: str,
    neighbors: int = 12,
) -> tuple[float, float]:
    scores = []
    for seed in range(10):
        train, test = train_test_split(
            np.arange(len(labels)),
            test_size=0.5,
            random_state=seed,
            stratify=labels,
        )
        scaler = StandardScaler().fit(features[train])
        train_scaled = scaler.transform(features[train])
        test_scaled = scaler.transform(features[test])
        if method == "PCA":
            reducer = PCA(n_components=2, random_state=42)
        else:
            reducer = LocallyLinearEmbedding(
                n_neighbors=neighbors,
                n_components=2,
                method="standard",
                eigen_solver="arpack",
                random_state=42,
                n_jobs=-1,
            )
        train_embedding = reducer.fit_transform(train_scaled)
        test_embedding = reducer.transform(test_scaled)
        classifier = KNeighborsClassifier(n_neighbors=5)
        classifier.fit(train_embedding, labels[train])
        scores.append(
            accuracy_score(labels[test], classifier.predict(test_embedding))
        )
    return float(np.mean(scores)), float(np.std(scores))


def main() -> None:
    cases = {
        "Case 1 - distant": (1.0, 2.0, 0.0, 1.0),
        "Case 2 - moderate overlap": (0.0, 1.0, 0.25, 1.25),
        "Case 3 - strong overlap": (0.0, 1.0, 0.125, 1.125),
    }
    rows = []
    generated = {}
    for name, bounds in cases.items():
        matrices, labels = generate_paper_case(*bounds)
        generated[name.replace(" ", "_")] = matrices
        generated[f"{name.replace(' ', '_')}_labels"] = labels
        features = log_upper_features(matrices)
        for method in ("PCA", "LLE"):
            mean, std = evaluate(features, labels, method)
            rows.append(
                {
                    "Case": name,
                    "Method": method,
                    "Mean accuracy": mean,
                    "Std": std,
                    "Matrices": len(labels),
                    "Dimension": matrices.shape[1],
                    "Neighbors": 12 if method == "LLE" else 0,
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "paper_lle_spd_results.csv", index=False)
    np.savez_compressed(ROOT / "paper_synthetic_spd_cases.npz", **generated)
    print(results.round(4).to_string(index=False))

    chart = results.pivot(index="Case", columns="Method", values="Mean accuracy")
    axis = chart.plot.bar(
        figsize=(10, 5), color=["#19b5fe", "#ff9f43"], edgecolor="#16324f"
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy moyenne")
    axis.set_title("Données du papier : PCA versus LLE sur matrices SPD")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=15)
    axis.legend(title="")
    FIGURES.mkdir(exist_ok=True)
    axis.get_figure().savefig(
        FIGURES / "12_paper_lle_spd_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(axis.get_figure())
    print(f"Saved: {ROOT / 'paper_lle_spd_results.csv'}")
    print(f"Saved: {ROOT / 'paper_synthetic_spd_cases.npz'}")


if __name__ == "__main__":
    main()
