"""Generate controlled SPD cases to compare global and local tangent PCA."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from spd_reduction.local_geometry import (
    aligned_local_tangent_coordinates,
    local_relative_tangent_coordinates,
    matrix_exp_symmetric,
    matrix_log_spd,
    local_tangent_coordinates,
    riemannian_mean,
    affine_invariant_distance,
    summarize_local_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RNG = np.random.default_rng(42)


def random_symmetric(dimension: int, scale: float) -> np.ndarray:
    matrix = RNG.normal(0.0, scale, size=(dimension, dimension))
    return (matrix + matrix.T) / 2.0


def make_case(
    name: str,
    n_subjects: int,
    dimension: int,
    center_separation: float,
    variation: float,
    noise: float,
    n_regions: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Create SPD matrices around one or two separated Riemannian regions."""
    if center_separation:
        centers = [
            matrix_exp_symmetric(random_symmetric(dimension, center_separation))
            for _ in range(n_regions)
        ]
    else:
        centers = [np.eye(dimension) for _ in range(n_regions)]

    matrices = []
    labels = []
    for index in range(n_subjects):
        label = index % n_regions
        local = centers[label]
        tangent_noise = random_symmetric(dimension, variation)
        if noise:
            tangent_noise += random_symmetric(dimension, noise)
        matrices.append(
            matrix_exp_symmetric(matrix_log_spd(local) + tangent_noise)
        )
        labels.append(label)
    return np.asarray(matrices), np.asarray(labels)


def evaluate_case(
    matrices: np.ndarray,
    labels: np.ndarray,
    representation: str,
) -> tuple[float, float, int]:
    """Evaluate one representation without fitting on the test fold."""
    scores = []
    selected_references = []
    for seed in range(10):
        train, test = train_test_split(
            np.arange(len(labels)),
            test_size=0.5,
            random_state=seed,
            stratify=labels,
        )
        train_matrices = matrices[train]
        test_matrices = matrices[test]
        if representation == "PCA classique":
            train_features = train_matrices.reshape(len(train), -1)
            test_features = test_matrices.reshape(len(test), -1)
        else:
            global_center = riemannian_mean(train_matrices)
            if representation == "Tangent PCA globale":
                centers = np.asarray([global_center])
                train_assignments = np.zeros(len(train), dtype=int)
                test_assignments = np.zeros(len(test), dtype=int)
            else:
                tangent_train = local_tangent_coordinates(
                    train_matrices, np.asarray([global_center]),
                    np.zeros(len(train), dtype=int)
                )
                best_k = 1
                best_score = 0.0
                best_assignments = np.zeros(len(train), dtype=int)
                for candidate in range(2, 7):
                    model = KMeans(
                        n_clusters=candidate, n_init=5, random_state=42
                    )
                    candidate_assignments = model.fit_predict(tangent_train)
                    sizes = np.bincount(
                        candidate_assignments, minlength=candidate
                    )
                    if sizes.min() < max(10, len(train) // (candidate * 5)):
                        continue
                    score = silhouette_score(
                        tangent_train, candidate_assignments, sample_size=300,
                        random_state=42
                    )
                    if score > best_score and score > 0.10:
                        best_k = candidate
                        best_score = score
                        best_assignments = candidate_assignments
                if best_k == 1:
                    centers = np.asarray([global_center])
                    train_assignments = best_assignments
                else:
                    train_assignments = best_assignments
                    centers = np.asarray([
                        riemannian_mean(train_matrices[train_assignments == cluster])
                        for cluster in range(best_k)
                    ])
                test_assignments = np.argmin(
                    [
                        [
                            affine_invariant_distance(matrix, center)
                            for center in centers
                        ]
                        for matrix in test_matrices
                    ],
                    axis=1,
                )
                selected_references.append(best_k)
            train_features = aligned_local_tangent_coordinates(
                train_matrices, centers, train_assignments, global_center
            )
            test_features = aligned_local_tangent_coordinates(
                test_matrices, centers, test_assignments, global_center
            )
        scaler = StandardScaler().fit(train_features)
        reducer = PCA(n_components=2, random_state=42).fit(
            scaler.transform(train_features)
        )
        train_embedding = reducer.transform(scaler.transform(train_features))
        test_embedding = reducer.transform(scaler.transform(test_features))
        model = LogisticRegression(max_iter=2000, random_state=42)
        model.fit(train_embedding, labels[train])
        scores.append(
            accuracy_score(labels[test], model.predict(test_embedding))
        )
    references = int(round(np.mean(selected_references))) if selected_references else 1
    return float(np.mean(scores)), float(np.std(scores)), references


def main() -> None:
    cases = [
        ("Proche d'un seul centre", 0.0, 0.18, 0.05, 2),
        ("Deux régions qui se chevauchent", 0.45, 0.35, 0.10, 2),
        ("Deux régions séparées", 1.20, 0.20, 0.05, 2),
        ("Quatre régions avec recouvrement", 0.90, 0.35, 0.10, 4),
        ("Quatre régions éloignées", 1.80, 0.18, 0.05, 4),
    ]
    rows = []
    generated_data = {}
    for name, separation, variation, noise, n_regions in cases:
        matrices, labels = make_case(
            name, 1000, 8, separation, variation, noise, n_regions
        )
        key = name.replace(" ", "_").replace("'", "").replace("é", "e")
        generated_data[f"{key}_matrices"] = matrices
        generated_data[f"{key}_labels"] = labels
        summary, centers, assignments, _ = summarize_local_geometry(matrices)
        measured = {}
        for representation in (
            "PCA classique",
            "Tangent PCA globale",
            "Repères adaptatifs",
        ):
            measured[representation] = evaluate_case(
                matrices, labels, representation
            )
        rows.extend(
            [
                {
                    "Case": name,
                    "Representation": "PCA classique",
                    "Mean accuracy": measured["PCA classique"][0],
                    "Std": measured["PCA classique"][1],
                    "References": 1,
                    "Silhouette": np.nan,
                },
                {
                    "Case": name,
                    "Representation": "Tangent PCA globale",
                    "Mean accuracy": measured["Tangent PCA globale"][0],
                    "Std": measured["Tangent PCA globale"][1],
                    "References": 1,
                    "Silhouette": summary["silhouette_scores"].get(2, np.nan),
                },
                {
                    "Case": name,
                    "Representation": "Repères adaptatifs",
                    "Mean accuracy": measured["Repères adaptatifs"][0],
                    "Std": measured["Repères adaptatifs"][1],
                    "References": measured["Repères adaptatifs"][2],
                    "Silhouette": summary["silhouette_scores"].get(
                        summary["n_centers"], np.nan
                    ),
                },
            ]
        )

    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "synthetic_geometry_comparison.csv", index=False)
    np.savez_compressed(ROOT / "synthetic_geometry_cases.npz", **generated_data)
    pivot = results.pivot(
        index="Case", columns="Representation", values="Mean accuracy"
    )
    axis = pivot.plot.bar(
        figsize=(12, 5.5),
        color=["#8e7dff", "#ff9f43"],
        edgecolor="#16324f",
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy moyenne")
    axis.set_title("Comparaison contrôlée de la géométrie locale")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=18)
    axis.legend(title="")
    figure = FIGURES / "10_synthetic_geometry_comparison.png"
    figure.parent.mkdir(exist_ok=True)
    axis.get_figure().savefig(figure, dpi=180, bbox_inches="tight")
    plt.close(axis.get_figure())
    print(results.round(4).to_string(index=False))
    print(f"\nSaved: {ROOT / 'synthetic_geometry_comparison.csv'}")
    print(f"Saved: {ROOT / 'synthetic_geometry_cases.npz'}")
    print(f"Saved: {figure}")


if __name__ == "__main__":
    main()
