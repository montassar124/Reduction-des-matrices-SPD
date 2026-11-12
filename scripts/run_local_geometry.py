"""Run the multi-reference local geometry analysis on MLSP/COBRE."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from spd_reduction.data import build_spd_subjects, load_connectomes, vectors_to_symmetric
from spd_reduction.local_geometry import (
    aligned_local_tangent_coordinates,
    affine_invariant_distance,
    pairwise_affine_invariant_distances,
    riemannian_mean,
    local_relative_tangent_coordinates,
    summarize_local_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def main() -> None:
    vectors, ids, labels = load_connectomes(ROOT / "train")
    spd, _ = build_spd_subjects(vectors)
    summary, centers, assignments, local_features = summarize_local_geometry(spd)
    common_center = riemannian_mean(spd)
    local_features = aligned_local_tangent_coordinates(
        spd, centers, assignments, common_center
    )

    center_rows = []
    for cluster in range(summary["n_centers"]):
        members = np.flatnonzero(assignments == cluster)
        distances = [
            affine_invariant_distance(spd[index], centers[cluster])
            for index in members
        ]
        center_rows.append(
            {
                "center": cluster,
                "subjects": len(members),
                "mean_distance": np.mean(distances),
                "max_distance": np.max(distances),
            }
        )
    center_table = pd.DataFrame(center_rows)
    center_distance_table = pd.DataFrame(
        pairwise_affine_invariant_distances(centers),
        index=[f"reference_{i}" for i in range(len(centers))],
        columns=[f"reference_{i}" for i in range(len(centers))],
    )
    subject_table = pd.DataFrame(
        {
            "Id": ids,
            "Class": labels,
            "local_center": assignments,
            "distance_to_local_center": [
                affine_invariant_distance(spd[i], centers[assignments[i]])
                for i in range(len(spd))
            ],
        }
    )
    center_table.to_csv(ROOT / "local_geometry_centers.csv", index=False)
    center_distance_table.to_csv(ROOT / "local_geometry_center_distances.csv")
    subject_table.to_csv(ROOT / "local_geometry_subject_assignments.csv", index=False)

    embeddings = {
        "Tangent PCA globale": PCA(n_components=2, random_state=42).fit_transform(
            StandardScaler().fit_transform(build_spd_subjects(vectors)[1])
        ),
        "Repères adaptatifs": PCA(n_components=2, random_state=42).fit_transform(
            StandardScaler().fit_transform(local_features)
        ),
    }
    classifiers = {
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic": LogisticRegression(max_iter=2000, random_state=42),
        "SVM": SVC(kernel="rbf", random_state=42),
        "Random forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }
    accuracy_rows = []
    for reducer, embedding in embeddings.items():
        for classifier_name, classifier_template in classifiers.items():
            scores = []
            for seed in range(10):
                train_idx, test_idx = train_test_split(
                    np.arange(len(labels)),
                    test_size=0.2,
                    random_state=seed,
                    stratify=labels,
                )
                classifier = classifier_template
                classifier.fit(embedding[train_idx], labels[train_idx])
                scores.append(
                    accuracy_score(
                        labels[test_idx], classifier.predict(embedding[test_idx])
                    )
                )
            accuracy_rows.append(
                {
                    "Reducer": reducer,
                    "Classifier": classifier_name,
                    "Mean_Accuracy": np.mean(scores),
                    "Std_Accuracy": np.std(scores),
                    "Repeats": 10,
                    "Subjects": len(ids),
                }
            )
    accuracy_table = pd.DataFrame(accuracy_rows)
    accuracy_table.to_csv(ROOT / "local_geometry_accuracy.csv", index=False)

    pairwise = pairwise_affine_invariant_distances(spd)
    cluster_mean_distance = np.zeros(summary["n_centers"])
    for cluster in range(summary["n_centers"]):
        members = np.flatnonzero(assignments == cluster)
        cluster_mean_distance[cluster] = np.mean(
            [pairwise[i, j] for i in members for j in members if i < j]
        )

    pca_embedding = PCA(n_components=2, random_state=42).fit_transform(local_features)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    axes[0].scatter(
        subject_table["distance_to_local_center"],
        np.zeros(len(subject_table)),
        c=assignments,
        cmap="tab10",
        s=45,
    )
    axes[0].set_title("Distance de chaque sujet à son repère")
    axes[0].set_xlabel("Distance riemannienne")
    axes[0].set_yticks([])

    axes[1].bar(
        np.arange(summary["n_centers"]),
        cluster_mean_distance,
        color=plt.cm.tab10(np.arange(summary["n_centers"])),
    )
    axes[1].set_title("Distance moyenne à l'intérieur de chaque repère")
    axes[1].set_xlabel("Repère local")
    axes[1].set_ylabel("Distance riemannienne")

    for cluster in range(summary["n_centers"]):
        mask = assignments == cluster
        axes[2].scatter(
            pca_embedding[mask, 0],
            pca_embedding[mask, 1],
            label=f"Repère {cluster}",
            s=45,
        )
    axes[2].set_title("PCA des coordonnées tangentes locales")
    axes[2].set_xlabel("Composante 1")
    axes[2].set_ylabel("Composante 2")
    axes[2].legend()
    fig.savefig(FIGURES / "08_local_geometry_multi_reference.png", dpi=180)
    plt.close(fig)

    plot_data = accuracy_table.pivot(
        index="Classifier", columns="Reducer", values="Mean_Accuracy"
    )
    plot_errors = accuracy_table.pivot(
        index="Classifier", columns="Reducer", values="Std_Accuracy"
    )
    axis = plot_data.plot.bar(
        figsize=(11, 5.5), yerr=plot_errors, capsize=4,
        color=["#8e7dff", "#ff9f43"], edgecolor="#16324f",
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy moyenne")
    axis.set_title("Tangent PCA globale versus plusieurs repères locaux")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=18)
    axis.legend(title="")
    axis.get_figure().savefig(
        FIGURES / "09_local_geometry_accuracy.png", dpi=180, bbox_inches="tight"
    )
    plt.close(axis.get_figure())

    print("Nombre de repères retenu :", summary["n_centers"])
    print("Scores silhouette testés :", summary["silhouette_scores"])
    print(center_table.to_string(index=False))
    print("Distances entre les repères :")
    print(center_distance_table.to_string())
    print("Erreur moyenne de reconstruction :", summary["mean_reconstruction_error"])
    print(accuracy_table.round(4).to_string(index=False))
    print("Résultats sauvegardés dans local_geometry_*.csv")


if __name__ == "__main__":
    main()
