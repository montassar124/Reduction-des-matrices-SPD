"""Evaluate Euclidean PCA and tangent PCA on the MLSP/COBRE dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from spd_reduction.data import build_spd_subjects, load_connectomes
from spd_reduction.pca_methods import euclidean_pca, tangent_pca


ROOT = Path(__file__).resolve().parents[1]
vectors, subject_ids, labels = load_connectomes(ROOT / "train")
spd_subjects, tangent_features = build_spd_subjects(vectors)

embeddings = {
    "PCA on MLSP SPD": euclidean_pca(spd_subjects)[0],
    "Tangent PCA on MLSP": tangent_pca(tangent_features)[0],
}
classifiers = {
    "kNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Logistic regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", random_state=42),
    "Random forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

rows = []
for reducer_name, embedding in embeddings.items():
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
                accuracy_score(labels[test_idx], classifier.predict(embedding[test_idx]))
            )
        rows.append(
            {
                "Dataset": "MLSP/COBRE",
                "Reducer": reducer_name,
                "Classifier": classifier_name,
                "Mean_Accuracy": np.mean(scores),
                "Std_Accuracy": np.std(scores),
                "Repeats": 10,
                "Subjects": len(subject_ids),
            }
        )

results = pd.DataFrame(rows)
output = ROOT / "mlsp_pca_tangent_pca_results.csv"
results.to_csv(output, index=False)

plot_data = results.pivot(index="Classifier", columns="Reducer", values="Mean_Accuracy")
plot_axis = plot_data.plot.bar(
    figsize=(11, 6),
    yerr=results.pivot(index="Classifier", columns="Reducer", values="Std_Accuracy"),
    capsize=4,
    color=["#19b5fe", "#8e7dff"],
    edgecolor="#16324f",
)
plot_axis.set_ylim(0, 1)
plot_axis.set_ylabel("Mean accuracy")
plot_axis.set_title("MLSP/COBRE: PCA versus tangent PCA")
plot_axis.grid(axis="y", alpha=0.25)
plot_axis.tick_params(axis="x", rotation=18)
plot_axis.legend(title="")
figure = ROOT / "figures" / "07_mlsp_pca_vs_tangent_pca.png"
figure.parent.mkdir(exist_ok=True)
plot_axis.get_figure().savefig(figure, dpi=180, bbox_inches="tight")
plt.close(plot_axis.get_figure())

print(f"Dataset: MLSP/COBRE, subjects={len(subject_ids)}, matrices={spd_subjects.shape[1:]}")
print(f"Minimum SPD eigenvalue: {np.linalg.eigvalsh(spd_subjects).min():.6f}")
print(results.round(4).to_string(index=False))
print(f"\nSaved: {output}")
print(f"Saved: {figure}")
