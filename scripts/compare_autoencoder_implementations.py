"""Compare the modular autoencoder with the matching final.ipynb baseline."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from spd_reduction.autoencoder import train_autoencoder


ROOT = Path(__file__).resolve().parents[1]
fnc = pd.read_csv(ROOT / "train" / "train_FNC.csv")
labels = pd.read_csv(ROOT / "train" / "train_labels.csv")
features = fnc.drop(columns=["Id"]).to_numpy(dtype=np.float32)
features = np.nan_to_num(StandardScaler().fit_transform(features)).astype(np.float32)
y = labels["Class"].to_numpy(dtype=np.int64)

np.random.seed(42)
torch.manual_seed(42)
model, losses = train_autoencoder(
    features,
    epochs=100,
    latent_dim=2,
    batch_size=32,
    hidden_dims=(features.shape[1] // 2, features.shape[1] // 8, features.shape[1] // 32),
)
with torch.no_grad():
    embedding = model.encoder(torch.from_numpy(features)).numpy()

train_idx, validation_idx = train_test_split(
    np.arange(len(y)),
    test_size=0.25,
    random_state=42,
    stratify=y,
)
classifiers = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Logistic Reg.": LogisticRegression(max_iter=1000, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}
rows = []
for name, classifier in classifiers.items():
    classifier.fit(embedding[train_idx], y[train_idx])
    rows.append(
        {
            "Implementation": "autoencoder.py",
            "Reducer": "AE_baseline_4L",
            "Classifier": name,
            "Accuracy": accuracy_score(
                y[validation_idx], classifier.predict(embedding[validation_idx])
            ),
        }
    )

notebook_results = {
    "KNN (k=5)": 0.7273,
    "Logistic Reg.": 0.6364,
    "SVM (RBF)": 0.5909,
    "Random Forest": 0.6364,
}
for classifier_name, accuracy in notebook_results.items():
    rows.append(
        {
            "Implementation": "final.ipynb",
            "Reducer": "AE_baseline_4L",
            "Classifier": classifier_name,
            "Accuracy": accuracy,
        }
    )

results = pd.DataFrame(rows)
output = ROOT / "autoencoder_comparison.csv"
results.to_csv(output, index=False)
plot_data = results.pivot(index="Classifier", columns="Implementation", values="Accuracy")
axis = plot_data.plot.bar(
    figsize=(10, 5.5),
    color=["#19b5fe", "#8e7dff"],
    edgecolor="#16324f",
)
axis.set_ylim(0, 1)
axis.set_ylabel("Validation accuracy")
axis.set_title("Autoencoder baseline: module vs final.ipynb")
axis.grid(axis="y", alpha=0.25)
axis.legend(title="")
figure = ROOT / "figures" / "06_autoencoder_comparison.png"
figure.parent.mkdir(exist_ok=True)
axis.get_figure().savefig(figure, dpi=180, bbox_inches="tight")
plt.close(axis.get_figure())
print(results.to_string(index=False))
print(f"\nFinal MSE: {losses[-1]:.6f}")
print(f"Saved: {output}")
print(f"Saved: {figure}")
