"""Run the modular SPD reduction experiment from the repository root."""

from pathlib import Path

import numpy as np
import torch

from spd_reduction.autoencoder import (
    group_features_sym_kl,
    train_autoencoder,
    train_miaf,
)
from spd_reduction.data import build_spd_subjects, load_connectomes
from spd_reduction.evaluation import repeated_knn_evaluation
from spd_reduction.pca_methods import euclidean_pca, tangent_pca
from spd_reduction.visualization import (
    plot_accuracy,
    plot_embeddings,
    plot_full_summary,
    plot_pipeline,
    plot_spd_spectrum,
)


ROOT = Path(__file__).resolve().parents[1]
vectors, _, labels = load_connectomes(ROOT / "train")
spd_subjects, tangent_features = build_spd_subjects(vectors)

Z_pca, _ = euclidean_pca(spd_subjects)
Z_tangent_pca, _ = tangent_pca(tangent_features)

ae_model, _ = train_autoencoder(tangent_features)
with torch.no_grad():
    Z_autoencoder = ae_model.encoder(torch.from_numpy(tangent_features)).numpy()

groups = group_features_sym_kl(tangent_features)
miaf_model, _ = train_miaf(tangent_features, groups)
grouped = [torch.from_numpy(tangent_features[:, group]) for group in groups]
with torch.no_grad():
    _, Z_miaf = miaf_model(grouped)

embeddings = {
    "PCA_on_SPD": Z_pca,
    "Tangent_PCA": Z_tangent_pca,
    "SPD_Autoencoder": Z_autoencoder,
    "SPD_MIAF": Z_miaf.numpy(),
}
results = repeated_knn_evaluation(embeddings, labels)
output_path = ROOT / "spd_subject_reduction_results.csv"
results.to_csv(output_path, index=False)
figure_dir = ROOT / "figures"
figure_dir.mkdir(exist_ok=True)
plot_pipeline(figure_dir / "01_pipeline.png")
plot_spd_spectrum(spd_subjects, figure_dir / "02_spd_eigenvalues.png")
plot_embeddings(embeddings, labels, figure_dir / "03_subject_embeddings.png")
plot_accuracy(results, figure_dir / "04_accuracy_comparison.png")
plot_full_summary(
    results,
    labels,
    spd_subjects,
    figure_dir / "05_full_results_summary.png",
)
print(results.round(4).to_string(index=False))
print(f"\nSaved: {output_path}")
print(f"Figures saved in: {figure_dir}")
