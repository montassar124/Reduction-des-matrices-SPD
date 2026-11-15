"""Plot Brain Medical rs-fMRI Connectome Benchmark Results (COBRE Cohort).

Paper: Ju, Collas, Bouchard, Thirion (INRIA Parietal / NeuroSpin, 2025/2026)
"Benchmarking External Generalization of SPD Matrix Learning for Resting-State fMRI Connectome Prediction"
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(r"C:\Users\Globaltech\.gemini\antigravity-cli\brain\b51bb64e-9dbc-4095-bb24-0686f7a52032\scratch")
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Colors
C_NAVY = "#0F224D"
C_ROYAL = "#164E87"
C_CYAN = "#00A8E8"
C_GREEN = "#059669"
C_AMBER = "#D97706"
C_RED = "#DC2626"
C_SLATE = "#64748B"

csv_path = SCRATCH / "benchmark_brain_medical_cobre_results.csv"
df = pd.read_csv(csv_path)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Subplot 1: Grouped bar chart comparing reduction x classifier on Brain Connectomes
methods = [
    "Données Brutes\n(D=406)",
    "ACP Naïve\n(d=5)",
    "PCA Tangente\n(d=5)",
    "Multi-Repères\n(d=5)",
]

# Accuracies
acc_knn_euc = [68.69, 65.16, 56.93, 62.88]
acc_knn_riem = [67.58, 65.16, 56.93, 62.88]
acc_svm_lin = [66.47, 62.75, 61.96, 73.40]
acc_svm_rbf = [73.46, 62.75, 66.34, 64.05]

x = np.arange(len(methods))
width = 0.20

rects1 = axes[0].bar(x - 1.5 * width, acc_knn_euc, width, label="k-NN Euclidien", color="#94A3B8")
rects2 = axes[0].bar(x - 0.5 * width, acc_knn_riem, width, label="k-NN Riemannien (AIRM)", color=C_ROYAL)
rects3 = axes[0].bar(x + 0.5 * width, acc_svm_lin, width, label="SVM Linéaire", color=C_CYAN)
rects4 = axes[0].bar(x + 1.5 * width, acc_svm_rbf, width, label="SVM RBF / Riemannien", color=C_GREEN)

axes[0].set_ylabel("Précision de Classification (%) [5-Fold CV]", fontsize=11, fontweight="bold", color=C_NAVY)
axes[0].set_title("A. Rôle du Classifieur sur Connectomes Cérébraux fMRI (COBRE, N=86)", fontsize=12, fontweight="bold", color=C_NAVY)
axes[0].set_xticks(x)
axes[0].set_xticklabels(methods, fontsize=10, fontweight="bold")
axes[0].set_ylim(45, 82)
axes[0].axhline(73.46, color=C_GREEN, linestyle="--", alpha=0.6, label="Max Baseline (73.46%)")
axes[0].legend(loc="upper right", frameon=True, fontsize=9.5)

# Highlight best on reduced
axes[0].annotate("Multi-Repères + SVM\n73.40% (+10.65% vs ACP)",
                 xy=(3 + 0.5 * width, 73.40), xytext=(2.2, 78),
                 arrowprops=dict(facecolor=C_GREEN, arrowstyle="->", lw=1.5),
                 fontsize=9.5, fontweight="bold", color=C_GREEN,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec=C_GREEN, lw=1))

# Highlight paradox
axes[0].annotate("Chute k-NN (-8.23%)\nParadoxe Antérieur",
                 xy=(2 - 1.5 * width, 56.93), xytext=(1.2, 49),
                 arrowprops=dict(facecolor=C_RED, arrowstyle="->", lw=1.5),
                 fontsize=9, fontweight="bold", color=C_RED,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec=C_RED, lw=1))

# Subplot 2: Geodesic Distortion vs Peak Accuracy
distortions = [0.000, 0.344, 0.324, 0.362]
peak_accs = [73.46, 65.16, 66.34, 73.40]
colors = [C_NAVY, C_SLATE, C_CYAN, C_GREEN]
markers = ["s", "o", "^", "D"]

for i in range(len(methods)):
    axes[1].scatter(distortions[i], peak_accs[i], color=colors[i], s=180, marker=markers[i], zorder=4, label=methods[i].replace('\n', ' '))
    axes[1].annotate(f"{methods[i].split('\n')[0]}\n({peak_accs[i]:.1f}%)",
                     xy=(distortions[i], peak_accs[i]),
                     xytext=(distortions[i] + 0.008, peak_accs[i] - 1.2),
                     fontsize=9.5, fontweight="bold", color=colors[i])

axes[1].set_xlabel("Distorsion Géodésique Relative AIRM", fontsize=11, fontweight="bold", color=C_NAVY)
axes[1].set_ylabel("Précision Maximale Atteinte (%)", fontsize=11, fontweight="bold", color=C_NAVY)
axes[1].set_title("B. Compromis Distorsion Géodésique vs Précision Diagnostique", fontsize=12, fontweight="bold", color=C_NAVY)
axes[1].set_xlim(-0.02, 0.42)
axes[1].set_ylim(60, 78)
axes[1].legend(loc="lower left", frameon=True, fontsize=9.5)

plt.tight_layout()

out_fig = SCRATCH / "21_brain_medical_fnc_cobre_benchmark.png"
fig.savefig(out_fig, dpi=300, bbox_inches="tight")
print(f"Saved figure to {out_fig}")
