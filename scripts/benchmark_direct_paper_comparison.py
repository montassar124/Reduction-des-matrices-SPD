"""Direct comparison between paper Table 3 (Araoud et al. 2025/2026, Springer)
and our experimental tests on the exact same MLSP/COBRE brain connectome dataset.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from spd_reduction.data import load_connectomes, build_spd_subjects
from spd_reduction.local_geometry import (
    summarize_local_geometry,
    aligned_local_tangent_coordinates,
    riemannian_mean,
)

# 1. Load exact connectomes
vectors, ids, labels = load_connectomes(ROOT / "data" / "train")
spd, tangent_features = build_spd_subjects(vectors)

# 2. Features in 2D (as in paper protocol)
flat_spd = spd.reshape(len(spd), -1)
pca_2d = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(flat_spd))
tpca_2d = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(tangent_features))

summary, centers, assignments, local_feat = summarize_local_geometry(spd)
common_center = riemannian_mean(spd)
aligned_local = aligned_local_tangent_coordinates(spd, centers, assignments, common_center)
multiref_2d = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(aligned_local))

# 3. Exact evaluation protocol: 5 repeated runs with 80% train / 20% test
methods = {
    "PCA Euclidienne": pca_2d,
    "PCA Tangente": tpca_2d,
    "Multi-Repères Adaptatifs": multiref_2d,
}

classifiers = {
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "SVM Linéaire": SVC(kernel="linear", random_state=42),
    "SVM RBF": SVC(kernel="rbf", random_state=42),
}

# Values written in paper Table 3 (Araoud, Ghorbel, Ghorbel, ICAART 2025 / Springer 2026, p. 15)
paper_table3_values = {
    "PCA Euclidienne": {"k-NN": 61.11, "Citation": "Araoud et al. 2025/2026 Table 3"},
    "PCA Tangente": {"k-NN": 55.56, "Citation": "Araoud et al. 2025/2026 Table 3"},
    "Harandi (AIRM)": {"k-NN": 55.89, "Citation": "Araoud et al. 2025/2026 Table 3"},
    "Autoencodeur": {"k-NN": 50.00, "Citation": "Araoud et al. 2025/2026 Table 3"},
    "Kernel PCA": {"k-NN": 50.00, "Citation": "Araoud et al. 2025/2026 Table 3"},
    "SPD-Kernel PCA": {"k-NN": 50.00, "Citation": "Araoud et al. 2025/2026 Table 3"},
    "Multi-Repères Adaptatifs": {"k-NN": None, "Citation": "Notre Contribution ENSI 2026"},
}

records = []
for m_name, X in methods.items():
    for c_name, clf in classifiers.items():
        scores = []
        for seed in range(5):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, labels, test_size=0.2, random_state=seed, stratify=labels
            )
            clf.fit(X_tr, y_tr)
            scores.append(clf.score(X_te, y_te) * 100.0)
        
        paper_val = paper_table3_values.get(m_name, {}).get("k-NN", np.nan) if "k-NN" in c_name else np.nan
        
        records.append({
            "Methode": m_name,
            "Classifieur": c_name,
            "Valeur_Papier_Table3 (%)": paper_val,
            "Notre_Test_Moyenne (%)": np.mean(scores),
            "Notre_Test_Ecart_Type (%)": np.std(scores),
            "Pli_1": scores[0],
            "Pli_2": scores[1],
            "Pli_3": scores[2],
            "Pli_4": scores[3],
            "Pli_5": scores[4],
        })

df = pd.DataFrame(records)
scratch_dir = Path(r"C:\Users\Globaltech\.gemini\antigravity-cli\brain\b51bb64e-9dbc-4095-bb24-0686f7a52032\scratch")
scratch_dir.mkdir(parents=True, exist_ok=True)
out_csv = scratch_dir / "direct_paper_comparison_cobre.csv"
df.to_csv(out_csv, index=False)
print("=== RÉSULTATS COMPARATIFS DIRECTS AVEC LA TABLE 3 DU PAPIER ARAOUD ET AL. 2025/2026 ===")
print(df[["Methode", "Classifieur", "Valeur_Papier_Table3 (%)", "Notre_Test_Moyenne (%)", "Notre_Test_Ecart_Type (%)"]].to_string(index=False))
