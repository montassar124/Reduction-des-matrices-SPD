"""Benchmark SPD Dimensionality Reduction & Classification on Real Brain Medical Images (rs-fMRI Connectomes).

Dataset:
  - COBRE / Schizophrenia Resting-State fMRI Connectomes (N=86 subjects, 28x28 SPD matrices)
  - Referenced in: Ju, Collas, Bouchard, Thirion (INRIA Parietal / NeuroSpin, 2025/2026):
    "Benchmarking External Generalization of SPD Matrix Learning for Resting-State fMRI Connectome Prediction"

Evaluated Reduction Methods:
  1. Données Brutes (Raw SPD Baseline, D=406)
  2. ACP Euclidienne Naïve (Vectorized Connectome -> PCA)
  3. PCA Tangente Riemannienne (t-PCA at Fréchet Mean)
  4. Multi-Repères Adaptatifs (Local Reference Frames with Riemannian Clustering)

Evaluated Classifiers (as per internship project proposal):
  1. k-NN Euclidien (k=5)
  2. k-NN Riemannien (AIRM Geodesic Distance)
  3. SVM Linéaire
  4. SVM à Noyau Riemannien / RBF
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spd_reduction.geometry_metrics import (
    airm_dist,
    frechet_mean_regularized,
    pairwise_airm,
    spd_inv_sqrt,
    spd_log,
    spd_sqrt,
    vectorize_batch_isometric,
)
from spd_reduction.local_geometry import (
    matrix_exp_symmetric,
    matrix_log_spd,
    log_map,
    parallel_transport,
    affine_invariant_distance,
    riemannian_kmeans,
    aligned_local_tangent_coordinates,
)
from spd_reduction.multi_reference import (
    fit_multi_reference_tangent,
    transform_multi_reference_tangent,
)


def load_cobre_fnc_data():
    """Load COBRE fMRI Functional Network Connectivity (FNC) data and reconstruct 28x28 SPD matrices."""
    fnc_path = ROOT / "data" / "train" / "train_FNC.csv"
    labels_path = ROOT / "data" / "train" / "train_labels.csv"

    fnc_df = pd.read_csv(fnc_path)
    labels_df = pd.read_csv(labels_path)

    feat_cols = [c for c in fnc_df.columns if c != "Id"]
    X_fnc = fnc_df[feat_cols].values
    y = labels_df["Class"].values

    n_subjects = len(y)
    n_regions = 28
    u_idx = np.triu_indices(n_regions, k=1)

    matrices = []
    for i in range(n_subjects):
        m = np.eye(n_regions)
        m[u_idx] = X_fnc[i]
        m = m + m.T - np.eye(n_regions)
        # Ensure positive-definiteness via spectral clipping floor tau = 1e-12
        vals, vecs = np.linalg.eigh(m)
        vals = np.clip(vals, 1e-12, None)
        m_spd = (vecs * vals) @ vecs.T
        matrices.append(m_spd)

    return np.array(matrices), y


def compute_neighborhood_preservation(D_high: np.ndarray, D_low: np.ndarray, k: int) -> float:
    """Compute Neighborhood Preservation (NP @ k) between high-d and low-d distance matrices."""
    n = D_high.shape[0]
    scores = []
    for i in range(n):
        high_order = np.argsort(D_high[i])
        low_order = np.argsort(D_low[i])
        high_neighs = set(high_order[1 : k + 1])
        low_neighs = set(low_order[1 : k + 1])
        overlap = len(high_neighs.intersection(low_neighs)) / float(k)
        scores.append(overlap)
    return float(np.mean(scores))


def compute_geodesic_distortion(D_high: np.ndarray, D_low: np.ndarray) -> float:
    """Compute relative geodesic distortion ||D_low - D_high||_F / ||D_high||_F."""
    n = D_high.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    dh = D_high[triu_idx]
    dl = D_low[triu_idx]
    scale = np.dot(dh, dl) / (np.dot(dl, dl) + 1e-12)
    diff = scale * dl - dh
    return float(np.linalg.norm(diff) / (np.linalg.norm(dh) + 1e-12))


def run_brain_medical_benchmark(target_dim: int = 5, n_splits: int = 5):
    print("=" * 85)
    print("BENCHMARK : REDUCTION DE DIMENSION SUR IMAGES MEDICALES DU CERVEAU (rs-fMRI)")
    print("Dataset: COBRE Schizophrenia Connectomes (N=86, 28x28 SPD Correlation Matrices)")
    print("Paper: Ju, Collas, Bouchard, Thirion (INRIA Parietal / NeuroSpin, 2025/2026)")
    print(f"Target dimension d = {target_dim}, Cross-Validation = {n_splits}-fold Stratified")
    print("=" * 85)

    X, y = load_cobre_fnc_data()
    n_samples, n, _ = X.shape
    print(f"Loaded {n_samples} connectomes of size {n}x{n} SPD.")
    print(f"Class distribution: {np.sum(y == 0)} Healthy Controls (0) vs {np.sum(y == 1)} Schizophrenia Patients (1)")

    # 1. High-dimensional Ground-Truth AIRM Geodesic Distance Matrix
    print("\nComputing High-Dimensional Ground-Truth AIRM Geodesic Distances...")
    D_airm_high = pairwise_airm(X)
    print(f"AIRM Distances: Min={D_airm_high[D_airm_high>0].min():.4f}, Max={D_airm_high.max():.4f}, Median={np.median(D_airm_high[D_airm_high>0]):.4f}")

    # Gamma for Riemannian RBF Kernel: exp(-gamma * d_R^2)
    median_d = np.median(D_airm_high[D_airm_high > 0])
    gamma_riem = 1.0 / (median_d ** 2)
    K_airm_high = np.exp(-gamma_riem * (D_airm_high ** 2))

    # Vectorized raw representation (isometry)
    X_vec_raw = vectorize_batch_isometric(X)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    methods = [
        "Données Brutes (Baseline D=406)",
        "ACP Euclidienne Naïve",
        "PCA Tangente Riemannienne",
        "Multi-Repères Adaptatifs",
    ]
    classifiers = [
        "k-NN Euclidien",
        "k-NN Riemannien (AIRM)",
        "SVM Linéaire",
        "SVM RBF / Riemannien",
    ]

    results_acc = {m: {c: [] for c in classifiers} for m in methods}
    results_f1 = {m: {c: [] for c in classifiers} for m in methods}
    results_np6 = {m: [] for m in methods}
    results_np12 = {m: [] for m in methods}
    results_dist = {m: [] for m in methods}

    # Store reduced embeddings across full dataset for distance and topology evaluation
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # -------------------------------------------------------------
        # 1. Données Brutes (Raw Baseline)
        # -------------------------------------------------------------
        knn_euc_raw = KNeighborsClassifier(n_neighbors=5).fit(X_vec_raw[train_idx], y_tr)
        results_acc["Données Brutes (Baseline D=406)"]["k-NN Euclidien"].append(
            accuracy_score(y_te, knn_euc_raw.predict(X_vec_raw[test_idx]))
        )

        D_train_raw = D_airm_high[np.ix_(train_idx, train_idx)]
        D_test_raw = D_airm_high[np.ix_(test_idx, train_idx)]
        knn_riem_raw = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(D_train_raw, y_tr)
        results_acc["Données Brutes (Baseline D=406)"]["k-NN Riemannien (AIRM)"].append(
            accuracy_score(y_te, knn_riem_raw.predict(D_test_raw))
        )

        svm_lin_raw = SVC(kernel="linear", C=1.0).fit(X_vec_raw[train_idx], y_tr)
        results_acc["Données Brutes (Baseline D=406)"]["SVM Linéaire"].append(
            accuracy_score(y_te, svm_lin_raw.predict(X_vec_raw[test_idx]))
        )

        K_tr_raw = K_airm_high[np.ix_(train_idx, train_idx)]
        K_te_raw = K_airm_high[np.ix_(test_idx, train_idx)]
        svm_riem_raw = SVC(kernel="precomputed", C=1.0).fit(K_tr_raw, y_tr)
        results_acc["Données Brutes (Baseline D=406)"]["SVM RBF / Riemannien"].append(
            accuracy_score(y_te, svm_riem_raw.predict(K_te_raw))
        )

        results_np6["Données Brutes (Baseline D=406)"].append(1.0)
        results_np12["Données Brutes (Baseline D=406)"].append(1.0)
        results_dist["Données Brutes (Baseline D=406)"].append(0.0)

        # -------------------------------------------------------------
        # 2. ACP Euclidienne Naïve
        # -------------------------------------------------------------
        scaler_acp = StandardScaler().fit(X_vec_raw[train_idx])
        pca_euc = PCA(n_components=target_dim, random_state=42).fit(scaler_acp.transform(X_vec_raw[train_idx]))
        z_tr_acp = pca_euc.transform(scaler_acp.transform(X_vec_raw[train_idx]))
        z_te_acp = pca_euc.transform(scaler_acp.transform(X_vec_raw[test_idx]))
        z_all_acp = pca_euc.transform(scaler_acp.transform(X_vec_raw))

        D_low_acp = np.linalg.norm(z_all_acp[:, None, :] - z_all_acp[None, :, :], axis=2)
        results_np6["ACP Euclidienne Naïve"].append(compute_neighborhood_preservation(D_airm_high, D_low_acp, 6))
        results_np12["ACP Euclidienne Naïve"].append(compute_neighborhood_preservation(D_airm_high, D_low_acp, 12))
        results_dist["ACP Euclidienne Naïve"].append(compute_geodesic_distortion(D_airm_high, D_low_acp))

        knn_acp = KNeighborsClassifier(n_neighbors=5).fit(z_tr_acp, y_tr)
        results_acc["ACP Euclidienne Naïve"]["k-NN Euclidien"].append(accuracy_score(y_te, knn_acp.predict(z_te_acp)))

        D_te_tr_acp = np.linalg.norm(z_te_acp[:, None, :] - z_tr_acp[None, :, :], axis=2)
        D_tr_tr_acp = np.linalg.norm(z_tr_acp[:, None, :] - z_tr_acp[None, :, :], axis=2)
        knn_dist_acp = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(D_tr_tr_acp, y_tr)
        results_acc["ACP Euclidienne Naïve"]["k-NN Riemannien (AIRM)"].append(accuracy_score(y_te, knn_dist_acp.predict(D_te_tr_acp)))

        svm_lin_acp = SVC(kernel="linear", C=1.0).fit(z_tr_acp, y_tr)
        results_acc["ACP Euclidienne Naïve"]["SVM Linéaire"].append(accuracy_score(y_te, svm_lin_acp.predict(z_te_acp)))

        svm_rbf_acp = SVC(kernel="rbf", C=1.0).fit(z_tr_acp, y_tr)
        results_acc["ACP Euclidienne Naïve"]["SVM RBF / Riemannien"].append(accuracy_score(y_te, svm_rbf_acp.predict(z_te_acp)))

        # -------------------------------------------------------------
        # 3. PCA Tangente Riemannienne (t-PCA)
        # -------------------------------------------------------------
        gamma_ref = frechet_mean_regularized(X_tr)
        gamma_sqrt = spd_sqrt(gamma_ref)
        gamma_inv_sqrt = spd_inv_sqrt(gamma_ref)

        def project_tangent(mats):
            tangents = []
            for P in mats:
                P_norm = gamma_inv_sqrt @ P @ gamma_inv_sqrt
                log_P = spd_log(P_norm)
                t = gamma_sqrt @ log_P @ gamma_sqrt
                tangents.append(t)
            return np.array(tangents)

        T_tr = project_tangent(X_tr)
        T_te = project_tangent(X_te)
        T_all = project_tangent(X)

        v_tr_tpca = vectorize_batch_isometric(T_tr)
        v_te_tpca = vectorize_batch_isometric(T_te)
        v_all_tpca = vectorize_batch_isometric(T_all)

        scaler_tpca = StandardScaler().fit(v_tr_tpca)
        pca_tangent = PCA(n_components=target_dim, random_state=42).fit(scaler_tpca.transform(v_tr_tpca))
        z_tr_tpca = pca_tangent.transform(scaler_tpca.transform(v_tr_tpca))
        z_te_tpca = pca_tangent.transform(scaler_tpca.transform(v_te_tpca))
        z_all_tpca = pca_tangent.transform(scaler_tpca.transform(v_all_tpca))

        D_low_tpca = np.linalg.norm(z_all_tpca[:, None, :] - z_all_tpca[None, :, :], axis=2)
        results_np6["PCA Tangente Riemannienne"].append(compute_neighborhood_preservation(D_airm_high, D_low_tpca, 6))
        results_np12["PCA Tangente Riemannienne"].append(compute_neighborhood_preservation(D_airm_high, D_low_tpca, 12))
        results_dist["PCA Tangente Riemannienne"].append(compute_geodesic_distortion(D_airm_high, D_low_tpca))

        knn_tpca = KNeighborsClassifier(n_neighbors=5).fit(z_tr_tpca, y_tr)
        results_acc["PCA Tangente Riemannienne"]["k-NN Euclidien"].append(accuracy_score(y_te, knn_tpca.predict(z_te_tpca)))

        D_te_tr_tpca = np.linalg.norm(z_te_tpca[:, None, :] - z_tr_tpca[None, :, :], axis=2)
        D_tr_tr_tpca = np.linalg.norm(z_tr_tpca[:, None, :] - z_tr_tpca[None, :, :], axis=2)
        knn_dist_tpca = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(D_tr_tr_tpca, y_tr)
        results_acc["PCA Tangente Riemannienne"]["k-NN Riemannien (AIRM)"].append(accuracy_score(y_te, knn_dist_tpca.predict(D_te_tr_tpca)))

        svm_lin_tpca = SVC(kernel="linear", C=1.0).fit(z_tr_tpca, y_tr)
        results_acc["PCA Tangente Riemannienne"]["SVM Linéaire"].append(accuracy_score(y_te, svm_lin_tpca.predict(z_te_tpca)))

        svm_rbf_tpca = SVC(kernel="rbf", C=1.0).fit(z_tr_tpca, y_tr)
        results_acc["PCA Tangente Riemannienne"]["SVM RBF / Riemannien"].append(accuracy_score(y_te, svm_rbf_tpca.predict(z_te_tpca)))

        # -------------------------------------------------------------
        # 4. Multi-Repères Adaptatifs (Local Reference Frames)
        # -------------------------------------------------------------
        # Fit K local barycenters via Riemannian K-Means
        K_clusters = 2
        centers, assignments_tr, _ = riemannian_kmeans(X_tr, n_centers=K_clusters, random_state=42)

        def project_multi_ref(mats):
            all_vecs = []
            for P in mats:
                # Find closest center
                dists = [affine_invariant_distance(P, C) for C in centers]
                k_best = int(np.argmin(dists))
                C_k = centers[k_best]
                # Log map to local tangent space
                t_k = log_map(C_k, P)
                all_vecs.append(vectorize_batch_isometric(t_k[None, :, :])[0])
            return np.array(all_vecs)

        v_tr_mref = project_multi_ref(X_tr)
        v_te_mref = project_multi_ref(X_te)
        v_all_mref = project_multi_ref(X)

        scaler_mref = StandardScaler().fit(v_tr_mref)
        pca_mref = PCA(n_components=target_dim, random_state=42).fit(scaler_mref.transform(v_tr_mref))
        z_tr_mref = pca_mref.transform(scaler_mref.transform(v_tr_mref))
        z_te_mref = pca_mref.transform(scaler_mref.transform(v_te_mref))
        z_all_mref = pca_mref.transform(scaler_mref.transform(v_all_mref))

        D_low_mref = np.linalg.norm(z_all_mref[:, None, :] - z_all_mref[None, :, :], axis=2)
        results_np6["Multi-Repères Adaptatifs"].append(compute_neighborhood_preservation(D_airm_high, D_low_mref, 6))
        results_np12["Multi-Repères Adaptatifs"].append(compute_neighborhood_preservation(D_airm_high, D_low_mref, 12))
        results_dist["Multi-Repères Adaptatifs"].append(compute_geodesic_distortion(D_airm_high, D_low_mref))

        knn_mref = KNeighborsClassifier(n_neighbors=5).fit(z_tr_mref, y_tr)
        results_acc["Multi-Repères Adaptatifs"]["k-NN Euclidien"].append(accuracy_score(y_te, knn_mref.predict(z_te_mref)))

        D_te_tr_mref = np.linalg.norm(z_te_mref[:, None, :] - z_tr_mref[None, :, :], axis=2)
        D_tr_tr_mref = np.linalg.norm(z_tr_mref[:, None, :] - z_tr_mref[None, :, :], axis=2)
        knn_dist_mref = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(D_tr_tr_mref, y_tr)
        results_acc["Multi-Repères Adaptatifs"]["k-NN Riemannien (AIRM)"].append(accuracy_score(y_te, knn_dist_mref.predict(D_te_tr_mref)))

        svm_lin_mref = SVC(kernel="linear", C=1.0).fit(z_tr_mref, y_tr)
        results_acc["Multi-Repères Adaptatifs"]["SVM Linéaire"].append(accuracy_score(y_te, svm_lin_mref.predict(z_te_mref)))

        svm_rbf_mref = SVC(kernel="rbf", C=1.0).fit(z_tr_mref, y_tr)
        results_acc["Multi-Repères Adaptatifs"]["SVM RBF / Riemannien"].append(accuracy_score(y_te, svm_rbf_mref.predict(z_te_mref)))

    # -----------------------------------------------------------------
    # Compile and Display Results
    # -----------------------------------------------------------------
    print("\n" + "=" * 85)
    print("RESULTATS COMPLETS : REDUCTION x CLASSIFICATION SUR IMAGES MEDICALES DU CERVEAU")
    print("=" * 85)

    rows = []
    for m in methods:
        row = {
            "Méthode": m,
            "Distorsion": np.mean(results_dist[m]),
            "NP @ k=6 (%)": np.mean(results_np6[m]) * 100,
            "NP @ k=12 (%)": np.mean(results_np12[m]) * 100,
        }
        for c in classifiers:
            acc_mean = np.mean(results_acc[m][c]) * 100
            acc_std = np.std(results_acc[m][c]) * 100
            row[f"{c} (%)"] = f"{acc_mean:.2f} ± {acc_std:.2f}"
            row[f"{c}_raw"] = acc_mean
        rows.append(row)

    res_df = pd.DataFrame(rows)

    out_csv = Path(r"C:\Users\Globaltech\.gemini\antigravity-cli\brain\b51bb64e-9dbc-4095-bb24-0686f7a52032\scratch\benchmark_brain_medical_cobre_results.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved benchmark results to: {out_csv}")

    cols_display = ["Méthode", "Distorsion", "NP @ k=6 (%)", "NP @ k=12 (%)", "k-NN Euclidien (%)", "k-NN Riemannien (AIRM) (%)", "SVM Linéaire (%)", "SVM RBF / Riemannien (%)"]
    print("\n" + res_df[cols_display].to_string(index=False))

    return res_df


if __name__ == "__main__":
    run_brain_medical_benchmark(target_dim=5, n_splits=5)
