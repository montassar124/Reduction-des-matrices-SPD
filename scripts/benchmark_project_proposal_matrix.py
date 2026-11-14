"""Benchmark the exact 2x4 (and 4x4) matrix specified in the internship proposal.

Reductions:
  - ACP Euclidienne (Classical PCA)
  - PCA Tangente Riemannienne (Tangent PCA)
  - Repères Adaptatifs (Adaptive Multi-References)
  - Données brutes (Raw SPD Baseline)

Classifiers:
  1. k-NN Euclidien (k=5)
  2. SVM Euclidien (RBF Kernel)
  3. k-NN Riemannien (AIRM Geodesic Distance)
  4. SVM à noyau Riemannien (Geodesic AIRM Kernel)

Datasets:
  - Case 1: Classes éloignées (Distant classes)
  - Case 2: Classes légèrement entremêlées (Moderate overlap)
  - Case 3: Classes fortement entremêlées (Strong overlap)
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
    _initial_centers,
)

RNG = np.random.default_rng(42)


def generate_synthetic_case(
    first: tuple[float, float],
    second: tuple[float, float],
    samples_per_class: int = 150,
    dim: int = 5,
    sphere_radius: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    matrices = []
    labels = []
    for label, (lower, upper) in enumerate((first, second)):
        for _ in range(samples_per_class):
            rand = RNG.uniform(lower, upper, size=(dim, dim))
            sym = 0.5 * (rand + rand.T)
            norm = np.linalg.norm(sym, ord="fro")
            if norm == 0:
                norm = 1.0
            sym = sphere_radius * sym / norm
            matrices.append(matrix_exp_symmetric(sym))
            labels.append(label)
    return np.asarray(matrices), np.asarray(labels)


def project_to_spd(mat: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    sym = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, floor, None)
    return (vecs * vals) @ vecs.T


def run_benchmark():
    cases = {
        "Classes éloignées": ((0.0, 1.0), (1.5, 2.5)),
        "Classes légèrement entremêlées": ((0.0, 1.0), (0.25, 1.25)),
        "Classes fortement entremêlées": ((0.0, 1.0), (0.125, 1.125)),
    }

    all_rows = []

    for case_name, bounds in cases.items():
        print(f"\n=======================================================")
        print(f"Running Case: {case_name}")
        print(f"=======================================================")
        X_all, y_all = generate_synthetic_case(*bounds, samples_per_class=150)
        n_samples = len(y_all)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Storage for fold accuracies
        methods = [
            "ACP Euclidienne",
            "PCA Tangente",
            "Repères Adaptatifs (Multi-repères)",
            "Données brutes (Sans réduction)",
        ]
        classifiers = [
            "k-NN Euclidien",
            "SVM Euclidien (RBF)",
            "k-NN Riemannien (AIRM)",
            "SVM à noyau Riemannien",
        ]

        fold_scores = {
            m: {c: [] for c in classifiers} for m in methods
        }

        # Precompute pairwise AIRM for raw matrices
        D_raw = pairwise_airm(X_all)
        gamma_raw = 1.0 / (np.median(D_raw[D_raw > 0]) ** 2)
        K_raw = np.exp(-gamma_raw * (D_raw ** 2))

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
            y_tr, y_te = y_all[train_idx], y_all[test_idx]
            X_tr, X_te = X_all[train_idx], X_all[test_idx]

            # ---------------------------------------------------------
            # 1. ACP Euclidienne
            # ---------------------------------------------------------
            vec_tr = vectorize_batch_isometric(X_tr)
            vec_te = vectorize_batch_isometric(X_te)
            scaler_acp = StandardScaler().fit(vec_tr)
            pca_euc = PCA(n_components=2, random_state=42).fit(scaler_acp.transform(vec_tr))
            z_tr_acp = pca_euc.transform(scaler_acp.transform(vec_tr))
            z_te_acp = pca_euc.transform(scaler_acp.transform(vec_te))

            # Reconstruct ACP matrices to compute Riemannian distances in subspace
            recon_tr_vec = scaler_acp.inverse_transform(pca_euc.inverse_transform(z_tr_acp))
            recon_te_vec = scaler_acp.inverse_transform(pca_euc.inverse_transform(z_te_acp))
            # Map back to 5x5 SPD
            dim = 5
            u_idx = np.triu_indices(dim)
            def vec_to_spd(v):
                m = np.zeros((dim, dim))
                m[u_idx] = v
                m = 0.5 * (m + m.T)
                return project_to_spd(m)

            recon_X_tr_acp = np.stack([vec_to_spd(v) for v in recon_tr_vec])
            recon_X_te_acp = np.stack([vec_to_spd(v) for v in recon_te_vec])
            recon_X_all_acp = np.empty_like(X_all)
            recon_X_all_acp[train_idx] = recon_X_tr_acp
            recon_X_all_acp[test_idx] = recon_X_te_acp

            D_acp = pairwise_airm(recon_X_all_acp)
            gamma_acp = 1.0 / (np.median(D_acp[D_acp > 0]) ** 2) if np.any(D_acp > 0) else 1.0
            K_acp = np.exp(-gamma_acp * (D_acp ** 2))

            # ACP + Classifiers
            # C1: k-NN Euclidien
            knn_euc = KNeighborsClassifier(n_neighbors=5).fit(z_tr_acp, y_tr)
            fold_scores["ACP Euclidienne"]["k-NN Euclidien"].append(
                accuracy_score(y_te, knn_euc.predict(z_te_acp))
            )
            # C2: SVM Euclidien
            svm_euc = SVC(kernel="rbf", random_state=42).fit(z_tr_acp, y_tr)
            fold_scores["ACP Euclidienne"]["SVM Euclidien (RBF)"].append(
                accuracy_score(y_te, svm_euc.predict(z_te_acp))
            )
            # C3: k-NN Riemannien (on reconstructed SPD subspace)
            knn_riem_acp = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
                D_acp[np.ix_(train_idx, train_idx)], y_tr
            )
            fold_scores["ACP Euclidienne"]["k-NN Riemannien (AIRM)"].append(
                accuracy_score(y_te, knn_riem_acp.predict(D_acp[np.ix_(test_idx, train_idx)]))
            )
            # C4: SVM à noyau Riemannien
            svm_riem_acp = SVC(kernel="precomputed", random_state=42).fit(
                K_acp[np.ix_(train_idx, train_idx)], y_tr
            )
            fold_scores["ACP Euclidienne"]["SVM à noyau Riemannien"].append(
                accuracy_score(y_te, svm_riem_acp.predict(K_acp[np.ix_(test_idx, train_idx)]))
            )

            # ---------------------------------------------------------
            # 2. PCA Tangente Riemannienne
            # ---------------------------------------------------------
            ref_mean = frechet_mean_regularized(X_tr)
            ref_inv_sqrt = spd_inv_sqrt(ref_mean)
            ref_sqrt = spd_sqrt(ref_mean)

            def to_tangent(mats):
                return np.stack([
                    ref_sqrt @ spd_log(ref_inv_sqrt @ M @ ref_inv_sqrt) @ ref_sqrt
                    for M in mats
                ])

            T_tr = to_tangent(X_tr)
            T_te = to_tangent(X_te)
            vec_T_tr = vectorize_batch_isometric(T_tr)
            vec_T_te = vectorize_batch_isometric(T_te)

            pca_tan = PCA(n_components=2, random_state=42).fit(vec_T_tr)
            z_tr_tan = pca_tan.transform(vec_T_tr)
            z_te_tan = pca_tan.transform(vec_T_te)

            # Reconstruct Tangent vectors -> Project back to SPD cone via Exp map
            recon_T_tr_vec = pca_tan.inverse_transform(z_tr_tan)
            recon_T_te_vec = pca_tan.inverse_transform(z_te_tan)

            def vec_to_sym(v):
                m = np.zeros((dim, dim))
                m[u_idx] = v
                off = u_idx[0] != u_idx[1]
                m[u_idx[0][off], u_idx[1][off]] /= np.sqrt(2.0)
                return 0.5 * (m + m.T)

            def from_tangent(T_mats):
                res = []
                for V in T_mats:
                    V_sym = 0.5 * (V + V.T)
                    w, U = np.linalg.eigh(ref_inv_sqrt @ V_sym @ ref_inv_sqrt)
                    exp_v = (U * np.exp(w)) @ U.T
                    res.append(ref_sqrt @ exp_v @ ref_sqrt)
                return np.stack(res)

            recon_X_tr_tan = from_tangent(np.stack([vec_to_sym(v) for v in recon_T_tr_vec]))
            recon_X_te_tan = from_tangent(np.stack([vec_to_sym(v) for v in recon_T_te_vec]))
            recon_X_all_tan = np.empty_like(X_all)
            recon_X_all_tan[train_idx] = recon_X_tr_tan
            recon_X_all_tan[test_idx] = recon_X_te_tan

            D_tan = pairwise_airm(recon_X_all_tan)
            gamma_tan = 1.0 / (np.median(D_tan[D_tan > 0]) ** 2) if np.any(D_tan > 0) else 1.0
            K_tan = np.exp(-gamma_tan * (D_tan ** 2))

            # Tangent PCA + Classifiers
            # C1: k-NN Euclidien
            knn_euc_tan = KNeighborsClassifier(n_neighbors=5).fit(z_tr_tan, y_tr)
            fold_scores["PCA Tangente"]["k-NN Euclidien"].append(
                accuracy_score(y_te, knn_euc_tan.predict(z_te_tan))
            )
            # C2: SVM Euclidien
            svm_euc_tan = SVC(kernel="rbf", random_state=42).fit(z_tr_tan, y_tr)
            fold_scores["PCA Tangente"]["SVM Euclidien (RBF)"].append(
                accuracy_score(y_te, svm_euc_tan.predict(z_te_tan))
            )
            # C3: k-NN Riemannien
            knn_riem_tan = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
                D_tan[np.ix_(train_idx, train_idx)], y_tr
            )
            fold_scores["PCA Tangente"]["k-NN Riemannien (AIRM)"].append(
                accuracy_score(y_te, knn_riem_tan.predict(D_tan[np.ix_(test_idx, train_idx)]))
            )
            # C4: SVM à noyau Riemannien
            svm_riem_tan = SVC(kernel="precomputed", random_state=42).fit(
                K_tan[np.ix_(train_idx, train_idx)], y_tr
            )
            fold_scores["PCA Tangente"]["SVM à noyau Riemannien"].append(
                accuracy_score(y_te, svm_riem_tan.predict(K_tan[np.ix_(test_idx, train_idx)]))
            )

            # ---------------------------------------------------------
            # 3. Repères Adaptatifs (Multi-repères)
            # ---------------------------------------------------------
            D_tr = D_raw[np.ix_(train_idx, train_idx)]
            init_c_idx = _initial_centers(D_tr, n_centers=2, random_state=42)
            centers = X_tr[init_c_idx].copy()
            for _ in range(15):
                dists_to_c = np.asarray([
                    [affine_invariant_distance(M, C) for C in centers]
                    for M in X_tr
                ])
                assigns_tr = np.argmin(dists_to_c, axis=1)
                updated = []
                for cluster in range(2):
                    members = X_tr[assigns_tr == cluster]
                    if len(members) == 0:
                        updated.append(centers[cluster])
                    else:
                        updated.append(frechet_mean_regularized(members))
                updated = np.asarray(updated)
                shift = max(affine_invariant_distance(o, n) for o, n in zip(centers, updated))
                centers = updated
                if shift <= 1e-6:
                    break

            dists_te = np.asarray([
                [affine_invariant_distance(M, C) for C in centers]
                for M in X_te
            ])
            assigns_te = np.argmin(dists_te, axis=1)

            def get_aligned_tangent(mats, assigns):
                aligned = []
                for M, a in zip(mats, assigns):
                    local_T = log_map(centers[a], M)
                    transported = parallel_transport(local_T, centers[a], ref_mean)
                    offset = log_map(ref_mean, centers[a])
                    aligned.append(offset + transported)
                return np.stack(aligned)

            T_multi_tr = get_aligned_tangent(X_tr, assigns_tr)
            T_multi_te = get_aligned_tangent(X_te, assigns_te)

            vec_T_multi_tr = vectorize_batch_isometric(T_multi_tr)
            vec_T_multi_te = vectorize_batch_isometric(T_multi_te)

            pca_multi = PCA(n_components=2, random_state=42).fit(vec_T_multi_tr)
            z_tr_multi = pca_multi.transform(vec_T_multi_tr)
            z_te_multi = pca_multi.transform(vec_T_multi_te)

            recon_T_multi_tr_vec = pca_multi.inverse_transform(z_tr_multi)
            recon_T_multi_te_vec = pca_multi.inverse_transform(z_te_multi)

            recon_X_tr_multi = from_tangent(np.stack([vec_to_sym(v) for v in recon_T_multi_tr_vec]))
            recon_X_te_multi = from_tangent(np.stack([vec_to_sym(v) for v in recon_T_multi_te_vec]))
            recon_X_all_multi = np.empty_like(X_all)
            recon_X_all_multi[train_idx] = recon_X_tr_multi
            recon_X_all_multi[test_idx] = recon_X_te_multi

            D_multi = pairwise_airm(recon_X_all_multi)
            gamma_multi = 1.0 / (np.median(D_multi[D_multi > 0]) ** 2) if np.any(D_multi > 0) else 1.0
            K_multi = np.exp(-gamma_multi * (D_multi ** 2))

            # Multi-repères + Classifiers
            knn_euc_multi = KNeighborsClassifier(n_neighbors=5).fit(z_tr_multi, y_tr)
            fold_scores["Repères Adaptatifs (Multi-repères)"]["k-NN Euclidien"].append(
                accuracy_score(y_te, knn_euc_multi.predict(z_te_multi))
            )
            svm_euc_multi = SVC(kernel="rbf", random_state=42).fit(z_tr_multi, y_tr)
            fold_scores["Repères Adaptatifs (Multi-repères)"]["SVM Euclidien (RBF)"].append(
                accuracy_score(y_te, svm_euc_multi.predict(z_te_multi))
            )
            knn_riem_multi = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
                D_multi[np.ix_(train_idx, train_idx)], y_tr
            )
            fold_scores["Repères Adaptatifs (Multi-repères)"]["k-NN Riemannien (AIRM)"].append(
                accuracy_score(y_te, knn_riem_multi.predict(D_multi[np.ix_(test_idx, train_idx)]))
            )
            svm_riem_multi = SVC(kernel="precomputed", random_state=42).fit(
                K_multi[np.ix_(train_idx, train_idx)], y_tr
            )
            fold_scores["Repères Adaptatifs (Multi-repères)"]["SVM à noyau Riemannien"].append(
                accuracy_score(y_te, svm_riem_multi.predict(K_multi[np.ix_(test_idx, train_idx)]))
            )

            # ---------------------------------------------------------
            # 4. Données brutes (Sans réduction / Baseline)
            # ---------------------------------------------------------
            # For raw data, Euclidean distance on vectorized matrices vs. AIRM on matrices
            vec_raw_tr = vectorize_batch_isometric(X_tr)
            vec_raw_te = vectorize_batch_isometric(X_te)
            # C1: k-NN Euclidien
            knn_raw_euc = KNeighborsClassifier(n_neighbors=5).fit(vec_raw_tr, y_tr)
            fold_scores["Données brutes (Sans réduction)"]["k-NN Euclidien"].append(
                accuracy_score(y_te, knn_raw_euc.predict(vec_raw_te))
            )
            # C2: SVM Euclidien
            svm_raw_euc = SVC(kernel="rbf", random_state=42).fit(vec_raw_tr, y_tr)
            fold_scores["Données brutes (Sans réduction)"]["SVM Euclidien (RBF)"].append(
                accuracy_score(y_te, svm_raw_euc.predict(vec_raw_te))
            )
            # C3: k-NN Riemannien
            knn_raw_riem = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
                D_raw[np.ix_(train_idx, train_idx)], y_tr
            )
            fold_scores["Données brutes (Sans réduction)"]["k-NN Riemannien (AIRM)"].append(
                accuracy_score(y_te, knn_raw_riem.predict(D_raw[np.ix_(test_idx, train_idx)]))
            )
            # C4: SVM à noyau Riemannien
            svm_raw_riem = SVC(kernel="precomputed", random_state=42).fit(
                K_raw[np.ix_(train_idx, train_idx)], y_tr
            )
            fold_scores["Données brutes (Sans réduction)"]["SVM à noyau Riemannien"].append(
                accuracy_score(y_te, svm_raw_riem.predict(K_raw[np.ix_(test_idx, train_idx)]))
            )

        # Collect summary for this case
        for m in methods:
            for c in classifiers:
                accs = fold_scores[m][c]
                mean_acc = np.mean(accs)
                std_acc = np.std(accs)
                all_rows.append({
                    "Cas de chevauchement": case_name,
                    "Méthode de réduction": m,
                    "Classifieur": c,
                    "Mean_Accuracy": mean_acc,
                    "Std_Accuracy": std_acc,
                    "Accuracy_Str": f"{mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%"
                })
                print(f"[{case_name}] {m:32s} | {c:25s} = {mean_acc * 100:6.2f}% ± {std_acc * 100:4.2f}%")

    df_results = pd.DataFrame(all_rows)
    scratch_csv = Path(r"C:\Users\Globaltech\.gemini\antigravity-cli\brain\b51bb64e-9dbc-4095-bb24-0686f7a52032\scratch\proposal_full_reduction_classifier_matrix.csv")
    df_results.to_csv(scratch_csv, index=False)
    csv_dir = ROOT / "results"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "proposal_full_reduction_classifier_matrix.csv"
    try:
        df_results.to_csv(csv_path, index=False)
        print(f"\nAll results saved to {csv_path}")
    except Exception as e:
        print(f"Could not write to {csv_path} ({e}), saved to {scratch_csv}")
    return df_results

if __name__ == "__main__":
    run_benchmark()
