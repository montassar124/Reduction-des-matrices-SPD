"""Benchmark local geometry and neighborhood preservation under AIRM for SPD reduction.

Compares:
- Classical PCA (Frobenius-isometric vectorization)
- Tangent PCA (at regularized Fréchet mean)

Against ground-truth high-dimensional Affine-Invariant Riemannian Metric (AIRM).
Metrics:
- NP@k (Neighborhood Preservation at k)
- Trustworthiness & Continuity
- Local & Global Distance Distortion
- Spearman Rank Correlation
- Local Tangent LLE Reconstruction Error
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from sklearn.manifold import MDS

from spd_reduction.geometry_metrics import (
    generate_multicluster_spd_dataset,
    pairwise_airm,
    classical_pca_multidim,
    tangent_pca_multidim,
    knn_from_dist,
    knn_from_points,
    neighbor_overlap_per_point,
    np_at_k,
    trustworthiness,
    continuity,
    distance_distortion,
    local_reconstruction_error,
)

ROOT = Path(__file__).resolve().parents[1]

def get_writable_dir(preferred: Path) -> Path:
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        test_file = preferred / ".test_write"
        test_file.touch()
        test_file.unlink()
        return preferred
    except (PermissionError, OSError):
        scratch = Path.home() / ".gemini" / "antigravity-cli" / "brain" / "b51bb64e-9dbc-4095-bb24-0686f7a52032" / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        return scratch

FIGURES = get_writable_dir(ROOT / "figures")
OUTPUT_DIR = get_writable_dir(ROOT)
DELIVERY_RESULTS = ROOT / "SPD_geometry_delivery" / "results"

K_NN = 2
DIMS = [2, 5, 10, 15, 20]
SEED = 42


def main() -> None:
    print("=" * 80)
    print("STEP 1: Generating multi-cluster SPD dataset (28x28, 6 clusters)")
    print("=" * 80)
    x_mats, y_clusters = generate_multicluster_spd_dataset(n_per_cluster=30, dim=28, seed=SEED)
    n_samples = len(x_mats)
    print(f"Generated N={n_samples} matrices across {len(np.unique(y_clusters))} condition clusters.")

    print("\n" + "=" * 80)
    print("STEP 2: Computing pairwise AIRM distance matrix (ground truth geometry)")
    print("=" * 80)
    d_high = pairwise_airm(x_mats)
    print(f"AIRM distances: mean={d_high[d_high > 0].mean():.4f}, max={d_high.max():.4f}")

    print("\n" + "=" * 80)
    print("STEP 3: Performing Classical PCA and Tangent PCA reductions")
    print("=" * 80)
    z_classic_dict = classical_pca_multidim(x_mats, DIMS)
    z_tangent_dict = tangent_pca_multidim(x_mats, DIMS)

    rows: list[dict[str, object]] = []
    for d in DIMS:
        for method, z_dict in [("Classical PCA", z_classic_dict), ("Tangent PCA", z_tangent_dict)]:
            z = z_dict[d]
            np_val = np_at_k(d_high, z, K_NN)
            tw_val = trustworthiness(d_high, z, K_NN)
            cont_val = continuity(d_high, z, K_NN)
            dist_res = distance_distortion(d_high, z, K_NN)
            recon_val = local_reconstruction_error(x_mats, d_high, z, K_NN)

            rows.append({
                "Method": method,
                "d": d,
                "NP@k": round(np_val, 4),
                "Trust": round(tw_val, 4),
                "Cont": round(cont_val, 4),
                "LocDistMean": round(dist_res["local_mean"], 4),
                "LocDistMedian": round(dist_res["local_median"], 4),
                "GlobDistMean": round(dist_res["global_mean"], 4),
                "GlobDistMedian": round(dist_res["global_median"], 4),
                "Spearman": round(dist_res["spearman"], 4),
                "LocRecon": recon_val,
            })

    df_metrics = pd.DataFrame(rows)
    csv_out = OUTPUT_DIR / "neighborhood_preservation_metrics.csv"
    df_metrics.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")
    try:
        if DELIVERY_RESULTS.exists():
            csv_delivery = DELIVERY_RESULTS / "neighborhood_preservation_metrics.csv"
            df_metrics.to_csv(csv_delivery, index=False)
            print(f"Saved: {csv_delivery}")
    except (PermissionError, OSError):
        pass

    print("\n" + "=" * 115)
    print(f"LOCAL GEOMETRY METRICS (k = {K_NN}) -- Higher NP/Trust/Cont/Spearman is better; Lower distortion is better")
    print("=" * 115)
    header = f"{'Method':<14}{'d':>3}  {'NP@k':>7}  {'Trust':>7}  {'Cont':>7}  {'LocDm':>8}  {'LocDmd':>8}  {'GlbDm':>10}  {'GlbDmd':>8}  {'Spear':>7}  {'Recon':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        recon_str = f"{r['LocRecon']:.2e}" if isinstance(r['LocRecon'], float) else str(r['LocRecon'])
        print(
            f"{r['Method']:<14}{r['d']:>3}  "
            f"{r['NP@k']:>7.4f}  {r['Trust']:>7.4f}  {r['Cont']:>7.4f}  "
            f"{r['LocDistMean']:>8.4f}  {r['LocDistMedian']:>8.4f}  "
            f"{r['GlobDistMean']:>10.4f}  {r['GlobDistMedian']:>8.4f}  "
            f"{r['Spearman']:>7.4f}  {recon_str:>10}"
        )
    print("-" * len(header))

    # Figure 1: Metrics vs reduced dimension
    plot_metrics_vs_dimension(rows)

    # Figure 2: Neighborhood graphs at d=2
    plot_neighborhood_graphs(x_mats, y_clusters, d_high, z_classic_dict[2], z_tangent_dict[2])

    # Figure 3 & 4: Per-cluster and per-point overlap
    plot_cluster_and_point_overlap(d_high, z_classic_dict[2], z_tangent_dict[2], y_clusters)


def plot_metrics_vs_dimension(rows: list[dict[str, object]]) -> None:
    metric_configs = [
        ("NP@k", "Neighbor Preservation (NP@k)", True),
        ("Trust", "Trustworthiness", True),
        ("Cont", "Continuity", True),
        ("Spearman", "Spearman rank corr (global)", True),
        ("LocDistMean", "Local distance distortion", False),
        ("GlobDistMean", "Global distance distortion", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, (metric_key, title, _) in zip(axes, metric_configs):
        c_vals = [next(r for r in rows if r["Method"] == "Classical PCA" and r["d"] == d)[metric_key] for d in DIMS]
        t_vals = [next(r for r in rows if r["Method"] == "Tangent PCA" and r["d"] == d)[metric_key] for d in DIMS]

        ax.plot(DIMS, c_vals, "o-", color="#d95f02", lw=2, label="Classical PCA")
        ax.plot(DIMS, t_vals, "s-", color="#1b9e77", lw=2, label="Tangent PCA")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Reduced Dimension d")
        ax.grid(alpha=0.3)

    axes[0].legend(fontsize=9)
    plt.suptitle("Local-Geometry & Neighborhood Preservation Metrics vs Reduced Dimension", fontsize=14)
    plt.tight_layout()
    fig1_path = FIGURES / "17_local_geometry_metrics_vs_dimension.png"
    plt.savefig(fig1_path, dpi=150)
    plt.close()
    print(f"Saved: {fig1_path}")


def plot_neighborhood_graphs(
    x_mats: np.ndarray,
    y_clusters: np.ndarray,
    d_high: np.ndarray,
    z_classic: np.ndarray,
    z_tangent: np.ndarray,
) -> None:
    n_samples = len(x_mats)
    knn_high = knn_from_dist(d_high, K_NN)
    knn_class = knn_from_points(z_classic, K_NN)
    knn_tang = knn_from_points(z_tangent, K_NN)

    z_airm_mds = MDS(
        n_components=2,
        metric=True,
        random_state=0,
        normalized_stress="auto",
    ).fit_transform(d_high)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    palette = [f"C{i}" for i in range(len(np.unique(y_clusters)))]

    panels = [
        ("AIRM Reference (MDS)", z_airm_mds, knn_high),
        ("Classical PCA (d=2)", z_classic, knn_class),
        ("Tangent PCA (d=2)", z_tangent, knn_tang),
    ]

    for ax, (title, z_coords, knn) in zip(axes, panels):
        for c in np.unique(y_clusters):
            mask = y_clusters == c
            ax.scatter(
                z_coords[mask, 0],
                z_coords[mask, 1],
                s=30,
                color=palette[c],
                edgecolor="k",
                linewidth=0.3,
                alpha=0.85,
                label=f"C{c}",
            )
        segments = [[z_coords[i], z_coords[j]] for i in range(n_samples) for j in knn[i]]
        ax.add_collection(LineCollection(segments, colors="grey", alpha=0.25, lw=0.5))
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.2)

    axes[0].legend(fontsize=8, loc="best")
    plt.suptitle(f"k = {K_NN} Neighborhood Graphs (AIRM Ground Truth vs PCA vs Tangent PCA)", fontsize=14)
    plt.tight_layout()
    fig2_path = FIGURES / "18_neighborhood_graphs_airm_vs_pca.png"
    plt.savefig(fig2_path, dpi=150)
    plt.close()
    print(f"Saved: {fig2_path}")


def plot_cluster_and_point_overlap(
    d_high: np.ndarray,
    z_classic: np.ndarray,
    z_tangent: np.ndarray,
    y_clusters: np.ndarray,
) -> None:
    knn_high = knn_from_dist(d_high, K_NN)
    knn_class = knn_from_points(z_classic, K_NN)
    knn_tang = knn_from_points(z_tangent, K_NN)

    ov_classic = neighbor_overlap_per_point(knn_high, knn_class)
    ov_tangent = neighbor_overlap_per_point(knn_high, knn_tang)
    clusters = np.unique(y_clusters)
    per_cluster_classic = [ov_classic[y_clusters == c].mean() for c in clusters]
    per_cluster_tangent = [ov_tangent[y_clusters == c].mean() for c in clusters]

    # Figure 3: Per-cluster neighbor overlap
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x_pos = np.arange(len(clusters))
    width = 0.38
    ax.bar(x_pos - width / 2, per_cluster_classic, width, color="#d95f02", alpha=0.85, label="Classical PCA")
    ax.bar(x_pos + width / 2, per_cluster_tangent, width, color="#1b9e77", alpha=0.85, label="Tangent PCA")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"C{c}" for c in clusters])
    ax.set_ylabel(f"Mean Neighbor Overlap (k={K_NN})")
    ax.set_title("Per-Cluster Neighborhood Preservation at d=2")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig3_path = FIGURES / "19_per_cluster_neighbor_overlap.png"
    plt.savefig(fig3_path, dpi=150)
    plt.close()
    print(f"Saved: {fig3_path}")

    # Figure 4: Per-point overlap heatmaps
    order = np.argsort(y_clusters, kind="stable")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, (name, ov) in zip(axes, [("Classical PCA", ov_classic), ("Tangent PCA", ov_tangent)]):
        im = ax.imshow(ov[order][None, :], aspect="auto", cmap="viridis", vmin=0, vmax=1)
        bounds = np.cumsum(np.bincount(y_clusters))
        for b in bounds[:-1]:
            ax.axvline(b - 0.5, color="w", lw=1.2)
        ax.set_yticks([])
        centers = bounds - np.bincount(y_clusters) / 2
        ax.set_xticks(centers)
        ax.set_xticklabels([f"C{c}" for c in clusters])
        ax.set_title(f"{name} (Mean Overlap = {ov.mean():.3f})")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    plt.suptitle(f"Per-Point Neighbor Overlap vs AIRM Ground Truth (k={K_NN})", fontsize=13)
    plt.tight_layout()
    fig4_path = FIGURES / "20_per_point_overlap_heatmaps.png"
    plt.savefig(fig4_path, dpi=150)
    plt.close()
    print(f"Saved: {fig4_path}")


if __name__ == "__main__":
    main()
