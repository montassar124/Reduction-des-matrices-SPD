"""Benchmark numerical regularization strategies for Tangent PCA near the SPD cone boundary.

Evaluates:
- Baseline (no regularization)
- M1: Diagonal Loading (C_i <- C_i + eps * I)
- M2: Eigenvalue Flooring (clamp eigenvalues >= tau)
- M3: Mean Regularization (Fréchet mean M* <- M* + eps * I)

Across three scenarios:
- Scenario A: Border of Cone (noise=1e-9)
- Scenario B: Moderate Distance (noise=1e-3)
- Scenario C: Deep Inside Cone (noise=1e+3)
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spd_reduction.regularization import (
    generate_connectome_data,
    evaluate_regularization_methods,
)

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


def plot_boundary_eigenvalues(scenarios: list[tuple[str, float, float]]) -> None:
    """Plot distribution of eigenvalues and condition numbers across scenarios."""
    fig, axes = plt.subplots(len(scenarios), 3, figsize=(16, 4 * len(scenarios)))

    for idx, (name, noise, shift) in enumerate(scenarios):
        x_mats, y_labels = generate_connectome_data(
            n_samples_per_class=500, dim=28, noise_level=noise, class_shift=shift, seed=42
        )
        eigenvalues = np.linalg.eigvalsh(x_mats)
        max_eigs_c1 = eigenvalues[y_labels == 0][:, -1]
        max_eigs_c2 = eigenvalues[y_labels == 1][:, -1]
        min_eigs_c1 = eigenvalues[y_labels == 0][:, 0]
        min_eigs_c2 = eigenvalues[y_labels == 1][:, 0]
        cond_c1 = max_eigs_c1 / np.clip(min_eigs_c1, 1e-15, None)
        cond_c2 = max_eigs_c2 / np.clip(min_eigs_c2, 1e-15, None)

        ax_all, ax_min, ax_cond = axes[idx]

        # Plot 1: All eigenvalues
        ax_all.hist(eigenvalues[y_labels == 0].flatten(), bins=100, alpha=0.5, label="Class 1", color="blue", density=True)
        ax_all.hist(eigenvalues[y_labels == 1].flatten(), bins=100, alpha=0.5, label="Class 2", color="red", density=True)
        ax_all.set_title(f"{name}\nAll Eigenvalues (Log Scale)")
        ax_all.set_xscale("log")
        ax_all.set_xlabel("Eigenvalue")
        ax_all.legend()

        # Plot 2: Min eigenvalues (cone border proximity)
        ax_min.hist(min_eigs_c1, bins=50, alpha=0.5, label="Class 1", color="blue")
        ax_min.hist(min_eigs_c2, bins=50, alpha=0.5, label="Class 2", color="red")
        ax_min.set_title(f"{name}\nMin Eigenvalues (Border Proximity)")
        ax_min.set_xscale("log")
        ax_min.set_xlabel("Min Eigenvalue")
        ax_min.legend()

        # Plot 3: Condition numbers
        ax_cond.hist(cond_c1, bins=50, alpha=0.5, label="Class 1", color="blue")
        ax_cond.hist(cond_c2, bins=50, alpha=0.5, label="Class 2", color="red")
        ax_cond.set_title(f"{name}\nCondition Numbers")
        ax_cond.set_xscale("log")
        ax_cond.set_xlabel("Condition Number")
        ax_cond.legend()

    plt.tight_layout()
    output_path = FIGURES / "15_spd_boundary_eigenvalues.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    scenarios = [
        ("A: Border of Cone", 1e-9, 0.025),
        ("B: Moderate Distance", 1e-3, 0.025),
        ("C: Deep Inside Cone", 1e+3, 0.025),
    ]

    print("=" * 80)
    print("STEP 1: Plotting eigenvalue & condition number distributions")
    print("=" * 80)
    plot_boundary_eigenvalues(scenarios)

    print("\n" + "=" * 80)
    print("STEP 2: Evaluating regularization methods over 10 repeated splits")
    print("=" * 80)

    methods = ["Baseline", "M1 DiagLoad", "M2 EigFloor", "M3 MeanReg"]
    all_results: dict[str, dict[str, list[float]]] = {}

    for name, noise, shift in scenarios:
        print(f"Running scenario: {name} (noise={noise:.1e}) ...")
        all_results[name] = evaluate_regularization_methods(
            noise_level=noise,
            class_shift=shift,
            eps=1e-3,
            tau=1e-3,
            n_runs=10,
            seed=42,
        )

    # Print summary table
    print("\n" + "=" * 90)
    print("CLASSIFICATION ACCURACY (%) -- Mean ± Std over 10 runs")
    print("=" * 90)
    header = f"{'Method':<14} | " + " | ".join(f"{name:<22}" for name, _, _ in scenarios)
    print(header)
    print("-" * len(header))

    table_rows = []
    for method in methods:
        row_str = f"{method:<14} | "
        row_dict: dict[str, object] = {"Method": method}
        for name, _, _ in scenarios:
            accs = all_results[name][method]
            mean_val = float(np.mean(accs))
            std_val = float(np.std(accs))
            row_str += f"{mean_val:>6.2f} ± {std_val:<6.2f}     | "
            row_dict[f"{name}_Mean"] = round(mean_val, 2)
            row_dict[f"{name}_Std"] = round(std_val, 2)
        print(row_str)
        table_rows.append(row_dict)

    df_results = pd.DataFrame(table_rows)
    csv_out = OUTPUT_DIR / "numerical_regularization_benchmark.csv"
    df_results.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")
    try:
        if DELIVERY_RESULTS.exists():
            csv_delivery = DELIVERY_RESULTS / "numerical_regularization_benchmark.csv"
            df_results.to_csv(csv_delivery, index=False)
            print(f"Saved: {csv_delivery}")
    except (PermissionError, OSError):
        pass

    # Plot comparative bar chart
    x_positions = np.arange(len(scenarios))
    bar_width = 0.2
    palette = ["#7570b3", "#d95f02", "#1b9e77", "#e7298a"]

    plt.figure(figsize=(11, 5))
    for i, method in enumerate(methods):
        means = [float(np.mean(all_results[s][method])) for s, _, _ in scenarios]
        stds = [float(np.std(all_results[s][method])) for s, _, _ in scenarios]
        plt.bar(
            x_positions + (i - 1.5) * bar_width,
            means,
            bar_width,
            yerr=stds,
            capsize=4,
            label=method,
            color=palette[i],
            alpha=0.85,
        )

    plt.xticks(x_positions, [s for s, _, _ in scenarios], fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.ylim(0, 100)
    plt.title("Tangent PCA Stability Near Cone Boundary: Baseline vs Regularization Schemes", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    bar_output = FIGURES / "16_tangent_pca_regularization_stability.png"
    plt.savefig(bar_output, dpi=150)
    plt.close()
    print(f"Saved: {bar_output}")


if __name__ == "__main__":
    main()
