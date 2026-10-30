"""Presentation-ready figures for the SPD reduction experiment."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _class_names(labels: np.ndarray) -> list[str]:
    return [f"Class {label}" for label in sorted(np.unique(labels))]


def plot_embeddings(
    embeddings: dict[str, np.ndarray],
    labels: np.ndarray,
    output_path: str | Path,
) -> None:
    """Plot all 2-D embeddings with consistent class colours."""
    names = list(embeddings)
    colors = {label: color for label, color in zip(sorted(np.unique(labels)), ["#19b5fe", "#ff6b6b"])}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for axis, name in zip(axes.flat, names):
        embedding = embeddings[name]
        for label in sorted(np.unique(labels)):
            mask = labels == label
            axis.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=42,
                alpha=0.85,
                color=colors[label],
                label=f"Class {label}",
                edgecolors="white",
                linewidths=0.4,
            )
        axis.set_title(name.replace("_", " "))
        axis.set_xlabel("Component 1")
        axis.set_ylabel("Component 2")
        axis.grid(alpha=0.2)
    handles, _ = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, _class_names(labels), loc="upper center", ncol=len(handles))
    fig.suptitle("Two-dimensional embeddings of COBRE subjects", fontsize=16)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(results: pd.DataFrame, output_path: str | Path) -> None:
    """Plot mean accuracy with standard-deviation error bars."""
    fig, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    axis.bar(
        results["Reducer"].str.replace("_", " "),
        results["Mean_Accuracy"],
        yerr=results["Std_Accuracy"],
        capsize=5,
        color=["#19b5fe", "#8e7dff", "#00c49a", "#ff9f43"],
        edgecolor="#16324f",
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean kNN accuracy")
    axis.set_title("Repeated 80/20 classification performance")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=18)
    for index, value in enumerate(results["Mean_Accuracy"]):
        axis.text(index, value + 0.035, f"{value:.2f}", ha="center", fontweight="bold")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_spd_spectrum(spd_subjects: np.ndarray, output_path: str | Path) -> None:
    """Show the eigenvalue distributions proving positive definiteness."""
    eigenvalues = np.linalg.eigvalsh(spd_subjects)
    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    axis.hist(eigenvalues.ravel(), bins=30, color="#00c49a", edgecolor="white")
    axis.axvline(0, color="#d62828", linestyle="--", label="SPD boundary (0)")
    axis.set_xlabel("Eigenvalue")
    axis.set_ylabel("Count")
    axis.set_title(
        f"SPD eigenvalue spectrum (minimum = {eigenvalues.min():.3g})"
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pipeline(output_path: str | Path) -> None:
    """Draw the methodological pipeline used in the presentation."""
    fig, axis = plt.subplots(figsize=(14, 3.6), constrained_layout=True)
    axis.axis("off")
    stages = [
        ("COBRE", "86 subjects\n378 FNC values"),
        ("SPD map", "28×28 symmetric\nmatrix exponential"),
        ("Representations", "Flattened SPD\nor matrix logarithm"),
        ("Reduction", "PCA · Tangent PCA\nAutoencoder · MIAF"),
        ("Evaluation", "2-D embedding\nrepeated 80/20 kNN"),
    ]
    x_positions = np.linspace(0.1, 0.9, len(stages))
    for index, (title, detail) in enumerate(stages):
        axis.text(
            x_positions[index],
            0.52,
            f"{title}\n\n{detail}",
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#16324f", edgecolor="#19b5fe"),
        )
        if index < len(stages) - 1:
            axis.annotate(
                "",
                xy=(x_positions[index + 1] - 0.08, 0.52),
                xytext=(x_positions[index] + 0.08, 0.52),
                arrowprops=dict(arrowstyle="->", color="#ff9f43", lw=2),
            )
    axis.set_title("Dimension reduction in the SPD space", fontsize=16, pad=12)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_full_summary(
    results: pd.DataFrame,
    labels: np.ndarray,
    spd_subjects: np.ndarray,
    output_path: str | Path,
) -> None:
    """Create one presentation-ready image with data, protocol, and results."""
    eigenvalues = np.linalg.eigvalsh(spd_subjects)
    class_counts = pd.Series(labels).value_counts().sort_index()
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.35])

    text_axis = fig.add_subplot(grid[0, :])
    text_axis.axis("off")
    text_axis.set_title(
        "SPD brain-connectome reduction: complete experiment summary",
        fontsize=19,
        fontweight="bold",
        pad=14,
    )
    class_text = "   ".join(
        f"Class {label}: {count} subjects" for label, count in class_counts.items()
    )
    summary = (
        "DATA USED\n"
        f"COBRE/Geomstats training set: {len(labels)} subjects | "
        "378 FNC values per subject | reconstructed matrices: "
        f"{spd_subjects.shape[1]}×{spd_subjects.shape[2]} | {class_text}\n\n"
        "SPD VERIFICATION\n"
        f"Matrix exponential applied to symmetric matrices | "
        f"{spd_subjects.shape[0]} SPD matrices | minimum eigenvalue: "
        f"{eigenvalues.min():.4f} (> 0)\n\n"
        "TEST PROTOCOL\n"
        "Four 2-D representations | 10 stratified 80/20 splits | "
        "5-nearest-neighbour classifier | metric: mean accuracy ± standard deviation"
    )
    text_axis.text(
        0.02,
        0.5,
        summary,
        va="center",
        fontsize=14,
        linespacing=1.65,
        bbox=dict(
            boxstyle="round,pad=0.9",
            facecolor="#eaf4fb",
            edgecolor="#16324f",
            linewidth=1.5,
        ),
    )

    bar_axis = fig.add_subplot(grid[1, 0])
    bar_axis.bar(
        results["Reducer"].str.replace("_", " "),
        results["Mean_Accuracy"],
        yerr=results["Std_Accuracy"],
        capsize=5,
        color=["#19b5fe", "#8e7dff", "#00c49a", "#ff9f43"],
        edgecolor="#16324f",
    )
    bar_axis.set_ylim(0, 1)
    bar_axis.set_ylabel("Mean accuracy")
    bar_axis.set_title("Results over repeated tests")
    bar_axis.tick_params(axis="x", rotation=22)
    bar_axis.grid(axis="y", alpha=0.25)
    for index, value in enumerate(results["Mean_Accuracy"]):
        bar_axis.text(index, value + 0.035, f"{value:.2f}", ha="center", fontweight="bold")

    spectrum_axis = fig.add_subplot(grid[1, 1])
    spectrum_axis.hist(eigenvalues.ravel(), bins=30, color="#00c49a", edgecolor="white")
    spectrum_axis.axvline(0, color="#d62828", linestyle="--", label="SPD boundary (0)")
    spectrum_axis.set_xlabel("Eigenvalue")
    spectrum_axis.set_ylabel("Count")
    spectrum_axis.set_title("All generated matrices are SPD")
    spectrum_axis.legend()
    spectrum_axis.grid(axis="y", alpha=0.2)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
