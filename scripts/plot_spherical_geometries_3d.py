"""Plot the spherical log-SPD data and its 3-D reduced geometries."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scripts.compare_paper_spd_reductions import (
    LieLLE,
    assign,
    fit_adaptive_charts,
    generate_case,
    logs,
    upper,
)
from spd_reduction.local_geometry import (
    aligned_local_tangent_coordinates,
    matrix_exp_symmetric,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
METHODS = (
    "Sphère log-SPD",
    "PCA classique",
    "PCA tangentielle",
    "Repères adaptatifs",
    "Lie-LLE",
)
COLORS = ("#1976d2", "#e53935")


def reduce_to_three(features: np.ndarray) -> np.ndarray:
    scaled = StandardScaler().fit_transform(features)
    return PCA(n_components=3, random_state=42).fit_transform(scaled)


def make_embeddings(matrices: np.ndarray) -> dict[str, np.ndarray]:
    matrix_logs = logs(matrices)
    flat_logs = matrix_logs.reshape(len(matrices), -1)
    original = flat_logs / np.maximum(
        np.linalg.norm(flat_logs, axis=1, keepdims=True), 1e-12
    )
    embeddings = {
        "Sphère log-SPD": reduce_to_three(original),
        "PCA classique": reduce_to_three(matrices.reshape(len(matrices), -1)),
        "PCA tangentielle": reduce_to_three(upper(matrix_logs)),
    }

    centers, assignments, _ = fit_adaptive_charts(matrices)
    global_center = matrix_exp_symmetric(matrix_logs.mean(axis=0))
    adaptive = aligned_local_tangent_coordinates(
        matrices, centers, assignments, global_center
    )
    embeddings["Repères adaptatifs"] = reduce_to_three(adaptive)

    lie_lle = LieLLE(n_neighbors=6, n_components=3)
    embeddings["Lie-LLE"] = lie_lle.fit_transform(matrices)
    return embeddings


def add_mesh(axis: plt.Axes, points: np.ndarray, color: str) -> None:
    sample_size = min(350, len(points))
    sample = points[np.linspace(0, len(points) - 1, sample_size).astype(int)]
    if len(sample) < 4:
        return
    try:
        hull = ConvexHull(sample, qhull_options="QJ")
    except Exception:
        return
    triangles = [sample[simplex] for simplex in hull.simplices]
    mesh = Poly3DCollection(
        triangles, alpha=0.08, facecolor=color, edgecolor=color, linewidth=0.2
    )
    axis.add_collection3d(mesh)


def add_reference_sphere(axis: plt.Axes) -> None:
    """Draw the unit sphere used to normalize the log-SPD descriptors."""
    phi = np.linspace(0.0, np.pi, 30)
    theta = np.linspace(0.0, 2.0 * np.pi, 60)
    x = np.sin(phi)[:, None] * np.cos(theta)[None, :]
    y = np.sin(phi)[:, None] * np.sin(theta)[None, :]
    z = np.cos(phi)[:, None] * np.ones_like(theta)[None, :]
    axis.plot_surface(
        x,
        y,
        z,
        color="#90caf9",
        alpha=0.16,
        linewidth=0.25,
        edgecolor="#1976d2",
        rstride=2,
        cstride=3,
    )
    axis.plot_wireframe(
        x,
        y,
        z,
        color="#1976d2",
        alpha=0.22,
        linewidth=0.35,
        rstride=4,
        cstride=6,
    )


def plot_case(name: str, bounds: tuple[tuple[float, float], tuple[float, float]]) -> Path:
    matrices, labels = generate_case(*bounds)
    embeddings = make_embeddings(matrices)
    figure = plt.figure(figsize=(19, 12))
    for position, method in enumerate(METHODS, start=1):
        axis = figure.add_subplot(2, 3, position, projection="3d")
        points = embeddings[method]
        scale = np.max(np.linalg.norm(points, axis=1))
        points = points / max(scale, 1e-12)
        for label, color in enumerate(COLORS):
            selection = labels == label
            axis.scatter(
                points[selection, 0],
                points[selection, 1],
                points[selection, 2],
                s=7,
                alpha=0.35,
                color=color,
                label=f"Classe {label + 1}",
            )
        if method == "Sphère log-SPD":
            add_reference_sphere(axis)
        else:
            add_mesh(axis, points, "#455a64")
        title = (
            "Sphère de départ (rayon 1)"
            if method == "Sphère log-SPD"
            else f"{method} : géométrie réduite"
        )
        axis.set_title(title)
        axis.set_xlim(-1.05, 1.05)
        axis.set_ylim(-1.05, 1.05)
        axis.set_zlim(-1.05, 1.05)
        axis.set_xlabel("Axe 1")
        axis.set_ylabel("Axe 2")
        axis.set_zlabel("Axe 3")
        axis.grid(alpha=0.2)
        if position == 1:
            axis.legend(loc="upper left", fontsize=8)
    figure.suptitle(
        f"{name} — sphère log-SPD et géométries 3D avec maillage",
        fontsize=16,
    )
    figure.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    slug = "modere" if "modéré" in name else "fort"
    output = FIGURES / f"14_spherical_geometries_{slug}.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    cases = {
        "Recouvrement modéré": ((0.0, 1.0), (0.25, 1.25)),
        "Recouvrement fort": ((0.0, 1.0), (0.125, 1.125)),
    }
    for name, bounds in cases.items():
        print(f"Saved: {plot_case(name, bounds)}")


if __name__ == "__main__":
    main()
