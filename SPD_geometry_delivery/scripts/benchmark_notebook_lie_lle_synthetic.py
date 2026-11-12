"""Compare the friend's notebook Lie-LLE workflow on synthetic SPD cases."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.compare_paper_spd_reductions import (
    evaluate_method,
    generate_case,
    neighbor_preservation,
)
from scripts.run_eth80_notebook_benchmark import (
    lie_lle,
    one_nn_log_spd,
)
from spd_reduction.local_geometry import matrix_exp_symmetric, matrix_log_spd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "synthetic_notebook_lie_lle_benchmark.csv"
RNG = np.random.default_rng(42)


def generate_spherical_case(
    first: tuple[float, float],
    second: tuple[float, float],
    samples_per_class: int = 1000,
    dimension: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    center_rng = np.random.default_rng(2026)
    centers = []
    for _ in range(2):
        direction = center_rng.normal(size=(dimension, dimension))
        direction = (direction + direction.T) / 2.0
        centers.append(direction / np.linalg.norm(direction, ord="fro"))
    noise_scale = 0.35 if first[1] - first[0] > 0.5 else 0.22
    matrices = []
    labels = []
    for label in range(2):
        for _ in range(samples_per_class):
            noise = RNG.normal(
                0.0, noise_scale, size=(dimension, dimension)
            )
            noise = (noise + noise.T) / 2.0
            symmetric = centers[label] + noise
            symmetric /= max(np.linalg.norm(symmetric, ord="fro"), 1e-12)
            matrices.append(matrix_exp_symmetric(symmetric))
            labels.append(label)
    return np.asarray(matrices), np.asarray(labels)


def evaluate_case(
    case: str, matrices: np.ndarray, labels: np.ndarray
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for method in (
        "PCA classique",
        "PCA tangent",
        "Reperes adaptatifs",
    ):
        mean, std, references, preservation = evaluate_method(
            matrices, labels, method, neighbors=6
        )
        rows.append(
            {
                "Case": case,
                "Method": method,
                "Protocol": "previous 50/50, 5-NN, train-only",
                "Mean accuracy": mean,
                "Std": std,
                "References mean": np.mean(references) if references else 1.0,
                "Neighbor preservation": preservation,
            }
        )

    lie_scores = []
    lie_preservation = []
    for seed in range(10):
        train, test = train_test_split(
            np.arange(len(labels)),
            test_size=0.5,
            random_state=seed,
            stratify=labels,
        )
        all_indices = np.concatenate([train, test])
        all_logs = np.asarray(
            [matrix_log_spd(matrix) for matrix in matrices[all_indices]]
        )
        lie = lie_lle(all_logs, k=6, dimension=30)
        prediction = one_nn_log_spd(
            lie[: len(train)], labels[train], lie[len(train) :]
        )
        lie_scores.append(float(np.mean(prediction == labels[test])))
        lie_preservation.append(
            neighbor_preservation(
                all_logs.reshape(len(all_logs), -1),
                lie.reshape(len(lie), -1),
                neighbors=6,
            )
        )
    rows.append(
        {
            "Case": case,
            "Method": "Lie-LLE notebook",
            "Protocol": "previous 50/50, notebook Lie-LLE, 1-NN",
            "Mean accuracy": np.mean(lie_scores),
            "Std": np.std(lie_scores),
            "References mean": 1.0,
            "Neighbor preservation": np.mean(lie_preservation),
        }
    )
    return rows


def main() -> None:
    cases = {
        "Sphere moderee": ((0.0, 1.0), (0.25, 1.25)),
        "Sphere forte": ((0.0, 1.0), (0.125, 1.125)),
    }
    all_rows = []
    for name, bounds in cases.items():
        matrices, labels = generate_spherical_case(*bounds)
        all_rows.extend(evaluate_case(name, matrices, labels))
        print(f"Finished: {name}")
    results = pd.DataFrame(all_rows)
    summary = results.set_index(["Case", "Method"])[
        ["Mean accuracy", "Std", "References mean", "Neighbor preservation"]
    ]
    summary.to_csv(RESULTS)
    display = summary.copy()
    display["Mean accuracy"] *= 100
    display["Std"] *= 100
    display["Neighbor preservation"] *= 100
    display = display.rename(
        columns={
            "Mean accuracy": "Accuracy (%)",
            "Std": "Accuracy std (%)",
            "Neighbor preservation": "Neighbours preserved (%)",
        }
    )
    print("\nSYNTHETIC SPD RESULTS")
    print(display.round(2).to_string())
    print(f"Saved: {RESULTS}")


if __name__ == "__main__":
    main()
