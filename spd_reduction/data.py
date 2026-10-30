"""Loading and constructing subject-level SPD connectomes."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm


def vectors_to_symmetric(vectors: np.ndarray, dimension: int = 28) -> np.ndarray:
    """Convert upper-triangular connectome vectors to symmetric matrices."""
    expected = dimension * (dimension - 1) // 2
    if vectors.ndim != 2 or vectors.shape[1] != expected:
        raise ValueError(f"Expected {expected} features, got {vectors.shape}.")
    matrices = np.zeros((len(vectors), dimension, dimension), dtype=np.float64)
    rows, cols = np.triu_indices(dimension, k=1)
    matrices[:, rows, cols] = vectors
    matrices[:, cols, rows] = vectors
    return matrices


def symmetric_upper_vectors(matrices: np.ndarray) -> np.ndarray:
    """Extract the strict upper triangle from a batch of symmetric matrices."""
    rows, cols = np.triu_indices(matrices.shape[-1], k=1)
    return matrices[:, rows, cols]


def load_connectomes(
    data_dir: str | Path = "train",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Geomstats connectome vectors, IDs, and class labels."""
    data_dir = Path(data_dir)
    fnc = pd.read_csv(data_dir / "train_FNC.csv")
    labels = pd.read_csv(data_dir / "train_labels.csv")
    if "Id" not in fnc or "Class" not in labels:
        raise ValueError("Unexpected connectome CSV columns.")
    if len(fnc) != len(labels) or not np.array_equal(
        fnc["Id"].to_numpy(), labels["Id"].to_numpy()
    ):
        raise ValueError("Feature rows and labels are not aligned.")
    return (
        fnc.drop(columns=["Id"]).to_numpy(dtype=np.float64),
        fnc["Id"].to_numpy(),
        labels["Class"].to_numpy(dtype=np.int64),
    )


def build_spd_subjects(
    vectors: np.ndarray,
    dimension: int = 28,
) -> tuple[np.ndarray, np.ndarray]:
    """Build SPD matrices and their tangent features from connectome vectors."""
    symmetric = vectors_to_symmetric(vectors, dimension)
    spd = np.stack([expm(matrix) for matrix in symmetric])
    eigenvalues = np.linalg.eigvalsh(spd)
    if np.any(eigenvalues <= 0):
        raise RuntimeError("Matrix exponential produced a non-SPD matrix.")

    from scipy.linalg import logm

    log_spd = np.stack([logm(matrix).real for matrix in spd])
    return spd, symmetric_upper_vectors(log_spd).astype(np.float32)
