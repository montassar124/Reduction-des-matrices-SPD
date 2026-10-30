"""Convert the geomstats connectome vectors into SPD matrices.

The geomstats connectome dataset stores the 378 upper-triangular entries of
28x28 connectomes.  A matrix exponential of a symmetric matrix is SPD, so this
script uses each connectome as the off-diagonal part of a symmetric matrix
with zero diagonal and saves its exponential.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm

from spd_reduction.data import vectors_to_symmetric


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "train"
FNC_PATH = TRAIN_DIR / "train_FNC.csv"
LABELS_PATH = TRAIN_DIR / "train_labels.csv"
OUTPUT_PATH = TRAIN_DIR / "train_spd.npy"


def main() -> None:
    if not FNC_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError(
            "Download train/train_FNC.csv and train/train_labels.csv first."
        )

    fnc = pd.read_csv(FNC_PATH)
    labels = pd.read_csv(LABELS_PATH)
    if "Id" not in fnc or "Class" not in labels:
        raise ValueError("Unexpected geomstats CSV columns.")
    if len(fnc) != len(labels):
        raise ValueError("The feature and label row counts do not match.")

    vectors = fnc.drop(columns=["Id"]).to_numpy(dtype=np.float64)
    symmetric = vectors_to_symmetric(vectors)
    spd = np.stack([expm(matrix) for matrix in symmetric])

    eigenvalues = np.linalg.eigvalsh(spd)
    if not np.all(eigenvalues > 0):
        raise RuntimeError("Matrix exponential did not produce strictly SPD output.")

    np.save(OUTPUT_PATH, spd)
    print(f"Saved {OUTPUT_PATH}")
    print(f"Shape: {spd.shape}")
    print(f"Smallest eigenvalue: {eigenvalues.min():.6g}")
    print(f"Labels: {labels['Class'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
