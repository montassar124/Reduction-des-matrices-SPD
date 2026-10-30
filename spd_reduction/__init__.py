"""Reusable methods for subject-level SPD brain-connectome reduction."""

from .data import build_spd_subjects, load_connectomes
from .pca_methods import euclidean_pca, tangent_pca
from .evaluation import repeated_knn_evaluation

__all__ = [
    "build_spd_subjects",
    "load_connectomes",
    "euclidean_pca",
    "tangent_pca",
    "repeated_knn_evaluation",
]
