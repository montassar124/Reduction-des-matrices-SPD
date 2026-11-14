"""Reusable methods for subject-level SPD brain-connectome reduction."""

from .data import build_spd_subjects, load_connectomes
from .pca_methods import euclidean_pca, tangent_pca
from .evaluation import repeated_knn_evaluation
from .local_geometry import summarize_local_geometry
from .regularization import (
    generate_connectome_data,
    diag_loading,
    eig_flooring,
    evaluate_regularization_methods,
)
from .geometry_metrics import (
    generate_multicluster_spd_dataset,
    vectorize_sym_isometric,
    airm_dist,
    pairwise_airm,
    np_at_k,
    trustworthiness,
    continuity,
    distance_distortion,
    local_reconstruction_error,
)

__all__ = [
    "build_spd_subjects",
    "load_connectomes",
    "euclidean_pca",
    "tangent_pca",
    "repeated_knn_evaluation",
    "summarize_local_geometry",
    "generate_connectome_data",
    "diag_loading",
    "eig_flooring",
    "evaluate_regularization_methods",
    "generate_multicluster_spd_dataset",
    "vectorize_sym_isometric",
    "airm_dist",
    "pairwise_airm",
    "np_at_k",
    "trustworthiness",
    "continuity",
    "distance_distortion",
    "local_reconstruction_error",
]
