"""Euclidean and tangent-space PCA baselines."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def euclidean_pca(
    spd_matrices: np.ndarray,
    n_components: int = 4,
) -> tuple[np.ndarray, PCA]:
    """Apply standard PCA to flattened SPD matrices."""
    features = spd_matrices.reshape(len(spd_matrices), -1)
    scaled = StandardScaler().fit_transform(features)
    model = PCA(n_components=n_components, random_state=42)
    return model.fit_transform(scaled), model


def tangent_pca(
    tangent_features: np.ndarray,
    n_components: int = 4,
) -> tuple[np.ndarray, PCA]:
    """Apply PCA to matrix-log features in the tangent space at identity."""
    scaled = StandardScaler().fit_transform(tangent_features)
    model = PCA(n_components=n_components, random_state=42)
    return model.fit_transform(scaled), model
