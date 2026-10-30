"""Autoencoder and MIAF models used for SPD tangent features."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from torch.utils.data import DataLoader, TensorDataset


class FlexAutoencoder(nn.Module):
    """Configurable autoencoder with a low-dimensional encoder output."""

    def __init__(self, input_dim, latent_dim=2, hidden_dims=(128, 64, 16)):
        super().__init__()
        layers = []
        previous = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(previous, hidden_dim), nn.ReLU()])
            previous = hidden_dim
        layers.append(nn.Linear(previous, latent_dim))
        self.encoder = nn.Sequential(*layers)
        decoder = []
        previous = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder.extend([nn.Linear(previous, hidden_dim), nn.ReLU()])
            previous = hidden_dim
        decoder.append(nn.Linear(previous, input_dim))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, features):
        latent = self.encoder(features)
        return self.decoder(latent), latent


def train_autoencoder(
    features: np.ndarray,
    epochs: int = 100,
    latent_dim: int = 2,
    batch_size: int = 16,
    hidden_dims: tuple[int, ...] = (128, 64, 16),
) -> tuple[FlexAutoencoder, np.ndarray]:
    """Train an autoencoder and return its model and loss history."""
    model = FlexAutoencoder(
        features.shape[1],
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    losses = []
    model.train()
    for _ in range(epochs):
        total = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstruction, _ = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        losses.append(total / len(features))
    return model, np.asarray(losses)


def group_features_sym_kl(
    features: np.ndarray,
    n_bins: int = 20,
    distance_threshold: float = 0.5,
) -> list[np.ndarray]:
    """Group features using symmetric KL divergence between histograms."""
    histograms = []
    for column in features.T:
        counts, _ = np.histogram(column, bins=n_bins)
        counts = counts.astype(float) + 1e-12
        histograms.append(counts / counts.sum())
    histograms = np.asarray(histograms)
    n_features = len(histograms)
    distances = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i + 1, n_features):
            p, q = histograms[i], histograms[j]
            distance = 0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p)))
            distances[i, j] = distances[j, i] = distance
    linkage_matrix = linkage(squareform(distances, checks=False), method="average")
    labels = fcluster(linkage_matrix, distance_threshold, criterion="distance")
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    return [np.asarray(group, dtype=np.int64) for group in groups.values()]


class MIAF(nn.Module):
    """Multiple-input autoencoder with a learned low-dimensional selector."""

    def __init__(self, group_dims, feature_dim=2, sub_encoder_dim=16):
        super().__init__()
        self.sub_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(group_dim, sub_encoder_dim),
                    nn.ReLU(),
                    nn.Linear(sub_encoder_dim, sub_encoder_dim),
                    nn.ReLU(),
                )
                for group_dim in group_dims
            ]
        )
        encoded_dim = sub_encoder_dim * len(group_dims)
        self.feature_selector = nn.Linear(encoded_dim, feature_dim)
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, sum(group_dims)),
        )

    def forward(self, grouped_features):
        encoded = torch.cat(
            [encoder(features) for encoder, features in zip(self.sub_encoders, grouped_features)],
            dim=1,
        )
        selected = torch.relu(self.feature_selector(encoded))
        return self.decoder(selected), selected


def train_miaf(
    features: np.ndarray,
    groups: list[np.ndarray],
    epochs: int = 100,
    batch_size: int = 16,
    feature_dim: int = 2,
) -> tuple[MIAF, np.ndarray]:
    """Train MIAF and return a model whose selected representation has feature_dim."""
    group_dims = [len(group) for group in groups]
    model = MIAF(group_dims, feature_dim=feature_dim)
    grouped = [torch.from_numpy(features[:, group].astype(np.float32)) for group in groups]
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        permutation = torch.randperm(len(features))
        total = 0.0
        for start in range(0, len(features), batch_size):
            indices = permutation[start : start + batch_size]
            batch = [values[indices] for values in grouped]
            target = torch.cat(batch, dim=1)
            optimizer.zero_grad()
            reconstruction, selected = model(batch)
            loss = criterion(reconstruction, target)
            loss = loss + torch.norm(model.feature_selector.weight, p=2, dim=1).sum()
            loss.backward()
            optimizer.step()
            total += criterion(reconstruction, target).item() * len(indices)
        losses.append(total / len(features))
    return model, np.asarray(losses)
