"""Evaluation utilities shared by all reduction methods."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def repeated_knn_evaluation(
    embeddings: dict[str, np.ndarray],
    labels: np.ndarray,
    repeats: int = 10,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """Evaluate embeddings with repeated stratified 80/20 5-NN splits."""
    rows = []
    for name, embedding in embeddings.items():
        scores = []
        for seed in range(repeats):
            train_idx, test_idx = train_test_split(
                np.arange(len(labels)),
                test_size=test_size,
                random_state=seed,
                stratify=labels,
            )
            classifier = KNeighborsClassifier(n_neighbors=5)
            classifier.fit(embedding[train_idx], labels[train_idx])
            scores.append(accuracy_score(labels[test_idx], classifier.predict(embedding[test_idx])))
        rows.append((name, float(np.mean(scores)), float(np.std(scores))))
    return pd.DataFrame(rows, columns=["Reducer", "Mean_Accuracy", "Std_Accuracy"])
