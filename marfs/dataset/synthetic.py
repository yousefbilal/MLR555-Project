"""Synthetic dataset generation with controlled correlation structure."""

import numpy as np


def make_synthetic_dataset(
    n_samples: int = 5000,
    n_features: int = 200,
    n_informative: int = 20,
    n_redundant: int = 80,
    n_classes: int = 4,
    noise_level: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate a synthetic classification dataset with controlled correlation.

    Creates three types of features:
    - Informative: directly useful for classification
    - Redundant: linear combinations of informative features (correlated)
    - Noise: random, uncorrelated with the target

    Args:
        n_samples: Number of samples.
        n_features: Total number of features.
        n_informative: Number of truly informative features.
        n_redundant: Number of redundant features (correlated with informative).
        n_classes: Number of target classes.
        noise_level: Std of Gaussian noise added to all features.
        seed: Random seed.

    Returns:
        X: Feature matrix (n_samples, n_features).
        y: Target labels (n_samples,).
        feature_names: List of feature name strings.
    """
    rng = np.random.RandomState(seed)
    n_noise = n_features - n_informative - n_redundant
    assert n_noise >= 0, "n_informative + n_redundant must be <= n_features"

    # Step 1: Generate informative features from class-conditional Gaussians
    centers = rng.randn(n_classes, n_informative) * 3.0
    y = rng.randint(0, n_classes, size=n_samples)
    X_informative = centers[y] + rng.randn(n_samples, n_informative) * 1.0

    # Step 2: Generate redundant features as linear combinations of informative ones
    # Each redundant feature is a random linear combination of 2-4 informative features
    X_redundant = np.zeros((n_samples, n_redundant), dtype=np.float32)
    for i in range(n_redundant):
        n_mix = rng.randint(2, min(5, n_informative + 1))
        indices = rng.choice(n_informative, n_mix, replace=False)
        weights = rng.randn(n_mix)
        X_redundant[:, i] = X_informative[:, indices] @ weights
        X_redundant[:, i] += rng.randn(n_samples) * noise_level

    # Step 3: Generate noise features
    X_noise = rng.randn(n_samples, n_noise).astype(np.float32)

    # Combine all features
    X = np.hstack([X_informative, X_redundant, X_noise]).astype(np.float32)

    # Add global noise
    X += rng.randn(*X.shape).astype(np.float32) * noise_level

    # Shuffle feature order so informative/redundant/noise aren't contiguous
    perm = rng.permutation(n_features)
    X = X[:, perm]

    # Create feature names tracking original type
    names_orig = (
        [f"info_{i}" for i in range(n_informative)]
        + [f"redund_{i}" for i in range(n_redundant)]
        + [f"noise_{i}" for i in range(n_noise)]
    )
    feature_names = [names_orig[p] for p in perm]

    return X, y.astype(np.int64), feature_names
