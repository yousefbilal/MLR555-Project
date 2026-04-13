"""Feature assignment strategies for distributing features to agents.

Includes random, K-Means, spectral, and GCN-informed spectral clustering.
"""

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering


def assign_features(X: np.ndarray, n_agents: int, strategy: str = "random",
                    seed: int = 42, gcn_embeddings: np.ndarray = None) -> list[list[int]]:
    """Assign N features to M agents.

    Args:
        X: Training data array of shape (n_samples, n_features).
        n_agents: Number of agents (M).
        strategy: One of 'random', 'kmeans', 'spectral', 'gcn_spectral'.
        seed: Random seed.
        gcn_embeddings: (n_features, embed_dim) GCN node embeddings.
            Required for 'gcn_spectral' strategy.

    Returns:
        List of M lists, each containing feature indices assigned to that agent.
    """
    n_features = X.shape[1]
    if n_agents > n_features:
        raise ValueError(f"n_agents ({n_agents}) > n_features ({n_features})")

    if strategy == "random":
        return _random_assignment(X, n_agents, seed)
    elif strategy == "kmeans":
        return _kmeans_assignment(X, n_agents, seed)
    elif strategy == "spectral":
        return _spectral_assignment(X, n_agents, seed)
    elif strategy == "gcn_spectral":
        if gcn_embeddings is None:
            raise ValueError("gcn_spectral requires gcn_embeddings")
        return _gcn_spectral_assignment(gcn_embeddings, n_agents, seed)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _random_assignment(X: np.ndarray, n_agents: int, seed: int) -> list[list[int]]:
    """Random shuffle and deal features into M equal groups."""
    n_features = X.shape[1]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_features)
    return _split_indices(indices, n_agents)


def _kmeans_assignment(X: np.ndarray, n_agents: int, seed: int) -> list[list[int]]:
    """Cluster features using K-Means on the correlation matrix."""
    corr_matrix = np.corrcoef(X.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    kmeans = KMeans(n_clusters=n_agents, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(corr_matrix)

    groups = _labels_to_groups(labels, n_agents)
    return _rebalance_groups(groups, n_agents, X.shape[1])


def _spectral_assignment(X: np.ndarray, n_agents: int, seed: int) -> list[list[int]]:
    """Cluster features using Spectral Clustering on the absolute correlation matrix."""
    corr_matrix = np.corrcoef(X.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    affinity = np.clip(np.abs(corr_matrix), 0, 1)
    np.fill_diagonal(affinity, 1.0)

    sc = SpectralClustering(
        n_clusters=n_agents, affinity="precomputed",
        random_state=seed, assign_labels="kmeans"
    )
    labels = sc.fit_predict(affinity)

    groups = _labels_to_groups(labels, n_agents)
    return _rebalance_groups(groups, n_agents, X.shape[1])


def _gcn_spectral_assignment(gcn_embeddings: np.ndarray, n_agents: int,
                             seed: int) -> list[list[int]]:
    """Cluster features using spectral clustering on GCN node embeddings.

    The GCN embeddings encode 2-hop structural relationships in the feature
    correlation graph, capturing higher-order structure that raw pairwise
    correlations miss.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Build affinity from embedding similarity
    sim = cosine_similarity(gcn_embeddings)
    sim = np.clip(sim, 0, 1)
    np.fill_diagonal(sim, 1.0)

    sc = SpectralClustering(
        n_clusters=n_agents, affinity="precomputed",
        random_state=seed, assign_labels="kmeans"
    )
    labels = sc.fit_predict(sim)

    groups = _labels_to_groups(labels, n_agents)
    return _rebalance_groups(groups, n_agents, gcn_embeddings.shape[0])


def _labels_to_groups(labels: np.ndarray, n_agents: int) -> list[list[int]]:
    """Convert cluster labels to list of groups."""
    groups = [[] for _ in range(n_agents)]
    for feat_idx, label in enumerate(labels):
        groups[label].append(feat_idx)
    return groups


def _split_indices(indices: np.ndarray, n_groups: int) -> list[list[int]]:
    """Split an array of indices into n roughly equal groups."""
    return [s.tolist() for s in np.array_split(indices, n_groups)]


def _rebalance_groups(groups: list[list[int]], n_agents: int,
                      n_features: int) -> list[list[int]]:
    """Rebalance groups so no group is empty and sizes are roughly equal."""
    non_empty = [g for g in groups if len(g) > 0]
    empty_count = n_agents - len(non_empty)

    if empty_count > 0:
        for _ in range(empty_count):
            non_empty.sort(key=len, reverse=True)
            largest = non_empty[0]
            mid = len(largest) // 2
            new_group = largest[mid:]
            non_empty[0] = largest[:mid]
            non_empty.append(new_group)

    max_iters = n_features
    for _ in range(max_iters):
        sizes = [len(g) for g in non_empty]
        max_idx = int(np.argmax(sizes))
        min_idx = int(np.argmin(sizes))
        if sizes[max_idx] - sizes[min_idx] <= 1:
            break
        feat = non_empty[max_idx].pop()
        non_empty[min_idx].append(feat)

    return non_empty
