"""Reward computation for feature selection.

Supports two modes:
- "simple":       R = Acc - λ₁·Redundancy + λ₂·Relevance
- "hierarchical": R = r_global + r_local  (EAC-FS-inspired)
    r_global = w_a·Acc + (1-w_a)·ΔAcc - w_s·|F_t|/d
    r_local_j = -w_d·Redundancy(F_j)
"""

from collections import OrderedDict

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from marfs.utils import safe_corrcoef
import warnings


# ---------------------------------------------------------------------------
# Accuracy cache
# ---------------------------------------------------------------------------
# The reward function is called once per (env step, agent), and many of those
# calls share the same global mask (all agents in a step) or repeat masks the
# policy has already evaluated. We memoize accuracy by mask bytes. The cache
# is keyed on (mask, classifier_type, X_train_id, n_train) so it's implicitly
# scoped to a trainer run — a fresh trainer gets a fresh array (new id) and
# therefore disjoint cache entries.

_ACC_CACHE: "OrderedDict[tuple, float]" = OrderedDict()
_CACHE_MAX = 10_000
_CACHE_HITS = 0
_CACHE_MISSES = 0


def configure_cache(max_size: int) -> None:
    """Set cache capacity. max_size <= 0 disables caching."""
    global _CACHE_MAX
    _CACHE_MAX = int(max_size)
    if _CACHE_MAX <= 0:
        _ACC_CACHE.clear()


def clear_cache() -> None:
    """Drop all cached entries and reset hit/miss counters."""
    global _CACHE_HITS, _CACHE_MISSES
    _ACC_CACHE.clear()
    _CACHE_HITS = 0
    _CACHE_MISSES = 0


def get_cache_stats() -> dict:
    """Return current hit/miss counts and cache size."""
    total = _CACHE_HITS + _CACHE_MISSES
    return {
        "hits": _CACHE_HITS,
        "misses": _CACHE_MISSES,
        "size": len(_ACC_CACHE),
        "hit_rate": (_CACHE_HITS / total) if total else 0.0,
    }


def compute_reward(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mask: np.ndarray,
    config,
    prev_accuracy: float = None,
    agent_groups: list[list[int]] = None,
) -> dict:
    """Compute reward for a given feature selection mask.

    Args:
        X_train, y_train, X_test, y_test: Data splits.
        mask: Binary mask of length n_features.
        config: MARFSConfig instance.
        prev_accuracy: Previous step's accuracy (for delta accuracy).
        agent_groups: List of M groups (for per-agent local reward).

    Returns:
        Dict with keys:
            "global": scalar global reward.
            "local": dict mapping agent_idx -> local reward.
            "accuracy": current accuracy.
            "n_selected": number of selected features.
    """
    selected = np.where(mask > 0)[0]
    n_selected = len(selected)
    n_features = len(mask)

    if n_selected == 0:
        n_agents = len(agent_groups) if agent_groups else 1
        return {
            "global": -1.0,
            "local": {i: 0.0 for i in range(n_agents)},
            "accuracy": 0.0,
            "n_selected": 0,
        }

    X_tr = X_train[:, selected]
    X_te = X_test[:, selected]

    # Current accuracy (cached by mask + classifier + dataset identity)
    acc = _accuracy_cached(mask, X_tr, y_train, X_te, y_test, X_train, config)

    if config.reward_type == "hierarchical":
        return _hierarchical_reward(
            acc, prev_accuracy, n_selected, n_features,
            X_train, mask, agent_groups, config
        )
    else:
        return _simple_reward(
            acc, X_tr, y_train, n_selected, n_features,
            agent_groups, config
        )


def _simple_reward(acc, X_tr, y_train, n_selected, n_features,
                   agent_groups, config) -> dict:
    """R = Acc - λ₁·Redundancy + λ₂·Relevance"""
    redundancy = _compute_redundancy(X_tr)
    relevance = _compute_relevance(X_tr, y_train)

    global_r = acc - config.lambda_redundancy * redundancy + config.lambda_relevance * relevance

    n_agents = len(agent_groups) if agent_groups else 1
    return {
        "global": float(global_r),
        "local": {i: 0.0 for i in range(n_agents)},
        "accuracy": float(acc),
        "n_selected": n_selected,
    }


def _hierarchical_reward(acc, prev_accuracy, n_selected, n_features,
                         X_train, mask, agent_groups, config) -> dict:
    """Hierarchical reward inspired by EAC-FS (Eq. 4).

    r_global = w_a * Acc + (1 - w_a) * ΔAcc - w_s * |F_t| / d
    r_local_j = -w_d * Redundancy(F_j)
    """
    # Delta accuracy
    if prev_accuracy is not None:
        delta_acc = acc - prev_accuracy
    else:
        delta_acc = 0.0

    # Global reward
    r_global = (config.w_acc * acc
                + (1 - config.w_acc) * delta_acc
                - config.w_size * n_selected / n_features)

    # Per-agent local reward (redundancy within each agent's selected features)
    local_rewards = {}
    if agent_groups:
        selected_set = set(np.where(mask > 0)[0])
        for agent_idx, group in enumerate(agent_groups):
            agent_selected = [f for f in group if f in selected_set]
            if len(agent_selected) > 1:
                X_agent = X_train[:, agent_selected]
                local_red = _compute_redundancy(X_agent)
                local_rewards[agent_idx] = -config.w_redundancy * local_red
            else:
                local_rewards[agent_idx] = 0.0
    else:
        local_rewards[0] = 0.0

    return {
        "global": float(r_global),
        "local": local_rewards,
        "accuracy": float(acc),
        "n_selected": n_selected,
    }


def _accuracy_cached(mask: np.ndarray, X_tr, y_train, X_te, y_test,
                     X_train_full, config) -> float:
    """Memoize _compute_accuracy by mask bytes + classifier + dataset identity.

    Key includes id(X_train_full) and len(y_train) so two trainers running in
    the same process don't collide. Bypasses the cache if size <= 0.
    """
    global _CACHE_HITS, _CACHE_MISSES

    if _CACHE_MAX <= 0:
        return _compute_accuracy(X_tr, y_train, X_te, y_test, config)

    classifier_type = getattr(config, "reward_classifier", "ridge").lower()
    key = (
        mask.astype(np.uint8, copy=False).tobytes(),
        classifier_type,
        id(X_train_full),
        int(len(y_train)),
        int(getattr(config, "seed", 0)),
    )

    cached = _ACC_CACHE.get(key)
    if cached is not None:
        _CACHE_HITS += 1
        _ACC_CACHE.move_to_end(key)  # LRU touch
        return cached

    _CACHE_MISSES += 1
    acc = _compute_accuracy(X_tr, y_train, X_te, y_test, config)
    _ACC_CACHE[key] = acc
    if len(_ACC_CACHE) > _CACHE_MAX:
        _ACC_CACHE.popitem(last=False)  # drop oldest
    return acc


def _compute_accuracy(X_train, y_train, X_test, y_test, config) -> float:
    """Train configured classifier and return accuracy."""
    try:
        classifier_type = getattr(config, "reward_classifier", "ridge").lower()
        
        if classifier_type == "rf":
            clf = RandomForestClassifier(n_estimators=50, random_state=config.seed, n_jobs=-1)
        elif classifier_type == "lightgbm":
            from lightgbm import LGBMClassifier
            clf = LGBMClassifier(n_estimators=50, verbose=-1, random_state=config.seed)
        elif classifier_type == "xgboost":
            from xgboost import XGBClassifier
            clf = XGBClassifier(n_estimators=50, verbosity=0, random_state=config.seed, n_jobs=-1)
        else:
            clf = RidgeClassifier(alpha=1.0)
            
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

        # class_counts = np.bincount(y_train)
        # imbalance_ratio = class_counts.max() / max(class_counts.min(), 1)
        # if imbalance_ratio > 5:
        #     return balanced_accuracy_score(y_test, y_pred)
        return accuracy_score(y_test, y_pred)
    except Exception:
        return 0.0


def _compute_redundancy(X: np.ndarray) -> float:
    """Mean absolute pairwise Pearson correlation between features."""
    if X.shape[1] <= 1:
        return 0.0
    corr = safe_corrcoef(X, axis=0)
    np.fill_diagonal(corr, 0.0)
    n = corr.shape[0]
    upper = np.abs(corr[np.triu_indices(n, k=1)])
    return float(upper.mean()) if len(upper) > 0 else 0.0


def _compute_relevance(X: np.ndarray, y: np.ndarray) -> float:
    """Mean absolute Pearson correlation between each feature and the target."""
    y_float = y.astype(np.float64)
    correlations = []
    for j in range(X.shape[1]):
        col = X[:, j].astype(np.float64)
        if np.std(col) < 1e-10 or np.std(y_float) < 1e-10:
            correlations.append(0.0)
            continue
        r = np.corrcoef(col, y_float)[0, 1]
        correlations.append(abs(r) if not np.isnan(r) else 0.0)
    return float(np.mean(correlations))
