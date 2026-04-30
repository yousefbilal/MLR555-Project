"""Baseline feature-selection methods evaluated under the MARFS protocol.

For each (dataset, method, k) it produces a binary mask and evaluates it with
the same RandomForest(100) downstream classifier used by MARFS, so numbers are
directly comparable to ablation/results tables.

Methods (sklearn always available; skfeature optional):
    - all              keep every feature (sanity ceiling)
    - random           random k features
    - variance         VarianceThreshold (top-k by variance)
    - mutual_info      SelectKBest(mutual_info_classif)
    - f_classif        SelectKBest(f_classif)              (ANOVA F)
    - chi2             SelectKBest(chi2)                   (requires non-negative; min-max scaled)
    - l1_logreg        SelectFromModel(LogisticRegression(L1))
    - rf_importance    SelectFromModel(RandomForestClassifier)
    - rfe_logreg       RFE(LogisticRegression)
    - lap_score        Laplacian Score                     (skfeature, unsupervised)
    - fisher           Fisher Score                        (skfeature, supervised)
    - mcfs             MCFS                                (skfeature, unsupervised)
    - mrmr             minimum Redundancy Maximum Relevance (mrmr-selection, supervised)

Usage:
    python scripts/baselines.py --datasets wdbc,covertype --k 10,20,30 --n-seeds 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.feature_selection import (
    SelectKBest, VarianceThreshold, mutual_info_classif, f_classif, chi2,
    SelectFromModel, RFE,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from marfs.dataset import load_dataset
from marfs.metrics import evaluate_final_selection

try:
    from skfeature.function.similarity_based import lap_score, fisher_score
    from skfeature.function.sparse_learning_based import MCFS
    from skfeature.utility.construct_W import construct_W
    SKFEATURE_OK = True
except Exception:
    SKFEATURE_OK = False

try:
    import pandas as pd
    from mrmr import mrmr_classif
    MRMR_OK = True
except Exception:
    MRMR_OK = False


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def _mask_from_indices(indices, n_features: int) -> np.ndarray:
    mask = np.zeros(n_features, dtype=np.float32)
    mask[np.asarray(indices, dtype=int)] = 1.0
    return mask


def _topk(scores: np.ndarray, k: int, descending: bool = True) -> np.ndarray:
    order = np.argsort(scores)
    if descending:
        order = order[::-1]
    return order[:k]


# ---------------------------------------------------------------------------
# Baseline implementations: each returns a binary mask of shape (n_features,)
# ---------------------------------------------------------------------------

def _all(X_train, y_train, k, seed):
    return np.ones(X_train.shape[1], dtype=np.float32)


def _random(X_train, y_train, k, seed):
    rng = np.random.RandomState(seed)
    idx = rng.choice(X_train.shape[1], size=k, replace=False)
    return _mask_from_indices(idx, X_train.shape[1])


def _variance(X_train, y_train, k, seed):
    var = X_train.var(axis=0)
    return _mask_from_indices(_topk(var, k), X_train.shape[1])


def _mutual_info(X_train, y_train, k, seed):
    sel = SelectKBest(mutual_info_classif, k=k).fit(X_train, y_train)
    return _mask_from_indices(sel.get_support(indices=True), X_train.shape[1])


def _f_classif(X_train, y_train, k, seed):
    sel = SelectKBest(f_classif, k=k).fit(X_train, y_train)
    return _mask_from_indices(sel.get_support(indices=True), X_train.shape[1])


def _chi2(X_train, y_train, k, seed):
    Xnn = MinMaxScaler().fit_transform(X_train)
    sel = SelectKBest(chi2, k=k).fit(Xnn, y_train)
    return _mask_from_indices(sel.get_support(indices=True), X_train.shape[1])


def _l1_logreg(X_train, y_train, k, seed):
    # saga supports L1 + multiclass natively (liblinear is binary-only and
    # raises on n_classes >= 3). saga is slower but is the right solver here.
    base = LogisticRegression(penalty="l1", solver="saga",
                              C=0.1, random_state=seed, max_iter=2000,
                              tol=1e-3)
    sel = SelectFromModel(base, max_features=k, threshold=-np.inf).fit(X_train, y_train)
    return _mask_from_indices(sel.get_support(indices=True), X_train.shape[1])


def _rf_importance(X_train, y_train, k, seed):
    base = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
    sel = SelectFromModel(base, max_features=k, threshold=-np.inf).fit(X_train, y_train)
    return _mask_from_indices(sel.get_support(indices=True), X_train.shape[1])


def _rfe_logreg(X_train, y_train, k, seed):
    base = LogisticRegression(max_iter=1000, random_state=seed)
    sel = RFE(base, n_features_to_select=k, step=0.1).fit(X_train, y_train)
    return _mask_from_indices(sel.get_support(indices=True), X_train.shape[1])


def _lap_score(X_train, y_train, k, seed):
    W = construct_W(X_train, **{"metric": "euclidean", "neighbor_mode": "knn",
                                "weight_mode": "heat_kernel", "k": 5, "t": 1.0})
    scores = lap_score.lap_score(X_train, W=W)
    return _mask_from_indices(_topk(-scores, k), X_train.shape[1])  # lower is better


def _fisher(X_train, y_train, k, seed):
    scores = fisher_score.fisher_score(X_train, y_train)
    return _mask_from_indices(_topk(scores, k), X_train.shape[1])


def _mrmr(X_train, y_train, k, seed):
    Xdf = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X_train.shape[1])])
    yser = pd.Series(y_train)
    selected = mrmr_classif(X=Xdf, y=yser, K=k, show_progress=False)
    idx = [int(name[1:]) for name in selected]
    return _mask_from_indices(idx, X_train.shape[1])


def _mcfs(X_train, y_train, k, seed):
    W = construct_W(X_train, **{"metric": "euclidean", "neighbor_mode": "knn",
                                "weight_mode": "heat_kernel", "k": 5, "t": 1.0})
    n_clusters = max(2, len(np.unique(y_train)))
    W_mcfs = MCFS.mcfs(X_train, n_selected_features=k, W=W, n_clusters=n_clusters)
    scores = (W_mcfs ** 2).max(axis=1)
    return _mask_from_indices(_topk(scores, k), X_train.shape[1])


SKLEARN_METHODS = {
    "all": _all,
    "random": _random,
    "variance": _variance,
    "mutual_info": _mutual_info,
    "f_classif": _f_classif,
    "chi2": _chi2,
    "l1_logreg": _l1_logreg,
    "rf_importance": _rf_importance,
    "rfe_logreg": _rfe_logreg,
}

SKFEATURE_METHODS = {
    "lap_score": _lap_score,
    "fisher": _fisher,
    "mcfs": _mcfs,
}

MRMR_METHODS = {
    "mrmr": _mrmr,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_one(method_name, fn, X_train, y_train, X_test, y_test, k, seed):
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = fn(X_train, y_train, k, seed)
    fit_time = time.time() - t0

    eval_result = evaluate_final_selection(X_train, y_train, X_test, y_test, mask)
    return {
        "method": method_name,
        "k": int(mask.sum()),
        "k_target": k,
        "rf_accuracy": eval_result["rf_accuracy"],
        "fit_time": fit_time,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, default="synthetic",
                    help="comma-separated dataset names")
    ap.add_argument("--methods", type=str, default="chi2,l1_logreg,mrmr,rfe_logreg",
                    help="comma-separated method names. Default: chi2, l1_logreg "
                         "(LASSO), mrmr, rfe_logreg. Pass 'all' to run every "
                         "available method.")
    ap.add_argument("--k", type=str, default="10,20,30",
                    help="comma-separated target feature counts (caps at n_features)")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--save-dir", type=str, default="results/baselines")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    available = dict(SKLEARN_METHODS)
    if SKFEATURE_OK:
        available.update(SKFEATURE_METHODS)
    else:
        print("[note] skfeature not installed — skipping lap_score / fisher / mcfs.")
        print("       pip install skfeature-chappers")
    if MRMR_OK:
        available.update(MRMR_METHODS)
    else:
        print("[note] mrmr not installed — skipping mrmr.")
        print("       pip install mrmr-selection")

    if args.methods.strip().lower() == "all":
        methods = list(available.keys())
    else:
        methods = args.methods.split(",")
    methods = [m for m in methods if m in available]
    if not methods:
        raise SystemExit(f"No valid methods. Available: {sorted(available.keys())}")

    datasets = args.datasets.split(",")
    ks = [int(x) for x in args.k.split(",")]
    seeds = list(range(args.n_seeds))

    all_results = {}
    for ds in datasets:
        all_results[ds] = []
        print(f"\n{'='*70}\nDataset: {ds}\n{'='*70}")
        for seed in seeds:
            X_train, X_test, y_train, y_test, _ = load_dataset(ds, seed=seed)
            n_features = X_train.shape[1]
            print(f"  seed={seed} n_features={n_features} "
                  f"train={len(X_train)} test={len(X_test)}")

            for k in ks:
                k_eff = min(k, n_features)
                for m in methods:
                    if m == "all" and k != ks[0]:
                        continue  # don't repeat
                    try:
                        r = run_one(m, available[m], X_train, y_train,
                                    X_test, y_test, k_eff, seed)
                    except Exception as e:
                        print(f"    {m:<14} k={k_eff:<4} FAILED: {e}")
                        continue
                    r["dataset"] = ds
                    r["seed"] = seed
                    print(f"    {m:<14} k={r['k']:<4} "
                          f"acc={r['rf_accuracy']:.4f}  ({r['fit_time']:.2f}s)")
                    all_results[ds].append(r)

    out_path = os.path.join(args.save_dir, "baselines.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print_summary(all_results)


def print_summary(all_results):
    print("\n" + "=" * 78)
    print("BASELINE SUMMARY (mean ± std over seeds)")
    print("=" * 78)
    for ds, runs in all_results.items():
        print(f"\n{ds}")
        print(f"  {'method':<14} {'k':<6} {'acc':<18} {'time(s)':<10}")
        agg = {}
        for r in runs:
            agg.setdefault((r["method"], r["k_target"]), []).append(r)
        for (m, k), rs in sorted(agg.items()):
            accs = np.array([x["rf_accuracy"] for x in rs])
            ts = np.array([x["fit_time"] for x in rs])
            print(f"  {m:<14} {k:<6} {accs.mean():.4f}±{accs.std():.4f}   "
                  f"{ts.mean():.2f}")


if __name__ == "__main__":
    main()
