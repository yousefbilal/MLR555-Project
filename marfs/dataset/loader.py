"""Dataset loading utilities for MARFS.

Supports all 9 EAC-FS benchmark datasets plus additional ones.
"""

import numpy as np
from sklearn.datasets import fetch_covtype, fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .synthetic import make_synthetic_dataset


def load_dataset(name: str, test_size: float = 0.2, seed: int = 42, max_rows: int = 20000):
    """Load and preprocess a dataset.

    Returns:
        X_train, X_test, y_train, y_test: numpy arrays
        feature_names: list of feature name strings
    """
    loaders = {
        # 9 EAC-FS benchmark datasets
        "wdbc": _load_wdbc,
        "usps": _load_usps,
        "isolet": _load_isolet,
        # "orl": _load_orl,
        # "yale": _load_yale,
        "coil20": _load_coil20,
        "colon": _load_colon,
        "covertype": _load_covertype,
        # Additional
        "musk": _load_musk,
        "spambase": _load_spambase,
        "mnist": _load_mnist,
        "genomics": _load_genomics,
        "synthetic": _load_synthetic,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(loaders.keys())}")

    X, y, feature_names = loaders[name]()

    if len(X) > max_rows:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), max_rows, replace=False)
        X, y = X[idx], y[idx]

    if y.dtype.kind not in ("i", "u"):
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, X_test, y_train, y_test, feature_names


# ---------------------------------------------------------------------------
# EAC-FS benchmark datasets
# ---------------------------------------------------------------------------

def _load_wdbc():
    """Wisconsin Diagnostic Breast Cancer: 30 features."""
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)
    feature_names = list(data.feature_names)
    return X, y, feature_names



def _load_usps():
    """USPS Handwritten Digits: 256 features."""
    data = fetch_openml("usps", version=2, as_frame=False, parser="auto")
    X = data.data.astype(np.float32)
    y = data.target
    feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
    return X, y, feature_names

def _load_spambase():
    data = fetch_openml(data_id=1116, as_frame=False, parser="auto")
    X = data.data.astype(np.float32)
    y = data.target
    feature_names = data.feature_names
    return X, y, feature_names

def _load_musk():
    data = fetch_openml(data_id=44, as_frame=False, parser="auto")
    X = data.data.astype(np.float32)
    y = data.target
    feature_names = data.feature_names
    return X, y, feature_names


def _load_isolet():
    """Isolet Spoken Letter Recognition: 617 features."""
    data = fetch_openml(id=300, as_frame=False, parser="auto")
    X = data.data.astype(np.float32)
    y = data.target
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]
    return X, y, feature_names


# def _load_orl():
#     """ORL/AT&T Faces: 1024 features (32x32 images, 40 subjects)."""
#     try:
#         data = fetch_openml(data_id=41083, as_frame=False, parser="auto")
#         X = data.data.astype(np.float32)
#         y = data.target
#         feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
#     except Exception:
#         print("ORL dataset unavailable from OpenML, using synthetic fallback (1024 features).")
#         X, y, feature_names = make_synthetic_dataset(
#             n_samples=400, n_features=1024, n_informative=50,
#             n_redundant=300, n_classes=40, seed=42
#         )
#     return X, y, feature_names


# def _load_yale():
#     """Yale Extended Faces: 1024 features (32x32 images, 15 subjects)."""
#     try:
#         data = fetch_openml(data_id=41069, as_frame=False, parser="auto")
#         X = data.data.astype(np.float32)
#         y = data.target
#         feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
#     except Exception:
#         print("Yale dataset unavailable from OpenML, using synthetic fallback (1024 features).")
#         X, y, feature_names = make_synthetic_dataset(
#             n_samples=165, n_features=1024, n_informative=50,
#             n_redundant=300, n_classes=15, seed=42
#         )
#     return X, y, feature_names


def _load_coil20():
    """COIL-20 Objects: 1024 features (32x32 images, 20 objects)."""
    data = fetch_openml(data_id=46783, as_frame=False, parser="auto")
    X = data.data.astype(np.float32)
    y = data.target
    feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
    return X, y, feature_names


def _load_colon():
    """Colon Cancer Gene Expression: 2000 features, 62 samples."""
    data = fetch_openml(data_id=45087, as_frame=False, parser="auto")
    X = data.data.astype(np.float32)
    y = data.target
    feature_names = [f"gene_{i}" for i in range(X.shape[1])]
    return X, y, feature_names


def _load_covertype():
    """Forest Cover Type: 56 features, ~580k rows."""
    data = fetch_covtype()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)
    feature_names = data.feature_names
    return X, y, feature_names


# ---------------------------------------------------------------------------
# Additional datasets
# ---------------------------------------------------------------------------

def _load_mnist():
    """MNIST flattened: 784 features, 70k rows."""
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int64)
    feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
    return X, y, feature_names


def _load_genomics():
    """Genomics dataset from OpenML (SRBCT: 2308 features)."""
    data = fetch_openml(data_id=45101, as_frame=False, parser="auto")
    X = data.data.astype(np.float32)
    y = data.target
    feature_names = [f"gene_{i}" for i in range(X.shape[1])]

    return X, y, feature_names


def _load_synthetic():
    """Synthetic dataset with controlled correlation structure."""
    X, y, feature_names = make_synthetic_dataset(
        n_samples=5000, n_features=200, n_informative=20,
        n_redundant=80, n_classes=4, seed=42
    )
    return X, y, feature_names
