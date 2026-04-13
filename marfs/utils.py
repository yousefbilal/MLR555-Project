import numpy as np

def safe_corrcoef(X, axis=0, min_std=1e-10):
    """
    Compute correlation matrix, safely handling constant columns.
    Any column with std < min_std is set to zero in the output.
    Args:
        X: np.ndarray, shape (n_samples, n_features)
        axis: 0 (features in columns) or 1 (features in rows)
        min_std: float, threshold below which a column is considered constant
    Returns:
        corr: np.ndarray, shape (n_features, n_features)
    """
    X = np.asarray(X)
    if axis == 1:
        X = X.T
    stds = np.std(X, axis=0)
    valid = stds >= min_std
    if not np.any(valid):
        # All columns are constant
        return np.zeros((X.shape[1], X.shape[1]), dtype=np.float32)
    X_valid = X[:, valid]
    corr = np.zeros((X.shape[1], X.shape[1]), dtype=np.float32)
    if X_valid.shape[1] > 1:
        corr_valid = np.corrcoef(X_valid, rowvar=False)
        corr[np.ix_(valid, valid)] = np.nan_to_num(corr_valid, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr
