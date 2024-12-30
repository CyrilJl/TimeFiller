from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_spd_matrix


def generate_random_time_series(
    n,
    start,
    freq,
    periods,
    cov=None,
    mean=None,
    kernel="exponential",
    window=12,
) -> pd.DataFrame:
    """
    Generates a set of correlated time series with optional covariance, mean, and smoothing kernel.

    Args:
        n (int): Number of dimensions (series).
        start (str or datetime-like): Start date for the time series.
        freq (str): Frequency string for the time index (e.g., 'D' for daily, 'h' for hourly).
        periods (int): Number of periods to generate.
        cov (ndarray, optional): Covariance matrix (if None, a random symmetric positive definite matrix is used).
            Defaults to None.
        mean (ndarray, optional): Mean vector (if None, a random mean vector is used). Defaults to None.
        kernel (str, optional): Type of kernel for smoothing ('exponential' or 'gaussian'). Defaults to 'exponential'.
        window (int, optional): The size of the smoothing window. Defaults to 12.

    Returns:
        pd.DataFrame: A pandas DataFrame with the generated time series and a time index.

    """
    if cov is None:
        cov = make_spd_matrix(n)
    if mean is None:
        mean = np.random.randn(n)

    X = np.random.multivariate_normal(cov=cov, mean=mean, size=periods + window)

    if kernel == "exponential":
        weights = np.flip(np.exp(-np.linspace(0, 5, window)))
    elif kernel == "gaussian":
        weights = np.flip(np.exp(-(np.linspace(0, np.sqrt(5), window) ** 2)))

    weights /= weights.sum()

    X_transformed = np.zeros((periods, n))
    for k in range(window, periods + window):
        X_transformed[k - window] = weights @ X[k - window : k]

    time_index = pd.date_range(start=start, periods=periods, freq=freq)
    df = pd.DataFrame(X_transformed, index=time_index, columns=[f"serie {i+1}" for i in range(n)])
    return df


def add_mar_nan(df, ratio) -> Tuple[pd.DataFrame, np.ndarray]:
    """Create Missing at Random Values"""
    df_copy = df.copy()
    total_values = df_copy.size
    n_nan = int(total_values * ratio)
    nan_indices = np.random.choice(total_values, n_nan, replace=False)
    nan_indices = np.unravel_index(nan_indices, df_copy.shape)
    if isinstance(df, np.ndarray):
        df_copy[nan_indices] = np.nan
    if isinstance(df, pd.DataFrame):
        df_copy.values[nan_indices] = np.nan
    return df_copy
