import numpy as np
import pandas as pd
from numba import njit, prange
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["PositiveOutput"]


class PositiveOutput(TransformerMixin, BaseEstimator):
    """Transforms values below a threshold by extending them into the negative domain.

    This class transforms values in an array or DataFrame that are below a certain threshold
    by extending them into the negative domain. Values greater than or equal to the threshold
    remain unchanged.

    Attributes:
        q (int or float, optional): The percentile used to calculate the thresholds. Default is 10.
        v (float, optional): The fixed threshold value to use. Default is None.
        thresholds_ (np.ndarray): The calculated or fixed thresholds for each column.
    """

    def __init__(self, q=10, v=None, columns=None):
        """Initializes the PositiveOutput object.

        Args:
            q (int or float, optional): The percentile used to calculate the thresholds. Default is 10.
            v (float, optional): The fixed threshold value to use. Default is None.

        Raises:
            ValueError: If both `q` and `v` arguments are None.
        """
        if q is None and v is None:
            raise ValueError("At least one of the arguments 'q' or 'v' must be provided.")
        self.q = q
        self.v = v
        self.columns = columns
        self.thresholds_ = v

    def fit(self, X, y=None):
        """Computes the thresholds from the input data.

        Args:
            X (np.ndarray or pd.DataFrame): The input data.
            y (ignored): Not used, present for compatibility with the scikit-learn API.

        Returns:
            self: The instance of the PositiveOutput object.

        Raises:
            ValueError: If the data contains negative values.
        """
        if np.nanmin(X) < 0:
            raise ValueError("The data must not contain negative values.")

        if isinstance(X, np.ndarray):
            if self.v is None:
                self.thresholds_ = np.nanpercentile(X, q=self.q, axis=0)
            else:
                self.thresholds_ = np.full(shape=X.shape[1], fill_value=self.v)
        if isinstance(X, pd.DataFrame):
            if self.columns is None:
                if self.v is None:
                    self.thresholds_ = X.quantile(q=self.q / 100.0).values
                else:
                    self.thresholds_ = pd.Series(data=self.v, index=X.columns).values
            else:
                if self.v is None:
                    self.thresholds_ = X[self.columns].quantile(q=self.q / 100.0).values
                else:
                    self.thresholds_ = pd.Series(data=self.v, index=self.columns).values
        return self

    @staticmethod
    @njit(parallel=True, boundscheck=False, fastmath=True, cache=True)
    def transform_numpy(X, thresholds):
        result = np.empty_like(X)
        if isinstance(thresholds, (float, int)):
            thresholds = np.full(shape=X.shape[1], fill_value=thresholds)
        for i in prange(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i, j] < thresholds[j]:
                    result[i, j] = 2 * X[i, j] - thresholds[j]
                else:
                    result[i, j] = X[i, j]
        return result

    @staticmethod
    @njit(parallel=True, boundscheck=False, fastmath=True, cache=True)
    def inverse_transform_numpy(X, thresholds):
        result = np.empty_like(X)
        if isinstance(thresholds, (float, int)):
            thresholds = np.full(shape=X.shape[1], fill_value=thresholds)
        for i in prange(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i, j] < thresholds[j]:
                    result[i, j] = 0.5 * X[i, j] + 0.5 * thresholds[j]
                else:
                    result[i, j] = X[i, j]
        return np.maximum(0, result)

    def transform(self, X, y=None):
        """Transforms the data by extending values below the threshold.

        Args:
            X (np.ndarray or pd.DataFrame): The data to transform.
            y (ignored): Not used, present for compatibility with the scikit-learn API.

        Returns:
            np.ndarray or pd.DataFrame: The transformed data.
        """
        if isinstance(X, np.ndarray):
            return self.transform_numpy(X, self.thresholds_)
        if isinstance(X, pd.DataFrame):
            if self.columns is not None:
                a = X.drop(columns=self.columns)
                b = pd.DataFrame(
                    self.transform_numpy(X[self.columns].values, self.thresholds_), index=X.index, columns=self.columns
                )
                return pd.concat([a, b], axis=1)[X.columns]
            else:
                return pd.DataFrame(self.transform_numpy(X.values, self.thresholds_), index=X.index, columns=X.columns)

    def inverse_transform(self, X, y=None):
        """Reverses the transformation by bringing the extended values back into the positive domain.

        Args:
            X (np.ndarray or pd.DataFrame): The data to reverse.
            y (ignored): Not used, present for compatibility with the scikit-learn API.

        Returns:
            np.ndarray or pd.DataFrame: The reversed data.
        """
        if isinstance(X, np.ndarray):
            return self.inverse_transform_numpy(X, self.thresholds_)
        if isinstance(X, pd.DataFrame):
            if self.columns is not None:
                a = X.drop(columns=self.columns)
                b = pd.DataFrame(
                    self.inverse_transform_numpy(X[self.columns].values, self.thresholds_),
                    index=X.index,
                    columns=self.columns,
                )
                return pd.concat([a, b], axis=1)[X.columns]
            else:
                return pd.DataFrame(
                    self.inverse_transform_numpy(X.values, self.thresholds_), index=X.index, columns=X.columns
                )
