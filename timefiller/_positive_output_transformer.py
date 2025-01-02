import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

__all__ = ["PositiveOutput"]


class PositiveOutput(TransformerMixin):
    """
    A transformer that applies negative expansion to data based on a threshold.

    Args:
        q (float, optional): The quantile used as a threshold for expansion. Default is `10`,
            which means the 10th percentile is used as the threshold. If `v` is provided,
            `q` is ignored.
        v (float, optional): A fixed value used as a threshold for negative expansion.
            If provided, this threshold will be used for all features. Default is `None`,
            which means the threshold is automatically calculated from the data.
        columns (list, optional): List of column names to process if the input is a DataFrame.
            If `None`, all columns will be processed. Default is `None`.

    Raises:
        ValueError: If both `q` and `v` are `None`.
    """

    def __init__(self, q=10, v=None, columns=None):
        if q is None and v is None:
            raise ValueError("At least one of the arguments 'q' or 'v' must be different from None.")

        self.q = q
        self.v = v
        self.columns = columns
        self.thresholds_ = None

    def fit(self, X, y=None):
        """
        Calculate and store the thresholds necessary for negative expansion.

        Args:
            X (array-like or DataFrame): The training data. Must not contain negative values.
            y (array-like, optional): The training labels. Not used in this method.

        Returns:
            PositiveOutput: The fitted instance of the transformer.

        Raises:
            ValueError: If the data contains negative values.
        """
        if isinstance(X, pd.DataFrame):
            if self.columns is not None:
                X_subset = X[self.columns]
                self.columns_ = self.columns
            else:
                X_subset = X
                self.columns_ = list(X.columns)
        if isinstance(X, np.ndarray):
            X_subset = X

        if np.nanmin(X_subset) < 0:
            raise ValueError("The data must not contain negative values.")

        if self.v is None:
            self.thresholds_ = np.nanpercentile(X_subset, q=self.q, axis=0)
        else:
            self.thresholds_ = np.full(shape=X_subset.shape[1], fill_value=self.v)
        return self

    def transform(self, X, y=None):
        """
        Apply negative expansion on the data.

        Args:
            X (array-like or DataFrame): The data to transform.
            y (array-like, optional): The labels. Not used in this method.

        Returns:
            array-like or DataFrame: The transformed data with negative expansion.
        """
        if isinstance(X, pd.DataFrame):
            if self.columns is not None:
                X_subset = X[self.columns]
            else:
                X_subset = X
        else:
            X_subset = X

        transformed = np.where(X_subset < self.thresholds_, 2 * X_subset - self.thresholds_, X_subset)

        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            X_transformed[self.columns_] = transformed
            return X_transformed
        else:
            return transformed

    def inverse_transform(self, X, y=None):
        """
        Reverse the negative expansion on the transformed data.

        Args:
            X (array-like or DataFrame): The transformed data to invert.
            y (array-like, optional): The labels. Not used in this method.

        Returns:
            array-like or DataFrame: The original data after reversing the negative expansion.
        """
        if isinstance(X, pd.DataFrame):
            if self.columns is not None:
                X_subset = X[self.columns]
            else:
                X_subset = X
        else:
            X_subset = X

        inverted = np.maximum(0, np.where(X_subset < self.thresholds_, 0.5 * X_subset + self.thresholds_ / 2, X_subset))

        if isinstance(X, pd.DataFrame):
            X_inverted = X.copy()
            X_inverted[self.columns_] = inverted
            return X_inverted
        else:
            return inverted
