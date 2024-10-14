# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import TransformerMixin

__all__ = ['PositiveOutput']


class PositiveOutput(TransformerMixin):
    """
    Parameters:
    - q (float, optional): The quantile used as a threshold for expansion. 
                            Default is q=10, which means the 10th percentile is used as the threshold.
    - v (float, optional): Fixed value used as a threshold for negative expansion.
                            If `v` is specified, this threshold will be used for all features.
                            Default is v=None, which means the threshold is automatically calculated from the data.
    """

    def __init__(self, q=10, v=None):
        if q is None and v is None:
            raise ValueError("At least one of the arguments 'q' or 'v' must be different from None.")

        self.q = q
        self.v = v
        self.thresholds_ = None

    def fit(self, X, y=None):
        """
        Calculate and store the thresholds necessary for negative expansion.

        Parameters:
        - X (array-like): The training data.
        - y (array-like, optional): The training labels. Not used here.

        Returns:
        - self: The fitted PositiveOutput object.
        """
        if np.nanmin(X) < 0:
            raise ValueError("The data must not contain negative values.")

        if self.v is None:
            self.thresholds_ = np.nanpercentile(X, q=self.q, axis=0)
        else:
            self.thresholds_ = np.full(shape=X.shape[1], fill_value=self.v)
        return self

    def transform(self, X, y=None):
        """
        Apply negative expansion on the data.

        Parameters:
        - X (array-like): The data to transform.
        - y (array-like, optional): The labels. Not used here.

        Returns:
        - array-like: The transformed data with negative expansion.
        """
        X = np.asarray(X)
        mask = X < self.thresholds_
        return np.where(mask, 2 * X - self.thresholds_, X)

    def inverse_transform(self, X, y=None):
        """
        Reverse the negative expansion on the transformed data.

        Parameters:
        - X (array-like): The transformed data.
        - y (array-like, optional): The labels. Not used here.

        Returns:
        - array-like: The inverted data after negative expansion.
        """
        X = np.asarray(X)
        mask = X < self.thresholds_
        return np.maximum(0, np.where(mask, 0.5 * X + self.thresholds_ / 2, X))
