import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ._misc import check_params


class FastRidge:
    """
    Ridge Regression implementation with adaptive regularization.

    This implementation avoids the overhead of scikit-learn by providing a custom
    Ridge Regression solution. Instead of a constant regularization term
    (alpha * I), it uses an adaptive regularization term (alpha * diag(X @ X.T)).

    Attributes:
        regularization (float): The regularization strength parameter. Default is 1e-2.
        coef_ (numpy.ndarray): Coefficients of the features after fitting the model.
        intercept_ (float): Intercept term after fitting the model.
    """

    def __init__(self, fit_intercept=True, regularization=1e-2):
        self.fit_intercept = check_params(param=fit_intercept, types=bool)
        self.regularization = check_params(param=regularization, types=(int, float))

    def __repr__(self):
        return f"FastRidge(fit_intercept={self.fit_intercept}, regularization={self.regularization})"

    def get_params(self, deep=True):
        return {"regularization": self.regularization, "fit_intercept": self.fit_intercept}

    def fit(self, X, y, sample_weight=None):
        """
        Fits the Ridge Regression model to the input data.

        Args:
            X (numpy.ndarray): Input data matrix of shape (n_samples, n_features).
            y (numpy.ndarray): Target values of shape (n_samples,).
            sample_weight (numpy.ndarray): Sample weights of shape (n_samples,). Default is None.

        Returns:
            FastRidge: The fitted model instance with updated coefficients and intercept.
        """
        n_samples, self.n_features_ = X.shape

        if sample_weight is not None:
            if len(sample_weight) != n_samples:
                raise ValueError("Sample weights must have the same length as the input data.")
            sample_weight = np.asarray(sample_weight).astype(float)
            if sample_weight.ndim != 1:
                raise ValueError("Sample weights must be a 1D array.")
            if not np.all(sample_weight >= 0):
                raise ValueError("Sample weights must be non-negative.")
            sample_weight /= sample_weight.sum()
            sample_weight = sample_weight.reshape(-1, 1)
            X_weighted = X * sample_weight
            y_weighted = y * sample_weight.flatten()
        else:
            X_weighted = X
            y_weighted = y

        if self.fit_intercept:
            a = np.empty(shape=(self.n_features_ + 1, self.n_features_ + 1), dtype=X.dtype)
            a[np.ix_(range(self.n_features_), range(self.n_features_))] = X_weighted.T @ X
            np.fill_diagonal(a, (1 + self.regularization) * a.diagonal())
            a[-1, :-1] = a[:-1, -1] = X_weighted.sum(axis=0)
            a[-1, -1] = sample_weight.sum() if sample_weight is not None else n_samples

            b = np.empty(shape=(self.n_features_ + 1,), dtype=X.dtype)
            b[:-1] = X_weighted.T @ y
            b[-1] = y_weighted.sum()

            w = cho_solve(cho_factor(a), b)

            self.coef_ = w[:-1]
            self.intercept_ = w[-1]
        else:
            a = X_weighted.T @ X
            np.fill_diagonal(a, (1 + self.regularization) * a.diagonal())
            b = X_weighted.T @ y

            self.coef_ = cho_solve(cho_factor(a), b)
            self.intercept_ = 0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values for the given input data.

        Args:
            X (numpy.ndarray): Input data matrix of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted target values of shape (n_samples,).
        """
        return X @ self.coef_ + self.intercept_
