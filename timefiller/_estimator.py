import numpy as np
from scipy.linalg import cho_factor, cho_solve


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

    Methods:
        fit(X, y):
            Fits the Ridge Regression model to the input data.
        predict(X):
            Predicts target values for the given input data using the fitted model.
    """

    def __init__(self, fit_intercept=True, regularization=1e-2):
        """
        Initializes the FastRidge model.

        Args:
            regularization (float): The regularization strength parameter. Default is 1e-2.
        """
        self.fit_intercept = fit_intercept
        self.regularization = regularization

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

        Returns:
            FastRidge: The fitted model instance with updated coefficients and intercept.
        """
        n_samples, n_features = X.shape

        if self.fit_intercept:
            a = np.empty(shape=(n_features + 1, n_features + 1), dtype=X.dtype)
            a[np.ix_(range(n_features), range(n_features))] = X.T @ X
            np.fill_diagonal(a, (1 + self.regularization) * a.diagonal())
            a[-1, :-1] = a[:-1, -1] = X.sum(axis=0)
            a[-1, -1] = n_samples

            b = np.concatenate([X.T @ y, [y.sum()]])

            cho_factorized = cho_factor(a)
            w = cho_solve(cho_factorized, b)

            self.coef_ = w[:-1]
            self.intercept_ = w[-1]
        else:
            a = X.T @ X
            np.fill_diagonal(a, (1 + self.regularization) * a.diagonal())
            b = X.T @ y

            self.coef_ = np.linalg.solve(a, b)
            self.intercept_ = 0

        return self

    def predict(self, X):
        """
        Predicts target values for the given input data.

        Args:
            X (numpy.ndarray): Input data matrix of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted target values of shape (n_samples,).
        """
        return X @ self.coef_ + self.intercept_
