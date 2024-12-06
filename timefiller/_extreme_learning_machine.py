import numpy as np
from numba import njit, prange
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._misc import check_params


class ExtremeLearningMachineRegressor(BaseEstimator, RegressorMixin):
    """
    This estimator first applies a random projection to the input features,
    followed by a ReLU activation function, and then fits a Ridge regression
    model on the transformed features. The random projection helps in learning
    non-linear patterns in the data.

    Args:        
        ratio_features_projection (float, optional): The ratio determining the number of random
            features relative to input features. Used only if `n_features_projection` is None.
            Must be greater than 0 if used. Defaults to 10.
        n_features_projection (int, optional): The number of random projection features.
            If None, `ratio_features_projection` is used to determine this value.
            Defaults to None.
        alpha (float, optional): Regularization term for the Ridge regressor. Defaults to 1e-3
        random_state (int, RandomState instance or None, optional): Controls the
            randomness of the random projection. Pass an int for reproducible
            results across multiple function calls. Defaults to None.

    Raises:
        ValueError: If both `n_features_projection` and `ratio_features_projection` are None,
            or if `ratio_features_projection` is <= 0.

    Example:
        .. code-block:: python

            from sklearn.datasets import fetch_california_housing
            from sklearn.linear_model import Ridge
            from sklearn.model_selection import cross_val_score
            from timefiller import ExtremeLearningMachine

            X, y = fetch_california_housing(as_frame=True, return_X_y=True)

            for estimator in (Ridge(), ExtremeLearningMachine(ratio_features_projection=3), ExtremeLearningMachine(ratio_features_projection=20)):
                print(estimator, cross_val_score(estimator=estimator, X=X, y=y).mean())
            >>> Ridge() 0.5530311140279566
            >>> ExtremeLearningMachine(ratio_features_projection=3) 0.5276448340024226
            >>> ExtremeLearningMachine(ratio_features_projection=20) 0.6344905228311806
    """

    def __init__(self, ratio_features_projection=10., n_features_projection=None, alpha=1e-3, random_state=None):
        if n_features_projection is None and ratio_features_projection is None:
            raise ValueError("Either 'n_features_projection' or 'ratio_features_projection' must be set.")
        if ratio_features_projection is not None and ratio_features_projection <= 0:
            raise ValueError("The 'ratio_features_projection' parameter must be greater than 0.")
        self.n_features_projection = check_params(param=n_features_projection, types=(int, type(None)))
        self.ratio_features_projection = check_params(param=ratio_features_projection, types=(float, int, type(None)))
        self.alpha = alpha
        self.random_state = random_state

    @staticmethod
    @njit(parallel=True, boundscheck=False)
    def _transform(X, means, W, b):
        n_samples, n_features = X.shape
        n_features_projection = W.shape[1]
        Xt = np.empty((n_samples, n_features_projection), dtype=X.dtype)

        for i in prange(n_samples):
            for j in range(n_features_projection):
                acc = 0.  # Accumulate for the dot product
                for k in range(n_features):
                    acc += (X[i, k] - means[k]) * W[k, j]
                Xt[i, j] = max(acc + b[j], 0)
        return Xt

    @staticmethod
    @njit(parallel=True, boundscheck=False)
    def _transform_predict(X, means, W, b, coef, intercept):
        n_samples, n_features = X.shape
        n_features_projection = W.shape[1]
        predictions = np.empty(n_samples, dtype=X.dtype)

        for i in prange(n_samples):
            pred = intercept  # Start with intercept for each sample
            for j in range(n_features_projection):
                acc = 0.0  # Accumulate for the dot product
                for k in range(n_features):
                    acc += (X[i, k] - means[k]) * W[k, j]
                pred += max(acc + b[j], 0) * coef[j]
            predictions[i] = pred
        return predictions

    def _check(self, X):
        check_is_fitted(self, ["linear_", "scaler_", "W_", "b_"])
        X = check_array(X, accept_sparse=False, ensure_2d=True)
        if X.shape[1] != self.W_.shape[0]:
            raise ValueError(f"Expected {self.W_.shape[0]} features, but got {X.shape[1]}.")
        return X

    def fit(self, X, y, sample_weight=None):
        """
        Fits the Extreme Learning Machine model to the training data.

        This method applies a random projection to the input features, followed by a
        ReLU activation function, and then fits a linear regression model on the transformed
        features.

        Args:
            X (array-like of shape (n_samples, n_features)): The input data.
            y (array-like of shape (n_samples,)): The target values.
            sample_weight (array-like of shape (n_samples,), optional): Individual
                weights for each sample. If None, all samples are given equal weight.

        Returns:
            self: The fitted estimator.

        Raises:
            ValueError: If `X` and `y` have incompatible shapes or if the number of features
                in `X` does not match the number of features expected by the model.
        """
        X, y = check_X_y(X, y, accept_sparse=False, ensure_2d=True)

        self.n_features_in_ = X.shape[1]
        self.scaler_ = StandardScaler().fit(X, sample_weight=sample_weight)
        rng = check_random_state(self.random_state)

        # Determine the number of projection features
        n_random_features = (self.n_features_projection
                             if self.n_features_projection is not None
                             else max(1, int(self.ratio_features_projection * self.n_features_in_))
                             )

        # Initialize random weights and bias for transformation
        self.W_ = rng.randn(X.shape[1], n_random_features)/self.scaler_.scale_[:, None]
        self.b_ = rng.randn(n_random_features)

        Xt = self._transform(X=X, means=self.scaler_.mean_, W=self.W_, b=self.b_)
        self.linear_ = Ridge(alpha=self.alpha).fit(Xt, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        """
        Predicts target values for samples in `X`.

        This method transforms the input data using the random projection and ReLU activation
        learned during fitting, and then uses the linear regression model to make predictions.

        Args:
            X (array-like of shape (n_samples, n_features)): The input data for which predictions are to be made.

        Returns:
            y_pred (ndarray of shape (n_samples,)): The predicted values for each input sample.

        Raises:
            ValueError: If `X` has an unexpected number of features or if the model has not
                been fitted before calling this method.
        """
        X = self._check(X)
        return self._transform_predict(X=X, means=self.scaler_.mean_, W=self.W_,
                                       b=self.b_, coef=self.linear_.coef_,
                                       intercept=self.linear_.intercept_)
