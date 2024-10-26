from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._misc import check_params


class ExtremeLearningMachine(BaseEstimator, RegressorMixin):
    """
    This estimator first applies a random projection to the input features,
    followed by a ReLU activation function, and then fits a linear regression
    model on the transformed features. The random projection helps in learning
    non-linear patterns in the data.

    Args:
        n_features_projection (int, optional): The number of random projection features.
            If None, `ratio_features_projection` is used to determine this value.
            Defaults to None.
        ratio_features_projection (float, optional): The ratio determining the number of random
            features relative to input features. Used only if `n_features_projection` is None.
            Must be greater than 0 if used. Defaults to 1.5.
        random_state (int, RandomState instance or None, optional): Controls the
            randomness of the random projection. Pass an int for reproducible
            results across multiple function calls. Defaults to None.

    Raises:
        ValueError: If both `n_features_projection` and `ratio_features_projection` are None,
            or if `ratio_features_projection` is <= 0.
    """

    def __init__(self, ratio_features_projection=1.5, n_features_projection=None, random_state=None):
        if n_features_projection is None and ratio_features_projection is None:
            raise ValueError("Either 'n_features_projection' or 'ratio_features_projection' must be set.")
        if ratio_features_projection is not None and ratio_features_projection <= 0:
            raise ValueError("The 'ratio_features_projection' parameter must be greater than 0.")
        self.n_features_projection = check_params(param=n_features_projection, types=(int, type(None)))
        self.ratio_features_projection = check_params(param=ratio_features_projection, types=(float, int, type(None)))
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, accept_sparse=False, ensure_2d=True)

        self.n_features_in_ = X.shape[1]
        self.scaler_ = StandardScaler().fit(X)
        rng = check_random_state(self.random_state)

        # Determine the number of projection features
        n_random_features = (self.n_features_projection
                             if self.n_features_projection is not None
                             else max(1, int(self.ratio_features_projection * self.n_features_in_))
                             )

        # Initialize random weights and bias for transformation
        self.W_ = rng.randn(X.shape[1], n_random_features)
        self.b_ = rng.randn(n_random_features)

        Xt = self.scaler_.transform(X) @ self.W_ + self.b_
        Xt[Xt < 0] = 0  # ReLU activation

        self.linear_ = LinearRegression().fit(Xt, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        check_is_fitted(self, ["linear_", "scaler_", "W_", "b_"])
        X = check_array(X, accept_sparse=False, ensure_2d=True)
        if X.shape[1] != self.W_.shape[0]:
            raise ValueError(f"Expected {self.W_.shape[0]} features, but got {X.shape[1]}.")

        Xt = self.scaler_.transform(X) @ self.W_ + self.b_
        Xt[Xt < 0] = 0  # ReLU activation
        return self.linear_.predict(Xt)
