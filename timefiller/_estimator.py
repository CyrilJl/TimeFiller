import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y, check_array


class RandomReluLinear(BaseEstimator, RegressorMixin):
    def __init__(self, ratio=1.5, random_state=None):
        if ratio <= 0:
            raise ValueError("The 'ratio' parameter must be greater than 0.")
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.linear = LinearRegression()

    def fit(self, X, y, sample_weight=None):
        # Check that X and y have correct shape and type
        X, y = check_X_y(X, y, accept_sparse=False, ensure_2d=True)
        
        # Fit scaler and transform input data
        self.scaler.fit(X)
        rng = check_random_state(self.random_state)
        
        # Initialize random weights for transformation
        self.W_ = rng.randn(X.shape[1], max(1, int(self.ratio * X.shape[1])))
        
        # Apply scaling and transformation with ReLU
        Xt = self.scaler.transform(X) @ self.W_
        Xt[Xt < 0] = 0  # ReLU activation
        
        # Fit the linear regression on the transformed data
        self.linear.fit(Xt, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        # Check that X has the correct shape and type
        X = check_array(X, accept_sparse=False, ensure_2d=True)

        # Check for feature mismatch
        if X.shape[1] != self.W_.shape[0]:
            raise ValueError(f"Expected {self.W_.shape[0]} features, but got {X.shape[1]}.")
        
        # Apply scaling and transformation with ReLU
        Xt = self.scaler.transform(X) @ self.W_
        Xt[Xt < 0] = 0  # ReLU activation
        
        # Predict using the linear model on transformed data
        return self.linear.predict(Xt)
