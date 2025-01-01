import numpy as np
import pytest

from timefiller import FastRidge


def test_fit_with_intercept():
    # Create sample data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])

    # Initialize and fit model
    model = FastRidge(fit_intercept=True)
    fitted_model = model.fit(X, y)

    # Check model attributes
    assert hasattr(fitted_model, "coef_")
    assert hasattr(fitted_model, "intercept_")
    assert fitted_model.coef_.shape == (2,)
    assert isinstance(fitted_model.intercept_, (float, np.floating))


def test_fit_without_intercept():
    # Create sample data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])

    # Initialize and fit model
    model = FastRidge(fit_intercept=False)
    fitted_model = model.fit(X, y)

    # Check model attributes
    assert hasattr(fitted_model, "coef_")
    assert hasattr(fitted_model, "intercept_")
    assert fitted_model.coef_.shape == (2,)
    assert fitted_model.intercept_ == 0


def test_fit_input_validation():
    # Test with invalid input shapes
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2, 3])  # Mismatched dimensions

    model = FastRidge()
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_fit_prediction():
    # Create sample data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])

    # Fit model and make predictions
    model = FastRidge()
    model.fit(X, y)
    predictions = model.predict(X)

    # Check predictions shape
    assert predictions.shape == y.shape
    assert isinstance(predictions, np.ndarray)
