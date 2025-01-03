import numpy as np
import pytest

from timefiller import FastRidge


@pytest.fixture
def data():
    np.random.seed(42)
    X = np.random.rand(10, 5)  # 10 échantillons, 5 features
    y = np.random.rand(10)  # 10 targets
    return X, y


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


def test_sample_weight(data):
    X, y = data

    sample_weight = np.array([0] + [1] * (X.shape[0] - 1))
    model_weighted = FastRidge(fit_intercept=True, regularization=1e-2)
    model_weighted.fit(X, y, sample_weight=sample_weight)

    X_filtered = X[1:]
    y_filtered = y[1:]
    model_filtered = FastRidge(fit_intercept=True, regularization=1e-2)
    model_filtered.fit(X_filtered, y_filtered)

    assert np.allclose(model_weighted.coef_, model_filtered.coef_)
    assert np.allclose(model_weighted.intercept_, model_filtered.intercept_)


def test_colinear_sample_weights(data):
    X, y = data
    sample_weight_1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sample_weight_2 = 20 * sample_weight_1

    model_1 = FastRidge(fit_intercept=True, regularization=1e-2)
    model_1.fit(X, y, sample_weight=sample_weight_1)

    model_2 = FastRidge(fit_intercept=True, regularization=1e-2)
    model_2.fit(X, y, sample_weight=sample_weight_2)

    assert np.allclose(model_1.coef_, model_2.coef_)
    assert np.allclose(model_1.intercept_, model_2.intercept_)


def test_different_sample_weights(data):
    X, y = data
    sample_weight_1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sample_weight_2 = sample_weight_1[::-1]

    model_1 = FastRidge(fit_intercept=True, regularization=1e-2)
    model_1.fit(X, y, sample_weight=sample_weight_1)

    model_2 = FastRidge(fit_intercept=True, regularization=1e-2)
    model_2.fit(X, y, sample_weight=sample_weight_2)

    assert not np.allclose(model_1.coef_, model_2.coef_)
