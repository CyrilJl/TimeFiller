import numpy as np
import pytest

from timefiller import FastRidge


@pytest.fixture
def data():
    np.random.seed(42)
    n_samples, n_features = 500, 10
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)
    return X, y


def test_fit_with_intercept():
    """
    Test the `fit` method of the `FastRidge` model with intercept.
    This test creates a sample dataset and fits a `FastRidge` model with the
    `fit_intercept` parameter set to True. It then checks if the fitted model
    has the expected attributes (`coef_` and `intercept_`) and verifies their
    shapes and types.
    Assertions:
        - The fitted model has a `coef_` attribute.
        - The fitted model has an `intercept_` attribute.
        - The shape of `coef_` is (2,).
        - The `intercept_` is an instance of `float` or `np.floating`.
    """
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
    """
    Test the FastRidge model fitting without an intercept.
    This test creates a sample dataset and fits a FastRidge model with the
    fit_intercept parameter set to False. It then checks if the fitted model
    has the expected attributes and values.
    Assertions:
        - The fitted model has a 'coef_' attribute.
        - The fitted model has an 'intercept_' attribute.
        - The shape of the 'coef_' attribute is (2,).
        - The 'intercept_' attribute is 0.
    """
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
    """
    Test the fit and prediction functionality of the FastRidge model.
    This test creates sample data, fits the FastRidge model to the data,
    and makes predictions. It then checks that the predictions have the
    correct shape and type, and that the model's coefficients and intercept
    are of the expected types.
    Assertions:
        - The shape of the predictions should match the shape of the target values.
        - The predictions should be a numpy array.
        - The model's coefficients should be a numpy array.
        - The model's intercept should be a float or numpy floating type.
    """
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
    assert isinstance(model.coef_, np.ndarray)
    assert isinstance(model.intercept_, (float, np.floating))


def test_sample_weight(data):
    """
    Test the FastRidge model with sample weights.
    This test verifies that fitting the FastRidge model with a sample weight
    array that has a zero for the first sample and ones for the rest produces
    the same coefficients and intercept as fitting the model on the filtered
    data (excluding the first sample).
    Parameters:
    data (tuple): A tuple containing the feature matrix X and target vector y.
    Asserts:
    - The coefficients of the weighted model are close to the coefficients of the filtered model.
    - The intercept of the weighted model is close to the intercept of the filtered model.
    """
    X, y = data

    sample_weight = np.array([0] + [1] * (X.shape[0] - 1))
    X_filtered = X[1:]
    y_filtered = y[1:]

    for fit_intercept in [True, False]:
        for regularization in [1e-3, 1e-1, 1e1]:
            model_weighted = FastRidge(fit_intercept=fit_intercept, regularization=regularization)
            model_weighted.fit(X, y, sample_weight=sample_weight)

            model_filtered = FastRidge(fit_intercept=fit_intercept, regularization=regularization)
            model_filtered.fit(X_filtered, y_filtered)

            assert np.allclose(model_weighted.coef_, model_filtered.coef_)
            assert np.allclose(model_weighted.intercept_, model_filtered.intercept_)


def test_colinear_sample_weights(data):
    """
    Test the FastRidge model with colinear sample weights.
    This test checks that the coefficients and intercept of the FastRidge model
    remain consistent when the sample weights are scaled by a constant factor.
    Parameters:
    data (tuple): A tuple containing the feature matrix X and target vector y.
    Asserts:
    - The coefficients of the model trained with the original sample weights
      are close to the coefficients of the model trained with the scaled sample weights.
    - The intercept of the model trained with the original sample weights
      is close to the intercept of the model trained with the scaled sample weights.
    """
    X, y = data
    sample_weight_1 = np.arange(len(X))
    sample_weight_2 = 20 * sample_weight_1

    for fit_intercept in [True, False]:
        for regularization in [1e-3, 1e-1, 1e1]:
            model_1 = FastRidge(fit_intercept=fit_intercept, regularization=regularization)
            model_1.fit(X, y, sample_weight=sample_weight_1)

            model_2 = FastRidge(fit_intercept=fit_intercept, regularization=regularization)
            model_2.fit(X, y, sample_weight=sample_weight_2)

            assert np.allclose(model_1.coef_, model_2.coef_)
            assert np.allclose(model_1.intercept_, model_2.intercept_)


def test_different_sample_weights(data):
    """
    Test the FastRidge model with different sample weights.
    This test checks that the coefficients of the FastRidge model differ
    when fitted with different sample weights. Two sets of sample weights
    are used: one in ascending order and the other in descending order.
    Parameters:
    data (tuple): A tuple containing the feature matrix X and the target vector y.
    Asserts:
    The test asserts that the coefficients of the model fitted with the first
    set of sample weights are not close to the coefficients of the model fitted
    with the second set of sample weights.
    """
    X, y = data
    sample_weight_1 = np.arange(len(X))
    sample_weight_2 = sample_weight_1[::-1]

    for fit_intercept in [True, False]:
        for regularization in [1e-3, 1e-1, 1e1]:
            model_1 = FastRidge(fit_intercept=fit_intercept, regularization=regularization)
            model_1.fit(X, y, sample_weight=sample_weight_1)

            model_2 = FastRidge(fit_intercept=fit_intercept, regularization=regularization)
            model_2.fit(X, y, sample_weight=sample_weight_2)

            assert not np.allclose(model_1.coef_, model_2.coef_)
