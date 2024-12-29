import numpy as np
import pytest
from sklearn.datasets import make_spd_matrix

from timefiller import ImputeMultiVariate, TimeSeriesImputer
from timefiller.utils import add_mar_nan, fetch_pems_bay


@pytest.fixture
def pems_data():
    """Fixture to generate random time series data with missing values."""
    df = fetch_pems_bay().sample(n=35, axis=1)
    df_with_nan = add_mar_nan(df, ratio=0.01)
    return df_with_nan


def test_impute_multivariate():
    n = 50
    mean = np.random.randn(n)
    cov = make_spd_matrix(n)
    X = np.random.multivariate_normal(mean=mean, cov=cov, size=1000)
    X_with_nan = add_mar_nan(X, ratio=0.01)
    imv = ImputeMultiVariate()
    X_imputed = imv(X_with_nan)
    assert np.isnan(X_imputed).sum() == 0


def test_tsi(pems_data):
    """Test imputation on the full time series."""
    tsi = TimeSeriesImputer()

    # Perform imputation on the entire dataset
    df_imputed = tsi(pems_data.sample(n=10, axis=1))

    # Check that there are no more missing values after imputation
    assert df_imputed.isnull().sum().sum() < pems_data.isnull().sum().sum()


def test_tsi_ar_lags(pems_data):
    tsi = TimeSeriesImputer(ar_lags=(1, 2, 3, 6))

    # Define the cutoff date for imputation
    after = '2017-05-01'
    subset_cols = pems_data.sample(n=3, axis=1).columns

    # Perform imputation only after the cutoff date
    df_imputed = tsi(pems_data, after=after, subset_cols=subset_cols)

    # Check that there are no missing values after the cutoff date
    assert df_imputed.isnull().sum().sum() < pems_data.isnull().sum().sum()


def test_tsi_multivariate_lags(pems_data):
    tsi = TimeSeriesImputer(ar_lags=(1, 2, 3, 6), multivariate_lags=12)

    # Define the cutoff date for imputation
    after = '2017-05-01'
    subset_cols = pems_data.sample(n=3, axis=1).columns

    # Perform imputation only after the cutoff date
    df_imputed = tsi(pems_data, after=after, subset_cols=subset_cols)

    # Check that there are no missing values after the cutoff date
    assert df_imputed.isnull().sum().sum() < pems_data.isnull().sum().sum()


def test_impute_ar_lags_auto(pems_data):
    tsi = TimeSeriesImputer(ar_lags='auto')

    # Define the cutoff date for imputation
    after = '2017-05-01'
    subset_cols = pems_data.sample(n=2, axis=1).columns

    # Perform imputation only after the cutoff date
    df_imputed = tsi(pems_data, after=after, subset_cols=subset_cols, n_nearest_covariates=15, preimpute_covariates_limit=1)

    # Check that there are no missing values after the cutoff date
    assert df_imputed.isnull().sum().sum() < pems_data.isnull().sum().sum()
