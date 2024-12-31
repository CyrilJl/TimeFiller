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


def impute_and_assert(tsi, data, after=None, subset_cols=None, **kwargs):
    """Helper function to perform imputation and check results."""
    imputed_data = tsi(data, after=after, subset_cols=subset_cols, **kwargs)
    assert imputed_data.isnull().sum().sum() < data.isnull().sum().sum()
    assert imputed_data.shape == data.shape
    assert imputed_data.index.equals(data.index)
    assert imputed_data.columns.equals(data.columns)
    return imputed_data


def test_impute_multivariate():
    n = 50
    mean = np.random.randn(n)
    cov = make_spd_matrix(n)
    X = np.random.multivariate_normal(mean=mean, cov=cov, size=1000)
    X_with_nan = add_mar_nan(X, ratio=0.01)
    imv = ImputeMultiVariate()
    X_imputed = imv(X_with_nan)
    assert np.isnan(X_imputed).sum() == 0


@pytest.mark.parametrize(
    "ar_lags, multivariate_lags, n_nearest_covariates",
    [((1, 2, 3, 6), None, None), ((1, 2, 3, 6), 12, None), (10, None, 15)],
)
def test_tsi_variants(pems_data, ar_lags, multivariate_lags, n_nearest_covariates):
    tsi = TimeSeriesImputer(ar_lags=ar_lags, multivariate_lags=multivariate_lags)

    after = "2017-05-01"
    subset_cols = pems_data.sample(n=3, axis=1).columns

    # Perform imputation and assert no missing values
    impute_and_assert(tsi, pems_data, after=after, subset_cols=subset_cols, n_nearest_covariates=n_nearest_covariates)


def test_tsi_full(pems_data):
    """Test imputation on the full time series."""
    tsi = TimeSeriesImputer()
    impute_and_assert(tsi, pems_data.sample(n=10, axis=1))


def test_tsi_mapie_uncertainties(pems_data):
    """Test imputation on the full time series."""
    subset_cols = pems_data.sample(n=2, axis=1).columns
    after = "2017-05-01"
    tsi = TimeSeriesImputer(alpha=0.2)
    tsi(pems_data, after=after, subset_cols=subset_cols)
