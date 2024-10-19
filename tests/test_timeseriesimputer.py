import pytest
from sklearn.linear_model import LassoCV
from timefiller import TimeSeriesImputer
from timefiller.utils import add_mar_nan, fetch_pems_bay


@pytest.fixture
def generate_data():
    """Fixture to generate random time series data with missing values."""
    df = fetch_pems_bay().sample(n=35, axis=1)
    df_with_nan = add_mar_nan(df, ratio=0.01)
    return df_with_nan

def test_impute_1(generate_data):
    """Test imputation on the full time series."""
    df = generate_data
    tsi = TimeSeriesImputer()
    
    # Perform imputation on the entire dataset
    df_imputed = tsi(df)
    
    # Check that there are no more missing values after imputation
    assert df_imputed.isnull().sum().sum() < df.isnull().sum().sum()

def test_impute_2(generate_data):
    """Test imputation only after a specific date."""
    df = generate_data
    tsi = TimeSeriesImputer(ar_lags=(1, 2, 3, 6))
    
    # Define the cutoff date for imputation
    after = '2017-05-01'
    
    # Perform imputation only after the cutoff date
    df_imputed = tsi(df, after=after)
    
    # Check that there are no missing values after the cutoff date
    assert df_imputed.isnull().sum().sum() < df.isnull().sum().sum()

def test_impute_3(generate_data):
    """Test imputation only after a specific date."""
    df = generate_data
    tsi = TimeSeriesImputer(estimator=LassoCV(), ar_lags=(1, 2, 3, 6), multivariate_lags=12)
    
    # Define the cutoff date for imputation
    after = '2017-05-01'
    
    # Perform imputation only after the cutoff date
    df_imputed = tsi(df, after=after, subset_cols='serie 5', n_nearest_features=15)
    
    # Check that there are no missing values after the cutoff date
    assert df_imputed.isnull().sum().sum() < df.isnull().sum().sum()

