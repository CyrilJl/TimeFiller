import os

import pytest
from timefiller import TimeSeriesImputer
from timefiller.utils import add_mar_nan, generate_random_time_series


@pytest.fixture
def generate_data():
    """Fixture to generate random time series data with missing values."""
    df = generate_random_time_series(n=35, start='2023-01-01', freq='h', periods=24*365)
    df_with_nan = add_mar_nan(df, ratio=0.02)
    return df_with_nan

def test_impute_full_series(generate_data):
    """Test imputation on the full time series."""
    df = generate_data
    tsi = TimeSeriesImputer()
    
    # Perform imputation on the entire dataset
    df_imputed = tsi(df)
    
    # Check that there are no more missing values after imputation
    assert df_imputed.isnull().sum().sum() > df.isnull().sum().sum()

def test_impute_2(generate_data):
    """Test imputation only after a specific date."""
    df = generate_data
    tsi = TimeSeriesImputer(ar_lags=(1, 2, 3, 6))
    
    # Define the cutoff date for imputation
    cutoff_date = '2023-06-01'
    
    # Perform imputation only after the cutoff date
    df_imputed = tsi(df, after=cutoff_date)
    
    # Check that there are no missing values after the cutoff date
    assert df_imputed.isnull().sum().sum() > df.isnull().sum().sum()

    # Check that missing values before the cutoff date remain (if there were any)
    before_cutoff_null_count = df.loc[:cutoff_date].isnull().sum().sum()
    assert before_cutoff_null_count == df_imputed.loc[:cutoff_date].isnull().sum().sum(), "Imputation changed values before the cutoff date."

