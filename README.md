<img src="https://raw.githubusercontent.com/CyrilJl/timefiller/main/_static/logo_timefiller.svg" alt="Logo BatchStats" width="50" height="50">

# timefiller

The `timefiller` Python package offers a robust solution for imputing missing data in time series, utilizing both statistical and machine learning methods. Designed for flexibility and customization, it can handle various types of missing data patterns across multiple correlated time series.

## Overview

`timefiller` is currently in development, aiming to provide advanced imputation and forecasting capabilities for time series data, particularly when multiple correlated series are involved. The package is built to accommodate missing data in a variety of formats and configurations. `timefiller` also provides forecasting capabilities when future covariates are available.

It will be uploaded to PyPI soon, and details about the algorithm will be presented.

### Usage Example

Below is an example of how to use `timefiller` for imputing missing values in a time series dataset:

```python
from timefiller import TimeSeriesImputer

# Assuming df is your DataFrame with time series data that may contain missing values
df = ...

# Initialize the TimeSeriesImputer with specified parameters
tsi = TimeSeriesImputer(ar_lags=6)

# Perform the imputation
df_imputed = tsi(df, n_nearest_features=50)

# df_imputed now contains the DataFrame with imputed values
```

### Key Features

1. **Time Series Imputation**: 
   - Efficiently fill missing values in time series data.
   - Support for handling multiple time series simultaneously, each potentially having different missing data patterns.

2. **Flexible Configuration**: 
   - Customize the imputation process with various parameters, including the number of autoregressive lags, nearest features, and more.

3. **Selective Imputation and Forecasting**: 
   - Imputation and forecasts can be limited to specific columns and time ranges, optimizing computational resources and allowing targeted data processing.

### Dependencies

- [scikit-learn](https://scikit-learn.org/stable/index.html): Used extensively for the `estimator` parameter, enabling compatibility with a wide array of machine learning models.
- [optimask](https://optimask.readthedocs.io/en/latest/index.html): Utilized for optimizing the imputation mask, ensuring efficient handling of missing data patterns.

### Performance Considerations

Current versions of `timefiller` might exhibit slower performance on large datasets due to the computational complexity of imputation methods. Efforts are underway to enhance both performance and scalability, making it suitable for bigger datasets.

### TimeSeriesImputer Class Parameters

The `TimeSeriesImputer` class is the core component of the `timefiller` package. It allows for detailed customization of the imputation process:

- **`estimator`**: (object, optional)  
  The machine learning model or algorithm to use for imputation. Any model compatible with `fit` and `predict` from scikit-learn can be used.

- **`preprocessing`**: (callable, optional)  
  A function for preprocessing the data before imputation, such as scaling or normalization. It accepts any scikit-learn transformer that has `fit_transform` and `inverse_transform` methods, allowing for easy integration of standard data preprocessing steps.

- **`ar_lags`**: (int, list, numpy.ndarray, or tuple, optional)  
  Defines the autoregressive lags to consider in imputation:
  - Integer: Number of lags to include.
  - Iterable of ints: Specific lags to use.

- **`multivariate_lags`**: (int or None, optional)  
  Number of multivariate lags to consider, useful when dealing with multiple correlated time series.

- **`na_frac_max`**: (float, optional)  
  Maximum allowed fraction of missing values for the imputation to proceed. Helps maintain data quality.

- **`min_samples_train`**: (int, optional)  
  Minimum number of samples required to train the imputation model.

- **`weighting_func`**: (callable, optional)  
  Custom function for weighting data points during imputation, allowing more recent or relevant data to have a greater impact.

- **`optimask_n_tries`**: (int, optional)  
  Number of optimization attempts for the missing data mask, improving imputation accuracy.

- **`verbose`**: (bool, optional)  
  If set to `True`, provides detailed progress output during imputation.

- **`random_state`**: (int or None, optional)  
  Seed for random number generation, ensuring reproducible results.

### Imputation and Forecasting on Specific Columns and Time Ranges

The `TimeSeriesImputer` class is designed to allow imputation and forecasting on specific subsets of columns and within specified time ranges. This feature is useful for targeting only the most critical parts of the dataset or reducing computational load.

#### Example Usage of Selective Imputation

Here’s an example demonstrating the selective imputation capabilities:

```python
from timefiller import TimeSeriesImputer
from sklearn.linear_model import LinearRegression

# Example DataFrame with time series data
df = ...

# Configure TimeSeriesImputer with custom parameters
tsi = TimeSeriesImputer(
    estimator=LinearRegression(fit_intercept=False),
    preprocessing=None,
    ar_lags=6,
    multivariate_lags=None,
    na_frac_max=0.2,
    min_samples_train=100,
    weighting_func=None,
    optimask_n_tries=5,
    verbose=True,
    random_state=42
)

# Perform imputation on a specific subset of columns and time range
df_imputed = tsi(
    df, 
    subset_cols=['column1', 'column2'],
    before='2024-01-15',
    after='2020-01-01',
    n_nearest_features=50
)

# df_imputed now contains the imputed DataFrame
```
