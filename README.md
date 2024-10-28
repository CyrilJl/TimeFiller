[![PyPI - Version](https://img.shields.io/pypi/v/timefiller)](https://pypi.org/project/timefiller/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/timefiller.svg)](https://anaconda.org/conda-forge/timefiller)
[![Documentation Status](https://readthedocs.org/projects/timefiller/badge/?version=latest)](https://timefiller.readthedocs.io/en/latest/?badge=latest)
[![Unit tests](https://github.com/CyrilJl/timefiller/actions/workflows/pytest.yml/badge.svg)](https://github.com/CyrilJl/timefiller/actions/workflows/pytest.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/51d0dd39565a410985a6836e7d6bcd0b)](https://app.codacy.com/gh/CyrilJl/TimeFiller/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

# <img src="https://raw.githubusercontent.com/CyrilJl/timefiller/main/_static/logo_timefiller.svg" alt="Logo BatchStats" width="200" height="200" align="right"> timefiller

`timefiller` is a Python package designed for time series imputation and forecasting. When applied to a set of correlated time series, each series is processed individually, utilizing both its own auto-regressive patterns and correlations with the other series. The package is user-friendly, making it accessible even to non-experts.

Originally developed for imputation, it also proves useful for forecasting, particularly when covariates contain missing values.

## Installation

You can get ``timefiller`` from PyPi:
```bash
pip install timefiller
```
But also from conda-forge:
```bash
conda install -c conda-forge timefiller
```

```bash
mamba install timefiller
```

## Why this package?

While there are other Python packages for similar tasks, this one is lightweight with a straightforward and simple API. Currently, its speed may be a limitation for large datasets, but it can still be quite useful in many cases.

## Basic Usage

The simplest usage example:

```python
from timefiller import TimeSeriesImputer

df = load_your_dataset()
tsi = TimeSeriesImputer()
df_imputed = tsi(X=df)
```

## Advanced Usage

```python
from sklearn.linear_model import LassoCV
from timefiller import PositiveOutput, TimeSeriesImputer

df = load_your_dataset()
tsi = TimeSeriesImputer(estimator=LassoCV(),
                        ar_lags=(1, 2, 3, 6, 24),
                        multivariate_lags=6,
                        preprocessing=PositiveOutput())
df_imputed = tsi(X=df,
                 subset_cols=['col_1', 'col_17'],
                 after='2024-06-14',
                 n_nearest_features=35)
```

Check out the [documentation](https://timefiller.readthedocs.io/en/latest/index.html) for details on available options to customize your imputation.

## Real data example

Let's evaluate how ``timefiller`` performs on a real-world dataset, the [PeMS-Bay traffic data](https://zenodo.org/records/5724362). A sensor ID is selected for the experiment, and a contiguous block of missing values is introduced. To increase the complexity, additional Missing At Random (MAR) data is simulated, representing 1% of the entire dataset:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from timefiller import TimeSeriesImputer
from timefiller.utils import add_mar_nan, fetch_pems_bay

# Fetch the time series dataset (e.g., PeMS-Bay traffic data)
df = fetch_pems_bay()
dfm = df.copy()  # Create a copy to introduce missing values later

# Randomly select one column (sensor ID) to introduce missing values
k = np.random.randint(df.shape[1])
col = df.columns[k]
i, j = 20_000, 22_500  # Define a range in the dataset to set as NaN (missing values)
dfm.iloc[i:j, k] = np.nan  # Introduce missing values in this range for the selected column

# Add more missing values randomly across the dataset (1% of the data)
dfm = add_mar_nan(dfm, ratio=0.01)

# Initialize the TimeSeriesImputer with AR lags and multivariate lags
tsi = TimeSeriesImputer(ar_lags=48, multivariate_lags=6)

# Apply the imputation method on the modified dataframe
df_imputed = tsi(dfm, subset_cols=col, n_nearest_features=75)

# Plot the imputed data alongside the data with missing values
df_imputed[col].rename('imputation').plot(figsize=(10, 3), lw=0.8, c='C0')
dfm[col].rename('data to impute').plot(ax=plt.gca(), lw=0.8, c='C1')
plt.title(f'sensor_id {col}')
plt.legend()
plt.show()

# Plot the imputed data vs the original complete data for comparison
df_imputed[col].rename('imputation').plot(figsize=(10, 3), lw=0.8, c='C0')
df[col].rename('complete data').plot(ax=plt.gca(), lw=0.8, c='C2')
plt.xlim(dfm.index[i], dfm.index[j])  # Focus on the region where data was missing
plt.legend()
plt.show()
```

<img src="https://raw.githubusercontent.com/CyrilJl/timefiller/main/_static/result_imputation.png" width="750">

## Algorithmic Approach

`timefiller` relies heavily on [scikit-learn](https://scikit-learn.org/stable/) for the learning process and uses [optimask](https://optimask.readthedocs.io/en/latest/index.html) to create NaN-free train and
predict matrices for the estimator.

For each column requiring imputation, the algorithm differentiates between rows with valid data and those with missing values. For rows with missing data, it identifies the available sets of other columns (features).
For each set, OptiMask is called to train the chosen sklearn estimator on the largest possible submatrix without any NaNs. This process can become computationally expensive if the available sets of features vary
greatly or occur infrequently. In such cases, multiple calls to OptiMask and repeated fitting and predicting using the estimator may be necessary.

One important point to keep in mind is that within a single column, two different rows (timestamps) may be imputed using different estimators (regressors), each trained on distinct sets of columns (covariate features)
and samples (rows/timestamps).
