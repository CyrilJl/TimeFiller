[![PyPI - Version](https://img.shields.io/pypi/v/timefiller)](https://pypi.org/project/timefiller/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/timefiller.svg)](https://anaconda.org/conda-forge/timefiller)
[![Documentation Status](https://readthedocs.org/projects/timefiller/badge/?version=latest)](https://timefiller.readthedocs.io/en/latest/?badge=latest)
[![Unit tests](https://github.com/CyrilJl/timefiller/actions/workflows/pytest.yml/badge.svg)](https://github.com/CyrilJl/timefiller/actions/workflows/pytest.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/51d0dd39565a410985a6836e7d6bcd0b)](https://app.codacy.com/gh/CyrilJl/TimeFiller/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

# <img src="https://raw.githubusercontent.com/CyrilJl/timefiller/main/_static/logo_timefiller.svg" alt="Logo BatchStats" width="200" height="200" align="right"> timefiller

`timefiller` is a Python package for time series imputation and forecasting. When applied to a set of correlated time series, each series is processed individually, leveraging correlations with the other series as well as its own auto-regressive patterns. The package is designed to be easy to use, even for non-experts.

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
df_imputed = tsi(df)
```

## Advanced Usage

```python
from sklearn.linear_model import LassoCV
from timefiller import TimeSeriesImputer, PositiveOutput

df = load_your_dataset()
tsi = TimeSeriesImputer(estimator=LassoCV(), ar_lags=(1, 2, 3, 6, 24), multivariate_lags=6, preprocessing=PositiveOutput())
df_imputed = tsi(df, subset_cols=['col_1', 'col_17'], after='2024-06-14', n_nearest_features=35)
```

Check out the [documentation](https://timefiller.readthedocs.io/en/latest/index.html) for details on available options to customize your imputation.

## Algorithmic Approach

`timefiller` relies heavily on [scikit-learn](https://scikit-learn.org/stable/) for the learning process and uses [optimask](https://optimask.readthedocs.io/en/latest/index.html) to create NaN-free train and predict matrices for the estimator.
