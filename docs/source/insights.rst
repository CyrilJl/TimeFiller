.. _insights:

A Tour of the Available Options
===============================

The core class for time series imputation is ``TimeSeriesImputer``. Several parameters can be adjusted
to meet specific needs.

In the following example, we assume ``df`` is a pandas DataFrame with missing values.

Class Arguments
---------------

``estimator``
~~~~~~~~~~~~~
The machine learning model or algorithm used for imputation can be any model compatible with scikit-learn’s ``fit`` and 
``predict`` methods. By default, this is set to ``timefiller``'s implementation of Ridge regression.

.. code-block:: python

    from sklearn.linear_model import Lasso

    tsi = TimeSeriesImputer(estimator=Lasso(fit_intercept=False))
    df_imputed = tsi(df)

``preprocessing``
~~~~~~~~~~~~~~~~~
A function for preprocessing data before imputation, such as scaling or normalization. It accepts
any scikit-learn transformer with ``fit_transform`` and ``inverse_transform`` methods, allowing for
easy integration of standard preprocessing steps. While preprocessing could be integrated directly
into the pipeline passed to the ``estimator``, doing so would repeatedly apply the transformation
only to submatrices of the dataset. In this approach, ``fit_transform`` is applied to the entire
dataset before imputation, and ``inverse_transform`` is used afterward to revert the data to its
original space.

.. code-block:: python

    from sklearn.preprocessing import PowerTransformer

    tsi = TimeSeriesImputer(preprocessing=PowerTransformer(method='box-cox'))
    df_imputed = tsi(df)

When imputing a set of positive time series, ``timefiller`` provides a useful tool, ``PositiveOutput``:

.. code-block:: python

    from timefiller import PositiveOutput

    tsi = TimeSeriesImputer(ar_lags=24, preprocessing=PositiveOutput())
    df_imputed = tsi(df)

While specifying ``estimator=LinearRegression(positive=True, fit_intercept=False)`` can enforce positive imputed values,
this approach may be less effective when using autoregressive lags. This is because it prevents the model from assigning
negative weights to lagged series, which could otherwise help create differential-like features. The ``PositiveOutput``
strategy, inspired by transformations like Box-Cox or Yeo-Johnson, addresses this by expanding values near zero into the
negative domain before fitting the model, and then applying the inverse transformation after prediction. This functions
as a softened ReLU, rather than working with the original data and forcing negative predictions to zero (as in a hard ReLU).
The threshold for "near-zero" values can be controlled by the user: ``PositiveOutput(q=10)`` sets the threshold at the 10th
percentile of each time series, while ``PositiveOutput(v=25.)`` sets a fixed threshold of 25 for all time series. 

``ar_lags``
~~~~~~~~~~~
Specifies the autoregressive lags to be considered during imputation. Lags are defined relative to
the ``freq`` of the time index in the DataFrame.

.. code-block:: python

    # For each time series, the lags -6, -3, -2, -1, +1, +2, +3, +6 will be used
    # as input data, making the imputation autoregressive, in addition to the covariates
    tsi = TimeSeriesImputer(ar_lags=(1, 2, 3, 6))
    df_imputed = tsi(df)

.. code-block:: python

    # Lags are automatically inferred as -3, -2, -1, +1, +2, +3
    tsi = TimeSeriesImputer(ar_lags=3)
    df_imputed = tsi(df)

Large values of ``ar_lags`` are not advised, as it leads to many features and may slow down the imputation process.
Instead of ``ar_lags=24``, consider using ``ar_lags=(1, 2, 3, 4, 5, 6, 12, 24)``, which leads to 3 times fewer features,
while leading to similar results.

``multivariate_lags``
~~~~~~~~~~~~~~~~~~~~~
``timefiller`` uses other time series to help impute missing values in a given series. However, sometimes
these series are more informative when lagged. ``multivariate_lags`` allows the model to search for the
best lag within the specified range. If it is specified as an integer, the optimal lags are searched into
[-multivariate_lags, multivariate_lags].  It defaults to 'auto' : the optimal lags are searched in [-n, n],
where n = min(50, 2% of the time series length).

.. code-block:: python

    # Covariates can be lagged as well
    tsi = TimeSeriesImputer(ar_lags=24, multivariate_lags=6)
    df_imputed = tsi(df)

``na_frac_max``
~~~~~~~~~~~~~~~
The maximum allowed fraction of missing values for imputation to proceed. This helps ensure data quality.

.. code-block:: python

    tsi = TimeSeriesImputer(ar_lags=24, multivariate_lags=6, na_frac_max=0.25)
    # Columns with more than 25% missing values will NOT be imputed
    df_imputed = tsi(df)


``__call__`` Arguments
----------------------

These arguments provide options for speeding up the process:

``subset_cols``
~~~~~~~~~~~~~~~
Specifies the columns to impute. By default, all columns are imputed (within the ``na_frac_max`` limit).
In the case you don't need to impute all columns, it is recommended to specify the columns to impute,
as imputing all columns may be computationally expensive.

.. code-block:: python

    tsi = TimeSeriesImputer()
    df_imputed = tsi(df, subset_cols=['col1', 'col2'])

``before`` and ``after``
~~~~~~~~~~~~~~~~~~~~~~~~
In some cases, imputation may only be needed for data within a certain time range. It is then recommended
to specify the time range to be imputed, as imputing all data may be computationally expensive.

.. code-block:: python

    tsi = TimeSeriesImputer()
    df_imputed = tsi(df, subset_cols=['col1', 'col2'], after='2024-01-01')

.. code-block:: python

    tsi = TimeSeriesImputer()
    df_imputed = tsi(df, subset_cols=['col1', 'col2'], after='2024-01-01', before='2024-01-31')

``n_nearest_covariates``
~~~~~~~~~~~~~~~~~~~~~~~~
To speed up the imputation process, you can perform variable selection before running the imputation, which is especially
useful for datasets with a large number of covariates. This method samples features based on their correlation with the
feature being processed, as well as the number of common valid samples. The selection is randomized to provide a diverse
set of features, and is not simply the n most correlated covariates. Defaults to 35. If None, all covariates are used, which
can be computationally expensive.

.. code-block:: python

    tsi = TimeSeriesImputer()
    %time df_imputed = tsi(df)
    
    tsi = TimeSeriesImputer()
    %time df_imputed = tsi(df, n_nearest_features=50)
