.. _index:

.. image:: https://img.shields.io/pypi/v/timefiller
   :alt: PyPI - Version

.. toctree::
   :hidden:
   :maxdepth: 1   

   insights
   api_reference

timefiller
==========

``timefiller`` is a Python package for time series imputation and forecasting. When applied to a set of correlated time series, each series is processed individually, leveraging correlations with the other series as well as its own auto-regressive patterns. The package is designed to be easy to use, even for non-experts.

Installation
------------

.. code-block:: console

   pip install timefiller

Why this package?
-----------------

While there are other Python packages for similar tasks, this one is lightweight with a straightforward and simple API. Currently, its speed may be a limitation for large datasets, but it can still be quite useful in many cases.

Basic Usage
-----------

The simplest usage example:

.. code-block:: python

   from timefiller import TimeSeriesImputer

   df = load_your_dataset()
   tsi = TimeSeriesImputer()
   df_imputed = tsi(df)

Advanced Usage
--------------

.. code-block:: python

   from timefiller import TimeSeriesImputer, PositiveOutput
   from sklearn.linear_model import LassoCV

   df = load_your_dataset()
   tsi = TimeSeriesImputer(estimator=LassoCV(), ar_lags=(1, 2, 3, 6, 24), multivariate_lags=6, preprocessing=PositiveOutput())
   df_imputed = tsi(df, subset_cols=['col_1', 'col_17'], after='2024-06-14', n_nearest_features=35)

Check out :ref:`insights` for details on available options to customize your imputation.

Algorithmic Approach
--------------------

``timefiller`` relies heavily on `scikit-learn <https://scikit-learn.org/stable/>`_ for the learning process and uses
`optimask <https://optimask.readthedocs.io/en/latest/index.html>`_ to create NaN-free train and predict matrices for the
estimator.