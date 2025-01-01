.. _whatsnew:

What's New ?
============

Version 1.0 (December 31th 2024)
--------------------------------
- Default estimator to custom implementation of Ridge regression ; no scikit-learn overhead for speed
- ``multivariate_lags`` is 'auto' by default ; it seeks for optimal lags in the covariates during the imputation of each column
- Numerous speed-ups improvements

Version 1.0.10 (December 23th 2024)
---------------------------------
- Improved and faster handling of lag discovery for covariates
- ``PositiveOutout`` has a `column` argument to specify the columns to treat as positive

Version 1.0.9 (December 6th 2024)
---------------------------------
- Major speed-up, sub-arrays creation handled by numba