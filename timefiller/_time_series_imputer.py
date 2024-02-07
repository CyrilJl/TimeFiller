# -*- coding: utf-8 -*-

# author : Cyril Joly

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm.auto import tqdm

from ._misc import check_params
from ._multivariate_imputer import ImputeMultiVariate


class TimeSeriesImputer:
    def __init__(self, estimator: str = 'positive with intercept', ar_lags=None, multivariate_lags=None,
                 na_frac_max: float = 0.33, min_samples_train: int = 50, optimask_n_tries: int = 10,
                 verbose: bool = False, random_state: int or None = None):
        """
        Initialize the TimeSeriesImputer object.

        Parameters:
        - estimator (str or object): Estimator for imputation. Defaults to 'positive with intercept'.
        - ar_lags (int, list, np.ndarray, tuple, None): Autoregressive lags for imputation.
        - multivariate_lags (int or None): Number of multivariate lags for imputation.
        - na_frac_max (float): Maximum fraction of missing values allowed for imputation.
        - min_samples_train (int): Minimum number of samples required for training the estimator.
        - optimask_n_tries (int): Number of tries for the OptiMask solver.
        - verbose (bool): Verbosity level.
        - random_state (int or None): Random seed for reproducibility.
        """
        self.imputer = ImputeMultiVariate(estimator=estimator, na_frac_max=na_frac_max, min_samples_train=min_samples_train, optimask_n_tries=optimask_n_tries, verbose=verbose)
        self.ar_lags = self._process_lags(ar_lags)
        self.multivariate_lags = check_params(multivariate_lags, types=(int, type(None)))
        self.verbose = bool(verbose)
        self.random_state = random_state

    @staticmethod
    def _process_lags(ar_lags):
        """
        Process autoregressive lags.

        Parameters:
        - ar_lags (int, list, np.ndarray, tuple, None): Autoregressive lags.

        Returns:
        - list or None: Processed autoregressive lags.
        """
        check_params(ar_lags, types=(int, list, np.ndarray, tuple, type(None)))
        if ar_lags is None:
            return None
        if isinstance(ar_lags, int):
            ar_lags = list(range(-abs(ar_lags)-1, -1)) + list(range(1, abs(ar_lags)+1))
            return sorted(ar_lags)
        if isinstance(ar_lags, (tuple, list, np.ndarray)):
            ar_lags = [-k for k in ar_lags if k != 0] + [k for k in ar_lags if k != 0]
            return sorted(list(set(ar_lags)))

    @staticmethod
    def _sample_features(corr, common_samples, col, n_nearest_features, rng):
        """
        Sample features based on correlation and common samples.

        Parameters:
        - corr: Correlation matrix.
        - common_samples: Matrix of common samples.
        - col: Column index.
        - n_nearest_features: Number of nearest features to sample.
        - rng: Random number generator.

        Returns:
        - list: Sampled features.
        """
        s1 = corr[col].drop(col)
        s2 = common_samples[col].drop(col)
        p = abs(s1.values) * s2.values
        size = min(n_nearest_features, len(s1), len(p[p > 0]))
        return list(rng.choice(a=s1.index, size=size, p=p/p.sum(), replace=False))

    @staticmethod
    def _compute_features_selection_data(X):
        corr = X.fillna(X.mean()).corr()
        x = (~X.isnull()).astype(float).values
        common_samples = (x.T@x)/len(x)
        common_samples = pd.DataFrame(common_samples, index=X.columns, columns=X.columns)
        return corr, common_samples

    @staticmethod
    def find_best_lags(x, col, max_lags):
        """
        Find the best lags for imputation.

        Parameters:
        - x: Input data.
        - col: Column index.
        - max_lags: Maximum number of lags.

        Returns:
        - pd.DataFrame: DataFrame with imputed lags.
        """
        df = x.fillna(x.mean())
        cols = df.drop(columns=col).columns
        ret = [x[col]]
        for other_col in cols:
            lag = sm.tsa.ccf(df[col], df[other_col], nlags=max_lags).argmax()
            ret.append(x[other_col])
            if lag > 0:
                ret.append(x[other_col].shift(lag).rename(f"{other_col}+{lag}"))
        return pd.concat(ret, axis=1)

    @staticmethod
    def _process_subset_cols(X, subset_cols):
        _, n = X.shape
        columns = list(X.columns)
        if subset_cols is None:
            return list(range(n))
        if isinstance(subset_cols, str):
            if subset_cols in columns:
                return [columns.index(subset_cols)]
            else:
                return []
        if isinstance(subset_cols, (list, tuple, pd.core.indexes.base.Index)):
            return [columns.index(_) for _ in subset_cols if _ in columns]
        raise TypeError()

    @staticmethod
    def _process_subset_rows(X, before, after):
        index = pd.Series(np.arange(len(X)), index=X.index)
        if before is not None:
            index = index[pd.to_datetime(str(before)) <= index.index]
        if after is not None:
            index = index[pd.to_datetime(str(after)) <= index.index]
        return list(index.values)

    def _impute_col(self, x, col, subset_rows):
        """
        Impute missing values for a specific column.

        Parameters:
        - x: Input data.
        - col: Column index.
        - subset_rows: Subset of rows.

        Returns:
        - pd.Series: Series with imputed values for the specified column.
        """
        if isinstance(self.multivariate_lags, int):
            x = self.find_best_lags(x, col, self.multivariate_lags)
        if self.ar_lags is not None:
            for k in sorted(self.ar_lags):
                x[f"{col}{k:+d}"] = x[col].shift(k).copy()
        index_col = list(x.columns).index(col)
        x_col_imputed = self.imputer(x.values, subset_rows=subset_rows, subset_cols=index_col)[:, index_col]
        return pd.Series(x_col_imputed, name=col, index=x.index)

    def __call__(self, X, subset_cols=None, before=None, after=None, n_nearest_features=None) -> pd.DataFrame:
        """
        Perform time series data imputation.

        Parameters:
        - X: Input data.
        - subset_cols: Subset of columns to consider.
        - before: Timestamp for filtering rows before (and up to) a certain time. Imputation is performed only for timestamps
        before or at this specified time.
        - after: Timestamp for filtering rows after (and starting from) a certain time. Imputation is performed only for
        timestamps after or at this specified time.
        - n_nearest_features: Number of nearest features for feature selection.

        Returns:
        - pd.DataFrame: DataFrame with imputed values.

        Note:
        If the DataFrame is not sampled equally up to a given frequency, it is resampled to a guessed frequency. This
        operation may create potentially missing values (NaNs) in rows, and it is important if autoregressive lags are provided.

        The nearest feature selection for imputation is performed using a random sampling based on correlation and the number
        of common valid (non NaN) timestamps they share pairwise. This ensures a robust selection of features for imputation.
        """
        rng = np.random.default_rng(self.random_state)
        X_ = check_params(X, types=pd.DataFrame).copy()
        check_params(X_.index, types=pd.DatetimeIndex)

        if X_.index.freq is None:
            X_ = X_.asfreq(pd.infer_freq(X_.index))
        X_ = X_[X_.columns[X_.std() > 0]].copy()
        columns = list(X_.columns)

        if isinstance(n_nearest_features, int):
            corr, common_samples = self._compute_features_selection_data(X_)

        ret = []
        subset_rows = self._process_subset_rows(X_, before, after)
        subset_cols = self._process_subset_cols(X_, subset_cols)
        for index_col in tqdm(subset_cols, disable=(not self.verbose)):
            col = columns[index_col]
            if isinstance(n_nearest_features, int):
                cols_in = [col] + self._sample_features(corr, common_samples, col, n_nearest_features, rng)
            else:
                cols_in = list(X_.columns)
            ret.append(self._impute_col(x=X_[cols_in], col=col, subset_rows=subset_rows))
        ret = pd.concat(ret, axis=1).reindex_like(X).combine_first(X)
        return ret
