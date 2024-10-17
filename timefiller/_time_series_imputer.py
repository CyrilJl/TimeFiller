import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.feature_selection import r_regression
from tqdm.auto import tqdm

from ._misc import check_params
from ._multivariate_imputer import ImputeMultiVariate


class TimeSeriesImputer:
    """Class for time series imputation.

    Args:
        estimator (object, optional): Estimator used for imputation.
        preprocessing (callable, optional): Data preprocessing.
        ar_lags (int, list, numpy.ndarray, or tuple, optional): Autoregressive lags to consider.
        multivariate_lags (int or None, optional): Multivariate lags to consider.
        na_frac_max (float, optional): Maximum fraction of missing values allowed.
        min_samples_train (int, optional): Minimum number of samples for training.
        weighting_func (callable, optional): Weighting function for imputation.
        optimask_n_tries (int, optional): Number of attempts for optimization.
        verbose (bool, optional): Display process details.
        random_state (int or None, optional): Random state for reproducibility.
    """

    def __init__(self, estimator=None, preprocessing=None, ar_lags=None, multivariate_lags=None, na_frac_max=0.33,
                 min_samples_train=50, weighting_func=None, optimask_n_tries=1, verbose=False, random_state=None):

        self.imputer = ImputeMultiVariate(estimator=estimator, preprocessing=preprocessing,
                                          na_frac_max=na_frac_max, min_samples_train=min_samples_train,
                                          weighting_func=weighting_func, optimask_n_tries=optimask_n_tries,
                                          verbose=verbose)
        self.ar_lags = self._process_lags(ar_lags)
        self.multivariate_lags = check_params(multivariate_lags, types=(int, type(None)))
        self.verbose = bool(verbose)
        self.random_state = random_state

    def __repr__(self):
        params = ", ".join(f"{k}={getattr(self, k)}" for k in ('ar_lags', 'multivariate_lags'))
        return f"TimeSeriesImputer({params})"

    @staticmethod
    def _process_lags(ar_lags):
        check_params(ar_lags, types=(int, list, np.ndarray, tuple, type(None)))
        if ar_lags is None:
            return None
        if isinstance(ar_lags, int):
            ar_lags = list(range(-abs(ar_lags), 0)) + list(range(1, abs(ar_lags)+1))
            return tuple(sorted(ar_lags))
        if isinstance(ar_lags, (tuple, list, np.ndarray)):
            ar_lags = [-k for k in ar_lags if k != 0] + [k for k in ar_lags if k != 0]
            return tuple(sorted(set(ar_lags)))

    @staticmethod
    def _sample_features(data, col, n_nearest_features, rng):
        x = data.fillna(data.mean())
        # computes pearson correlation between col and others series
        s1 = r_regression(X=x.drop(columns=col), y=x[col])
        # computes the mean number of timestamps containing common valid data between col and others series
        s2 = ((~data[col].isnull()).astype(float).values@(~data.drop(columns=col).isnull()).astype(float).values)/len(data)
        # features are sampled according those computed features
        p = np.sqrt(abs(s1) * s2)
        size = min(n_nearest_features, len(s1), len(p[p > 0]))
        cols_to_sample = list(data.drop(columns=col).columns)
        return list(rng.choice(a=cols_to_sample, size=size, p=p/p.sum(), replace=False))

    @staticmethod
    def _best_lag(s1, s2, max_lags):
        c1 = sm.tsa.ccf(s1, s2, nlags=max_lags)[::-1]
        c2 = sm.tsa.ccf(s2, s1, nlags=max_lags)[1:]
        c = np.concatenate([c1, c2])
        return np.abs(c).argmax() - max_lags + 1

    @classmethod
    def find_best_lags(cls, x, col, max_lags):
        df = x.fillna(x.mean())
        cols = df.drop(columns=col).columns
        ret = [x[col]]
        for other_col in cols:
            lag = cls._best_lag(df[col], df[other_col], max_lags=max_lags)
            if lag != 0:
                ret.append(x[other_col].shift(-lag).rename(f"{other_col}{-lag:+d}"))
            else:
                ret.append(x[other_col])
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
        raise TypeError(f"subset_cols should be of type str, list, tuple, or pandas Index. Received type {type(subset_cols)} instead.")

    @staticmethod
    def _process_subset_rows(X, before, after):
        index = pd.Series(np.arange(len(X)), index=X.index)
        if before is not None:
            index = index[index.index <= pd.to_datetime(str(before))]
        if after is not None:
            index = index[pd.to_datetime(str(after)) <= index.index]
        return list(index.values)

    def _impute_col(self, x, col, subset_rows):
        if isinstance(self.multivariate_lags, int):
            x = self.find_best_lags(x, col, self.multivariate_lags)
        x = x.copy()
        if self.ar_lags is not None:
            for k in sorted(self.ar_lags):
                x[f"{col}{k:+d}"] = x[col].shift(k).copy()
        index_col = list(x.columns).index(col)
        x_col_imputed = self.imputer(x.values, subset_rows=subset_rows, subset_cols=index_col)[:, index_col]
        return pd.Series(x_col_imputed, name=col, index=x.index)

    def __call__(self, X, subset_cols=None, before=None, after=None, n_nearest_features=None) -> pd.DataFrame:
        """Call method for imputation.

        Args:
            X (DataFrame): Data to be imputed.
            subset_cols (str, list, tuple, or pandas.core.indexes.base.Index, optional): Columns to be imputed. By default, all columns will be imputed.
            before (str or pd.Timestamp or None, optional): Date before which the data is imputed. By default, no lower temporal limit is set.
            after (str or pd.Timestamp or None, optional): Date after which the data is imputed. By default, no upper temporal limit is set.
            n_nearest_features (int, optional): Number of nearest features to consider. A heuristic is used: the features are selected randomly, based on
            their correlations with the feature to be imputed, as well as the number of common temporal observations with the feature to be imputed.

        Returns:
            DataFrame or tuple: Imputed data.
        """

        rng = np.random.default_rng(self.random_state)
        X_ = check_params(X, types=pd.DataFrame).copy()
        check_params(X_.index, types=pd.DatetimeIndex)

        if X_.index.freq is None:
            X_ = X_.asfreq(pd.infer_freq(X_.index))
        X_ = X_[X_.columns[X_.std() > 0]].copy()
        columns = list(X_.columns)

        ret = [pd.Series(index=X.index)]
        subset_rows = self._process_subset_rows(X_, before, after)
        subset_cols = self._process_subset_cols(X_, subset_cols)
        for index_col in tqdm(subset_cols, disable=(not self.verbose)):
            col = columns[index_col]
            if isinstance(n_nearest_features, int):
                cols_in = [col] + self._sample_features(X_, col, n_nearest_features, rng)
            else:
                cols_in = list(X_.columns)
            ret.append(self._impute_col(x=X_[cols_in], col=col, subset_rows=subset_rows))
        ret = pd.concat(ret, axis=1).reindex_like(X).combine_first(X)
        return ret
