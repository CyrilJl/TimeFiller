import numpy as np
import pandas as pd
from numba import njit, prange
from sklearn.pipeline import make_pipeline
from tqdm.auto import tqdm

from ._misc import check_params
from ._multivariate_imputer import ImputeMultiVariate


class TimeSeriesImputer:
    """
    Imputation of multivariate time series.

    This class extends the capabilities of `ImputeMultiVariate` by accounting for
    autoregressive and multivariate lags, as well as preprocessing.
    """

    def __init__(
        self,
        estimator=None,
        alpha=None,
        preprocessing=None,
        ar_lags=None,
        multivariate_lags="auto",
        na_frac_max=0.33,
        min_samples_train=50,
        weighting_func=None,
        optimask_n_tries=1,
        negative_ar=False,
        random_state=None,
        verbose=0,
    ):
        """
        Initializes the imputer for time series.

        Args:
            estimator (object, optional): Estimator for regression.
            alpha (float or list, optional): Quantile level(s) for prediction intervals,
                MAPIE style: if set to 0.1, 5% and 95% confidence bounds are computed. Under the
                hood, the provided estimator is passed to MapieRegressor.
                Default is None.
            preprocessing (object, optional): Preprocessing step (e.g., StandardScaler).
                Default is StandardScaler(with_mean=False).
            ar_lags (int, list, optional): Autoregressive lags to consider.
                If int, generates lags from -ar_lags to +ar_lags (excluding 0).
                Default is None.
            multivariate_lags (int or str, optional): If specified, the optimal positive
                and negative lags are determined for each column based on correlation.
                These optimal lags are recalculated for each target column. If 'auto',
                optimal lags are searched within 2% of the series length. If an integer,
                optimal lags are searched within the range [-multivariate_lags, multivariate_lags].
                Default is 'auto'.
            na_frac_max (float, optional): Maximum fraction of missing values per column.
                Default is 0.33.
            min_samples_train (int, optional): Minimum number of training samples.
                Default is 50.
            weighting_func (callable, optional): Function to weight samples.
                Default is None.
            optimask_n_tries (int, optional): Number of attempts for OptiMask.
                Default is 1.
            negative_ar (bool, optional): If True, includes negative versions of
                autoregressive lags. This can be useful when combined with a linear model
                that has positive coefficients, as it enforces positive coefficients for
                the covariates without sign constraints on the autoregressive features.
                Default is False.
            random_state (int, optional): Random seed. Default is None.
            verbose (int or bool, optional): Verbosity level. Default is 0.
        """
        self.imputer = ImputeMultiVariate(
            estimator=estimator,
            alpha=alpha,
            na_frac_max=na_frac_max,
            min_samples_train=min_samples_train,
            weighting_func=weighting_func,
            optimask_n_tries=optimask_n_tries,
            verbose=verbose,
        )
        self.preprocessing = self._get_preprocessing(preprocessing)
        self.ar_lags = self._get_ar_lags(ar_lags)
        self.multivariate_lags = check_params(multivariate_lags, types=(int, str, type(None)))
        self.negative_ar = negative_ar
        self.verbose = verbose
        self.random_state = random_state

    def __repr__(self):
        return f"TimeSeriesImputer(ar_lags={self.ar_lags}, multivariate_lags={self.multivariate_lags})"

    @staticmethod
    def _get_preprocessing(preprocessing):
        """
        Returns the preprocessing as a pipeline if necessary.

        Args:
            preprocessing (object, None, tuple, list): Preprocessing.

        Returns:
            object: Preprocessing or default pipeline.
        """
        if preprocessing is None:
            return None
        elif isinstance(preprocessing, (tuple, list)):
            return make_pipeline(preprocessing)
        else:
            return preprocessing

    @staticmethod
    def _get_ar_lags(ar_lags):
        """
        Processes autoregressive lags to ensure they are consistent (sorted list without 0).

        Args:
            ar_lags (int, list, tuple, np.ndarray, None): Autoregressive lags.

        Returns:
            list or None: List of lags or None if not specified.
        """
        if ar_lags == "auto":
            return "auto"
        if isinstance(ar_lags, int):
            return list(range(-abs(ar_lags), 0)) + list(range(1, abs(ar_lags) + 1))
        if isinstance(ar_lags, (list, tuple, np.ndarray)):
            return sorted(sum([[-k, k] for k in ar_lags if k != 0], []))
        return None

    @staticmethod
    def _sample_features(data, col, n_nearest_covariates, rng):
        """
        Randomly selects the most relevant columns for imputation
        based on correlation and data availability.

        Args:
            data (pd.DataFrame): Data.
            col (str): Target column.
            n_nearest_covariates (int): Number of features to select.
            rng (np.random.Generator): Random number generator.

        Returns:
            list: List of selected columns.
        """
        data_col = data[col]
        data_others = data.drop(columns=col)
        s1 = data_others.corrwith(data_col).values
        s2 = ((~data_col.isnull()).astype(float).values @ (~data_others.isnull()).astype(float).values) / len(data)
        p = np.sqrt(abs(s1) * s2)
        p[~np.isfinite(p)] = 0
        size = min(n_nearest_covariates, len(s1), len(p[p > 0]))
        cols_to_sample = [_ for _ in data.columns if _ != col]
        if cols_to_sample and size:
            return list(rng.choice(a=cols_to_sample, size=size, p=p / p.sum(), replace=False))
        else:
            return []

    @staticmethod
    @njit(parallel=True, boundscheck=False)
    def cross_correlation(s1, s2, max_lags):
        """
        Computes cross-correlation between two series with lags.
        Equivalent to [pd.Series(s1).corr(pd.Series(s2.shift(lag))) for lag in range(-max_lags, max_lags+1)],
        but much faster: NaNs are taken into account, and the function uses Numba plus the Welford algorithm.
        Also slightly faster than statsmodels.tsa.stattools.ccf, and prevent from depending on statsmodels
        Args:
            s1 (array): First numpy array.
            s2 (array): Second numpy array.
            max_lags (int): Maximum lag to compute.

        Returns:
            tuple: Lags and cross-correlation values.
        """
        n = len(s1)
        cross_corr = np.empty(2 * max_lags + 1, dtype=s1.dtype)
        lags = np.arange(-max_lags, max_lags + 1)
        for k in prange(len(lags)):
            lag = lags[k]
            m1, m2, v1, v2, cov = 0.0, 0.0, 0.0, 0.0, 0.0
            count = 0.0
            for i in range(n):
                j = i + lag
                s1i, s2j = s1[i], s2[j]
                if (j >= 0) and (j < n) and np.isfinite(s1i) and np.isfinite(s2j):
                    m1u = (count * m1 + s1i) / (count + 1.0)
                    m2u = (count * m2 + s2j) / (count + 1.0)
                    if count != 0:
                        d1 = s1i - m1
                        d2u = s2j - m2u
                        v1 = ((count - 1.0) * v1 + d1 * (s1i - m1u)) / (count)
                        v2 = ((count - 1.0) * v2 + (s2j - m2) * d2u) / (count)
                        cov += d1 * d2u / count
                        cov *= count / (1.0 + count)
                    count += 1.0
                    m1, m2 = m1u, m2u
            cross_corr[lag + max_lags] = count / (count - 1) * cov / np.sqrt(v1 * v2)
        return lags, cross_corr

    @classmethod
    def _best_multivariate_lag(cls, s1, s2, max_lags):
        """
        Finds the best multivariate lags for a series `s2` relative to `s1`.

        Args:
            s1 (pd.Series): Reference series.
            s2 (pd.Series): Series to lag.
            max_lags (int or str): Maximum lags or 'auto'.

        Returns:
            list: List of lags offering the best correlation.
        """
        if len(s1) != len(s2):
            raise ValueError("The length of s1 and s2 must be the same.")
        if max_lags == "auto":
            max_lags = min(50, int(0.02 * len(s1)))
        lags, cc = cls.cross_correlation(s1=s1.values, s2=s2.values, max_lags=max_lags)
        ret, cc0 = [0], cc[max_lags]
        if cc[lags > 0].max() >= cc0:
            ret.append(lags[lags > 0][cc[lags > 0].argmax()])
        if cc[lags < 0].max() >= cc0:
            ret.append(lags[lags < 0][cc[lags < 0].argmax()])
        return ret

    @classmethod
    def find_best_lags(cls, x, col, max_lags):
        """
        Finds and applies the best multivariate lags to a given column.

        Args:
            x (pd.DataFrame): Data.
            col (str): Target column.
            max_lags (int or str): Maximum lags or 'auto'.

        Returns:
            pd.DataFrame: DataFrame with lagged series.
        """
        cols = [_ for _ in x.columns if _ != col]
        ret = [x[col]]
        for other_col in cols:
            ret.append(x[other_col])
            lags = cls._best_multivariate_lag(x[col], x[other_col], max_lags=max_lags)
            for lag in lags:
                ret.append(x[other_col].shift(-lag).rename(f"{other_col}{-lag:+d}"))
        return pd.concat(ret, axis=1)

    @staticmethod
    def _process_subset_cols(X, subset_cols):
        """
        Transforms the subset of columns into indices.

        Args:
            X (pd.DataFrame): Data.
            subset_cols (None, str, list, tuple): Subset of columns.

        Returns:
            list: Indices of the columns.
        """
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
        """
        Selects a subset of rows based on `before` and `after` dates.

        Args:
            X (pd.DataFrame): Data indexed by time.
            before (str, optional): Upper date limit.
            after (str, optional): Lower date limit.

        Returns:
            list: Indices of the retained rows.
        """
        index = pd.Series(np.arange(len(X)), index=X.index)
        if before is not None:
            index = index[index.index <= pd.to_datetime(str(before))]
        if after is not None:
            index = index[pd.to_datetime(str(after)) <= index.index]
        return list(index.values)

    def _impute_col(self, x, col, subset_rows):
        """
        Imputes a specific column in a DataFrame.

        Args:
            x (pd.DataFrame): Data (including other columns potentially used
                for imputation).
            col (str): Target column to impute.
            subset_rows (list): Rows to consider.

        Returns:
            pd.Series or (pd.Series, pd.DataFrame): Imputed series and optionally a
            DataFrame of uncertainties if alpha is defined.
        """
        if isinstance(self.multivariate_lags, int) or self.multivariate_lags == "auto":
            x = self.find_best_lags(x, col, self.multivariate_lags)
        if self.ar_lags is not None:
            x = self.add_ar_lags(x, col)
        index_col = x.columns.get_loc(col)
        if self.imputer.alpha is None:
            x_col_imputed = self.imputer(x.values, subset_rows=subset_rows, subset_cols=index_col)[:, index_col]
            return pd.Series(x_col_imputed, name=col, index=x.index)
        else:
            x_imputed, uncertainties = self.imputer(x.values, subset_rows=subset_rows, subset_cols=index_col)
            x_imputed_col = x_imputed[:, index_col]
            uncertainties_col = uncertainties[:, :, index_col]
            alphas = sorted(sum([[a / 2, 1 - a / 2] for a in self.imputer.alpha], []))
            return (
                pd.Series(x_imputed_col, name=col, index=x.index),
                pd.concat(
                    [pd.Series(_, index=x.index, name=alpha) for _, alpha in zip(uncertainties_col, alphas)], axis=1
                ),
            )

    def add_ar_lags(self, x, col):
        if self.ar_lags == "auto":
            raise NotImplementedError
        else:
            ar_lags = self.ar_lags
        x_ar = [x]
        for k in sorted(ar_lags):
            x_ar.append(x[col].shift(k).rename(f"{col}{k:+d}"))
            if self.negative_ar:
                x_ar.append(-x[col].shift(k).rename(f"{col}{k:+d}_neg"))
        x = pd.concat(x_ar, axis=1)
        return x

    def _preprocess_data(self, X):
        """
        Applies preprocessing to the data and sets a time frequency.

        Args:
            X (pd.DataFrame): Original data.

        Returns:
            pd.DataFrame: Preprocessed data.
        """
        if self.preprocessing is not None:
            X_ = pd.DataFrame(self.preprocessing.fit_transform(X), index=X.index, columns=X.columns)
        else:
            X_ = X.copy()
        if X_.index.freq is None:
            X_ = X_.asfreq(pd.infer_freq(X_.index))
        non_usable_cols = X_.columns[(X_.max() - X_.min()) == 0]
        if len(non_usable_cols) > 0:
            X_ = X_.drop(columns=non_usable_cols)
        return X_

    def _select_imputation_features(self, X_, col, n_nearest_covariates, rng):
        """
        Selects the most relevant features for imputing a column.

        Args:
            X_ (pd.DataFrame): Preprocessed data.
            col (str): Target column.
            n_nearest_covariates (int or None): Number of most relevant features
                to select.
            rng (np.random.Generator): Random number generator.

        Returns:
            list: List of selected column names.
        """
        if isinstance(n_nearest_covariates, int):
            return [col] + self._sample_features(X_, col, n_nearest_covariates, rng)
        else:
            return list(X_.columns)

    @staticmethod
    def interpolate_small_gaps(df: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        Interpolates series of missing values (NaN) in a Pandas DataFrame,
        but only for gaps of length n or less.

        Parameters:
            df (pd.DataFrame): The DataFrame containing missing values.
            n (int): The maximum length of gaps to interpolate.

        Returns:
            pd.DataFrame: The DataFrame with small gaps interpolated.
        """

        def interpolate_series_with_limit(series):
            is_nan = series.isna()
            gaps = (is_nan != is_nan.shift()).cumsum()
            mask = series.groupby(gaps).transform("size") <= n
            return series.interpolate().where(mask, series)

        return df.apply(interpolate_series_with_limit, axis=0)

    def __call__(
        self,
        X,
        subset_cols=None,
        before=None,
        after=None,
        n_nearest_covariates=35,
        preimpute_covariates_limit=1,
    ) -> pd.DataFrame:
        """
        Imputes missing values in a time series DataFrame.

        Args:
            X (pd.DataFrame): Data to impute, indexed by time.
            subset_cols (None, str, list, optional): Columns to impute. Default is None
                for all columns.
            before (str, optional): Upper date limit for row selection.
            after (str, optional): Lower date limit for row selection.
            n_nearest_covariates (int, optional): Number of most relevant covariates
                to select for the imputation of a column. The selection is randomized : covariates
                highly-correlated are more likely to be selected ; covariates wich are not likely to be
                available when the imputed columns is are not likely selected. If None, uses all available
                features. Defaults to 35.
            preimpute_covariates_limit (int, optional): Fast linear interpolation of covariates before
                imputation of one column. The idea is to limit the number of calls to the Optimask solver
                as well as the chosen regressor. Values larger than a few units are not advised. Default is 1.
        Returns:
            pd.DataFrame or (pd.DataFrame, dict): Imputed data and, if alpha is
            defined, a dictionary containing uncertainties for each column.

        Note:
            About n_nearest_covariates, the probability of selecting a feature is proportional to:
            sqrt(|correlation| * co_occurrence_ratio)
            where:
            - correlation is calculated using r_regression between the feature and target
            - co_occurrence_ratio is the proportion of rows where both feature and target are non-null
        """
        rng = np.random.default_rng(self.random_state)
        check_params(X, types=pd.DataFrame)
        check_params(X.index, types=pd.DatetimeIndex)

        X_ = self._preprocess_data(X)

        columns = list(X_.columns)
        ret = []
        uncertainties = {}

        subset_rows = self._process_subset_rows(X_, before, after)
        subset_cols = self._process_subset_cols(X_, subset_cols)

        for index_col in tqdm(subset_cols, disable=(not self.verbose)):
            col = columns[index_col]
            if X_[col].isnull().mean() > 0:
                cols_in = self._select_imputation_features(X_, col, n_nearest_covariates, rng)
                X_col = X_[cols_in].copy()
                if isinstance(preimpute_covariates_limit, int):
                    covariates = [_ for _ in cols_in if _ != col]
                    X_col[covariates] = self.interpolate_small_gaps(df=X_col[covariates], n=preimpute_covariates_limit)
                if self.imputer.alpha is None:
                    ret.append(self._impute_col(x=X_col, col=col, subset_rows=subset_rows))
                else:
                    imputed_col, uncertainties_col = self._impute_col(x=X_col, col=col, subset_rows=subset_rows)
                    ret.append(imputed_col)
                    uncertainties[col] = uncertainties_col
        if ret:
            ret = pd.concat(ret, axis=1).combine_first(X)[X.columns]
        else:
            ret = X_

        if self.preprocessing is not None:
            ret = pd.DataFrame(self.preprocessing.inverse_transform(ret), columns=ret.columns, index=ret.index)

        if self.imputer.alpha is None:
            return ret
        else:
            alphas = sorted(sum([[a / 2, 1 - a / 2] for a in self.imputer.alpha], []))
            uncertainties = {
                alpha: pd.concat([uncertainties[col][alpha].rename(col) for col in uncertainties], axis=1)
                for alpha in alphas
            }
            for alpha in uncertainties:
                uncertainties[alpha] = uncertainties[alpha].reindex_like(X)
                if self.preprocessing is not None:
                    uncertainties[alpha] = self.preprocessing.inverse_transform(uncertainties[alpha])
            uncertainties = {
                col: pd.concat([uncertainties[alpha][col].rename(alpha) for alpha in alphas], axis=1)
                for col in X.columns[subset_cols]
            }
            return ret, uncertainties
