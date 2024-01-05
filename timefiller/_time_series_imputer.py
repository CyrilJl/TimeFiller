import pandas as pd

from ._multivariate_imputer import ImputeMultiVariate


class TimeSeriesImputer:
    """Class TimeSeriesImputer: This class allows imputing time series data using a hybrid approach
    between multivariate imputation and autoregressive imputation, using lags to create shifted variables.
    """

    def __init__(self, estimator=None, lags=None, na_frac_max=0.33, min_samples_train=50, verbose=False):
        """
        Initialize a time series imputer.

        Args:
            estimator (object, optional): The sklearn estimator used for multivariate imputation. Default is None.
            lags (int, tuple, list, optional): The lags to use for creating shifted variables.
                Default is None.
            na_frac_max (float, optional): The maximum fraction of missing data allowed. Default is 0.33.
            min_samples_train (int, optional): The minimum number of samples required for estimator training.
                Default is 50.
            verbose (bool, optional): If True, display debug messages. Default is False.
        """
        self.imputer = ImputeMultiVariate(estimator=estimator, na_frac_max=na_frac_max, min_samples_train=min_samples_train, verbose=verbose)
        self.lags = self._process_lags(lags)
        self.verbose = bool(verbose)

    @staticmethod
    def _process_lags(lags):
        if lags is None:
            return None
        if isinstance(lags, int):
            lags = list(range(-abs(lags)-1, -1)) + list(range(1, abs(lags)+1))
            return sorted(lags)
        if isinstance(lags, (tuple, list)):
            lags = [-k for k in lags if k != 0] + [k for k in lags if k != 0]
            return sorted(list(set(lags)))
        raise TypeError("Unrecognized format for lags!")

    def __call__(self, X: pd.DataFrame, subset=None) -> pd.DataFrame:
        """
        Apply multivariate imputation to the data using the specified parameters.

        Args:
            X (pd.DataFrame): The DataFrame containing the data to impute.

        Returns:
            pd.DataFrame: The imputed DataFrame.

        Raises:
            ValueError: If the format of the input data is not valid.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError()
        else:
            X_ = X.copy()
        if not isinstance(X_.index, pd.DatetimeIndex):
            raise TypeError()

        if X_.index.freq is None:
            guessed_freq = pd.infer_freq(X_.index)
            X_ = X_.asfreq(guessed_freq).copy()

        ret = []

        if subset is None:
            cols_to_impute = X_.columns
        if isinstance(subset, str):
            cols_to_impute = [subset]
        if isinstance(subset, (tuple, list)):
            cols_to_impute = list(subset)

        for col in cols_to_impute:
            x = X_.copy()
            if self.lags is not None:
                for k in sorted(self.lags):
                    x[f"{col}{k:+d}"] = x[col].shift(k)
            ret.append(self.imputer(x, subset=col)[col])

        ret = pd.concat(ret, axis=1).reindex_like(X).combine_first(X)
        return ret
