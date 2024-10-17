import numpy as np
from optimask import OptiMask
from sklearn.linear_model import LinearRegression

from ._misc import InvalidEstimatorError, InvalidSubsetError, check_params


class ImputeMultiVariate:
    """
    The ``ImputeMultiVariate`` class has been developed to address the problem of imputing missing values in multivariate data.
    It relies on regression techniques to estimate missing values using information available in other columns.
    This class offers great flexibility by allowing users to specify a custom regression estimator, while also providing
    a default option to use linear regression from the scikit-learn library. Additionally, it takes into account important parameters
    such as the maximum fraction of missing values allowed in a column and the minimum number of samples required for
    imputation.
    """

    def __init__(self, estimator=None, preprocessing=None, na_frac_max=0.33, min_samples_train=50, weighting_func=None, optimask_n_tries=1, verbose=False):
        self.estimator = self._process_estimator(estimator)
        self.preprocessing = preprocessing
        self.na_frac_max = na_frac_max
        self.min_samples_train = min_samples_train
        self.weighting_func = weighting_func
        self.optimask = OptiMask(n_tries=optimask_n_tries)
        self.verbose = bool(verbose)

    def __repr__(self):
        params = ", ".join(f"{k}={getattr(self, k)}" for k in ('estimator',))
        return f"ImputeMultiVariate({params})"

    @staticmethod
    def _process_estimator(estimator):
        if estimator is None:
            return LinearRegression()
        if not (hasattr(estimator, 'fit') and hasattr(estimator, 'predict')):
            raise InvalidEstimatorError()
        else:
            return estimator

    @staticmethod
    def _process_subset(X, subset, axis):
        n = X.shape[axis]
        check_params(subset, types=(int, list, np.ndarray, tuple, type(None)))
        if isinstance(subset, int):
            if subset >= n:
                raise InvalidSubsetError(f"The subset index {subset} is out of bounds for axis {axis} with size {n}.")
            else:
                return [subset]
        if isinstance(subset, (list, np.ndarray, tuple)):
            return sorted(list(subset))
        if subset is None:
            return list(range(n))

    @staticmethod
    def _prepare_data(mask_nan, col_to_impute, subset_rows):
        rows_to_impute = np.flatnonzero(mask_nan[:, col_to_impute] & ~mask_nan.all(axis=1))
        rows_to_impute = np.intersect1d(ar1=rows_to_impute, ar2=subset_rows)
        other_cols = np.setdiff1d(ar1=np.arange(mask_nan.shape[1]), ar2=[col_to_impute])
        patterns, indexes = np.unique(~mask_nan[np.ix_(rows_to_impute, other_cols)], return_inverse=True, axis=0)
        index_predict = [rows_to_impute[indexes == k] for k in range(len(patterns))]
        columns = [other_cols[pattern] for pattern in patterns]
        return index_predict, columns

    def _prepare_train_and_pred_data(self, X, mask_nan, columns, col_to_impute, index_predict):
        trainable_rows = np.flatnonzero(~mask_nan[:, col_to_impute])
        rows, cols = self.optimask.solve(X[np.ix_(trainable_rows, columns)])
        selected_rows, selected_cols = trainable_rows[rows], columns[cols]
        X_train, y_train = X[np.ix_(selected_rows, selected_cols)], X[selected_rows, col_to_impute]
        X_predict = X[np.ix_(index_predict, selected_cols)]
        return X_train, y_train, X_predict, selected_rows, selected_cols

    def _perform_imputation(self, X_train, y_train, X_predict, selected_rows):
        if callable(self.weighting_func):
            sample_weight = self.weighting_func(selected_rows)
        else:
            sample_weight = None
        model = self.estimator.fit(X_train, y_train, sample_weight=sample_weight)
        return model.predict(X_predict)

    def _impute(self, X, subset_rows, subset_cols):
        ret = np.array(X, dtype=float)
        mask_nan = np.isnan(X)
        imputable_cols = (0 < mask_nan[subset_rows].sum(axis=0)) & (mask_nan.mean(axis=0) <= self.na_frac_max) & (np.nanstd(X, axis=0) > 0)
        imputable_cols = np.intersect1d(np.flatnonzero(imputable_cols), subset_cols)

        for col_to_impute in imputable_cols:
            index_predict, columns = self._prepare_data(mask_nan=mask_nan, col_to_impute=col_to_impute, subset_rows=subset_rows)
            for cols, index in zip(columns, index_predict):
                X_train, y_train, X_predict, selected_rows, _ = self._prepare_train_and_pred_data(X, mask_nan, cols, col_to_impute, index)
                if len(X_train) >= self.min_samples_train:
                    ret[index, col_to_impute] = self._perform_imputation(X_train, y_train, X_predict, selected_rows)
        return ret

    def __call__(self, X, subset_rows=None, subset_cols=None):
        """
        Main method to perform missing value imputation in a multidimensional array.

        Args:
            X (numpy.ndarray): The input array with missing values.
            subset_rows (int, list, numpy.ndarray, tuple, None): The row indices to consider for imputation. If None, all rows are considered. (default: None)
            subset_cols (int, list, numpy.ndarray, tuple, None): The column indices to consider for imputation. If None, all columns are considered. (default: None)

        Returns:
            numpy.ndarray: The array with missing values imputed.
        """
        check_params(X, types=np.ndarray)
        subset_rows = self._process_subset(X=X, subset=subset_rows, axis=0)
        subset_cols = self._process_subset(X=X, subset=subset_cols, axis=1)
        if self.preprocessing is not None:
            Xt = self.preprocessing.fit_transform(X)
        else:
            Xt = X.copy()

        Xt = self._impute(X=Xt, subset_rows=subset_rows, subset_cols=subset_cols)
        if self.preprocessing is not None:
            Xt = self.preprocessing.inverse_transform(Xt)
        return Xt
