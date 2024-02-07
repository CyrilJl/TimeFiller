# -*- coding: utf-8 -*-

# author : Cyril Joly

import numpy as np
from optimask import OptiMask
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, RobustScaler

from ._misc import check_params


class ImputeMultiVariate:
    """
    The ImputeMultiVariate algorithm takes a distinct approach to multivariate data imputation,
    relying on user-provided regressors (defaulting to Lasso regression) to establish relationships
    between available data and the target column for imputation. Unlike conventional methods like
    Expectation-Maximization or MICE, this algorithm dynamically identifies valid data subsets using
    the OptiMask solver, adapting its input features for regression based on the most substantial
    available data. This adaptability allows the algorithm to effectively handle complex
    relationships in the data, offering a pragmatic and versatile solution for imputation.
    """

    def __init__(self, estimator='positive with intercept', na_frac_max=0.33, min_samples_train=50, optimask_n_tries=5, verbose=False):
        """
        Initialize the ImputeMultiVariate object.

        Parameters:
        - estimator (str or object): Estimator for imputation. Defaults to 'positive with intercept'.
        - na_frac_max (float): Maximum fraction of missing values allowed for imputation.
        - min_samples_train (int): Minimum number of samples required for training the estimator.
        - optimask_n_tries (int): Number of tries for the OptiMask solver.
        - verbose (bool): Verbosity level.
        """
        self.estimator = self._process_estimator(estimator)
        self.na_frac_max = na_frac_max
        self.min_samples_train = min_samples_train
        self.optimask = OptiMask(n_tries=optimask_n_tries)
        self.verbose = bool(verbose)

    @staticmethod
    def _process_estimator(estimator):
        if estimator == 'positive with intercept':
            return make_pipeline(PolynomialFeatures(degree=1, include_bias=True),
                                 Lasso(max_iter=10000, positive=True, fit_intercept=False))
        if estimator == 'positive without intercept':
            return make_pipeline(Lasso(max_iter=10000, positive=True, fit_intercept=False))
        if estimator is None:
            return Lasso(max_iter=10000)
        if not (hasattr(estimator, 'fit') and hasattr(estimator, 'predict')):
            raise TypeError()
        else:
            return estimator

    @staticmethod
    def _process_subset(X, subset, axis):
        n = X.shape[axis]
        check_params(subset, types=(int, list, np.ndarray, tuple, type(None)))
        if isinstance(subset, int):
            if subset >= n:
                raise ValueError()
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
        patterns, indexes = np.unique(~mask_nan[rows_to_impute][:, other_cols], return_inverse=True, axis=0)
        index_predict = [rows_to_impute[indexes == k] for k in range(len(patterns))]
        columns = [other_cols[pattern] for pattern in patterns]
        return index_predict, columns

    def _prepare_train_and_pred_data(self, X, mask_nan, columns, col_to_impute, index_predict):
        trainable_rows = np.flatnonzero(~mask_nan[:, col_to_impute])
        rows, cols = self.optimask.solve(X[trainable_rows][:, columns])
        selected_rows, selected_cols = trainable_rows[rows], columns[cols]
        X_train, y_train = X[selected_rows][:, selected_cols], X[selected_rows][:, col_to_impute]
        X_predict = X[index_predict][:, selected_cols]
        return X_train, y_train, X_predict

    def _perform_imputation(self, X_train, y_train, X_predict):
        model = self.estimator.fit(X_train, y_train)
        return model.predict(X_predict)

    def _impute(self, X, subset_rows, subset_cols):
        ret = np.array(X, dtype=float)
        mask_nan = np.isnan(X)
        nan_mean = mask_nan.mean(axis=0)
        imputable_cols = (0 < mask_nan[subset_rows].sum(axis=0)) & (mask_nan.mean(axis=0) <= self.na_frac_max) & (np.nanstd(X, axis=0) > 0)
        imputable_cols = np.intersect1d(np.flatnonzero(imputable_cols), subset_cols)

        for col_to_impute in imputable_cols:
            index_predict, columns = self._prepare_data(mask_nan=mask_nan, col_to_impute=col_to_impute, subset_rows=subset_rows)
            for cols, index in zip(columns, index_predict):
                X_train, y_train, X_predict = self._prepare_train_and_pred_data(X, mask_nan, cols, col_to_impute, index)
                if len(X_train) >= self.min_samples_train:
                    ret[index, col_to_impute] = self._perform_imputation(X_train, y_train, X_predict)
        return ret

    def __call__(self, X, subset_rows=None, subset_cols=None) -> np.ndarray:
        """
        Perform data imputation.

        Parameters:
        - X (np.ndarray): Input array.
        - subset_rows (list or None): Subset of rows to consider.
        - subset_cols (list or None): Subset of columns to consider.

        Returns:
        - np.ndarray: Imputed array.
        """
        check_params(X, types=np.ndarray)
        subset_rows = self._process_subset(X=X, subset=subset_rows, axis=0)
        subset_cols = self._process_subset(X=X, subset=subset_cols, axis=1)
        scaler = RobustScaler(with_centering=False)
        Xt = scaler.fit_transform(X)
        Xt = self._impute(X=Xt, subset_rows=subset_rows, subset_cols=subset_cols)
        Xt = scaler.inverse_transform(Xt)
        return Xt
