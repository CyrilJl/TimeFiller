from typing import Union

import numpy as np
import pandas as pd
from optimask import OptiMask
from sklearn.linear_model import LinearRegression


class ImputeMultiVariate:
    """
    The `ImputeMultiVariate` class has been developed to address the problem of imputing missing values in multivariate data.
    It relies on regression techniques to estimate missing values using information available in other columns.
    This class offers great flexibility by allowing users to specify a custom regression estimator, while also providing
    a default option to use linear regression from the scikit-learn library. Additionally, it takes into account important parameters
    such as the maximum fraction of missing values allowed in a column and the minimum number of samples required for
    imputation.

    Using the `ImputeMultiVariate` class transforms incomplete data into complete data rigorously, facilitating subsequent
    analysis and modeling steps. This class is particularly useful for engineers and data analysts who want to effectively
    handle missing values in their multivariate datasets.
    """

    def __init__(self, estimator=None, na_frac_max=0.33, min_samples_train=50, verbose=False):
        """
        Initialize an instance of the ImputeMultiVariate class.

        Args:
            estimator (object, optional): The regression estimator to use for imputation. Default uses
            LinearRegression() from scikit-learn.
            na_frac_max (float, optional): The maximum fraction of missing values accepted for a column to
            impute. Default is 0.33.
            min_samples_train (int, optional): The minimum number of samples required to perform imputation.
            Default is 50.
            verbose (bool, optional): If True, display debug information during imputation. Default is False.
        """
        self.estimator = LinearRegression() if estimator is None else estimator
        self.na_frac_max = na_frac_max
        self.min_samples_train = min_samples_train
        self.verbose = bool(verbose)

    @staticmethod
    def process_subset(X, subset):
        if subset is None:
            return np.arange(X.shape[1])
        if isinstance(X, pd.DataFrame):
            columns = list(X.columns)
            if isinstance(subset, str):
                return [columns.index(subset)]
            if isinstance(subset, (list, tuple)):
                return [columns.index(_) for _ in subset]
        if isinstance(X, np.ndarray):
            if isinstance(subset, int):
                return [subset]
            if isinstance(subset, (list, tuple)):
                return list(subset)
        raise TypeError()

    @staticmethod
    def _prepare_data_for_imputation(Xc, mask_na, col_to_impute):
        rows_to_impute = mask_na[:, col_to_impute].nonzero()[0]
        other_cols = np.array([_ for _ in range(Xc.shape[1]) if _ != col_to_impute])

        patterns, indexes = np.unique(~mask_na[rows_to_impute][:, other_cols], return_inverse=True, axis=0)
        index_predict = [rows_to_impute[indexes == k] for k in range(len(patterns))]
        columns = [other_cols[pattern] for pattern in patterns]
        return index_predict, columns

    @staticmethod
    def _prepare_train_and_pred_data(Xc, mask_na, columns, col_to_impute, index_predict):
        trainable_rows = (~mask_na[:, col_to_impute]).nonzero()[0]
        rows, cols = OptiMask().solve(Xc[trainable_rows][:, columns])

        X_train, y_train = Xc[trainable_rows[rows]][:, columns[cols]], Xc[trainable_rows[rows]][:, col_to_impute]
        X_predict = Xc[index_predict][:, columns[cols]]
        return X_train, y_train, X_predict

    def _perform_imputation(self, X_train, y_train, X_predict):
        model = self.estimator.fit(X_train, y_train)
        return model.predict(X_predict)

    def _impute(self, X, subset):
        Xc = np.array(X, dtype=float).copy()
        ret = np.array(X, dtype=float).copy()

        mask_na = np.isnan(Xc)

        imputable_cols = (mask_na.mean(axis=0) <= self.na_frac_max).nonzero()[0]
        subset = self.process_subset(X, subset)
        imputable_cols = [k for k in imputable_cols if k in subset]

        for col_to_impute in imputable_cols:
            index_predict, columns = self._prepare_data_for_imputation(Xc=Xc, mask_na=mask_na, col_to_impute=col_to_impute)
            for cols, index in zip(columns, index_predict):
                X_train, y_train, X_predict = self._prepare_train_and_pred_data(
                    Xc=Xc, mask_na=mask_na, columns=cols, col_to_impute=col_to_impute, index_predict=index)
                if len(X_train) >= self.min_samples_train:
                    ret[index, col_to_impute] = self._perform_imputation(X_train, y_train, X_predict)

        return ret

    def __call__(self, X: Union[np.ndarray, pd.DataFrame], subset=None) -> Union[np.ndarray, pd.DataFrame]:
        """
        Perform imputation on the input data.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): The input data.
            subset (Optional[Union[str, int, List[Union[str, int]], Tuple[Union[str, int]]]], optional): The subset of columns to include in
            imputation. Default is None.

        Returns:
            Union[np.ndarray, pd.DataFrame]: Data with imputed missing values.
        Raises:
            TypeError: If NaN values are present in the training data.
        """
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise TypeError()
        Xs = self._impute(X=X, subset=subset)

        if isinstance(X, np.ndarray):
            return Xs

        if isinstance(X, pd.DataFrame):
            kwargs = dict(index=X.index, columns=X.columns)
            return pd.DataFrame(Xs, **kwargs)
