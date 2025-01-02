import numpy as np
from mapie.regression import MapieRegressor
from numba import njit, prange
from optimask import OptiMask
from tqdm.auto import tqdm

from ._estimator import FastRidge
from ._misc import InvalidEstimatorError, check_params


class ImputeMultiVariate:
    """
    Multivariate imputation of missing data in a numerical array.

    This class uses an estimator to fit a model on training data (non-missing)
    and predict missing values. It can also estimate prediction intervals
    if an instance of `MapieRegressor` is used with the `alpha` parameter.
    """

    def __init__(
        self,
        estimator=None,
        alpha=None,
        na_frac_max=0.33,
        min_samples_train=50,
        weighting_func=None,
        optimask_n_tries=1,
        random_state=None,
        verbose=0,
    ):
        """
        Initialize the multivariate imputer.

        Args:
            estimator (object, optional): Estimator to use for regression.
                Must have `fit` and `predict` methods. Default is `Ridge()`.
            alpha (float or list, optional): Quantile levels for obtaining
                prediction intervals via `MapieRegressor`. If None, no intervals.
                Default is None.
            na_frac_max (float, optional): Maximum fraction of missing values
                allowed per column for imputation. Default is 0.33.
            min_samples_train (int, optional): Minimum number of training samples
                required to fit the model. Default is 50.
            weighting_func (callable, optional): Weighting function for samples
                during model fitting. Must accept an array of indices and return
                an array of weights. Default is None.
            optimask_n_tries (int, optional): Number of attempts to solve the
                sample selection problem using `OptiMask`. Default is 1.
            random_state (int, optional): Seed for the random generator.
                Default is None.
            verbose (int or bool, optional): Verbosity level. Default is 0.
        """
        self.estimator = self._process_estimator(estimator)
        self.alpha = (alpha,) if isinstance(alpha, float) else alpha
        if self.alpha is not None:
            self.estimator = MapieRegressor(estimator=self.estimator)
        self.na_frac_max = check_params(na_frac_max, types=(float))
        self.min_samples_train = check_params(min_samples_train, types=int)
        self.weighting_func = weighting_func
        self.optimask = OptiMask(n_tries=optimask_n_tries, random_state=random_state)
        self.verbose = check_params(verbose, types=(bool, int))

    @staticmethod
    def _process_estimator(estimator):
        """
        Validate and process the provided estimator.

        Args:
            estimator (object): The estimator to validate.

        Returns:
            object: The validated estimator.

        Raises:
            TypeError: If the estimator does not have `fit` and `predict` methods.
        """
        return (
            FastRidge()
            if estimator is None
            else estimator
            if hasattr(estimator, "fit") and hasattr(estimator, "predict")
            else InvalidEstimatorError()
        )

    @staticmethod
    def _process_subset(X, subset, axis):
        """
        Processes the provided subset (rows or columns) and returns a list of valid indices.

        Args:
            X (np.ndarray): Data array.
            subset (int, list, np.ndarray, tuple, None): Subset of rows or columns.
            axis (int): 0 for rows, 1 for columns.

        Returns:
            list: List of valid indices for the subset.

        Raises:
            InvalidSubsetError: If an index is out of bounds.
        """
        n = X.shape[axis]
        check_params(subset, types=(int, list, np.ndarray, tuple, type(None)))
        return (
            list(range(n)) if subset is None else [subset] if isinstance(subset, int) and subset < n else sorted(subset)
        )

    @classmethod
    def _prepare_data(cls, mask_nan, col_to_impute, subset_rows):
        """
        Prepares the data for imputing a specific column.

        Args:
            mask_nan (np.ndarray): Boolean mask of missing values.
            col_to_impute (int): Index of the column to impute.
            subset_rows (list): List of row indices to consider.

        Returns:
            tuple: (index_predict, columns)
            - index_predict (list): List of arrays of row indices to predict.
            - columns (list): List of arrays of column indices available for training.
        """
        rows_to_impute = np.flatnonzero(mask_nan[:, col_to_impute] & ~mask_nan.all(axis=1))
        rows_to_impute = np.intersect1d(ar1=rows_to_impute, ar2=subset_rows)
        other_cols = np.setdiff1d(ar1=np.arange(mask_nan.shape[1]), ar2=[col_to_impute])
        patterns, indexes = np.unique(
            ~cls._subset(X=mask_nan, rows=rows_to_impute, columns=other_cols), return_inverse=True, axis=0
        )
        index_predict = [rows_to_impute[indexes == k] for k in range(len(patterns))]
        columns = [other_cols[pattern] for pattern in patterns]
        return index_predict, columns

    @staticmethod
    @njit(parallel=True, fastmath=True, boundscheck=False, cache=True)
    def _split(X, index_predict, selected_rows, selected_cols, col_to_impute):
        """
        Splits the data into training and prediction sets.

        Args:
            X (np.ndarray): Data array.
            index_predict (np.ndarray): Indices of rows to predict.
            selected_rows (np.ndarray): Indices of rows selected for training.
            selected_cols (np.ndarray): Indices of columns selected for training.
            col_to_impute (int): Index of the column to impute.

        Returns:
            tuple: (X_train, y_train, X_pred)
            - X_train (np.ndarray): Training data.
            - y_train (np.ndarray): Training targets.
            - X_pred (np.ndarray): Data to predict.
        """
        X_train = np.empty((len(selected_rows), len(selected_cols)))
        y_train = np.empty(len(selected_rows))
        X_pred = np.empty((len(index_predict), len(selected_cols)))
        for i in prange(len(selected_rows)):
            for j in prange(len(selected_cols)):
                X_train[i, j] = X[selected_rows[i], selected_cols[j]]
            y_train[i] = X[selected_rows[i], col_to_impute]
        for i in prange(len(index_predict)):
            for j in prange(len(selected_cols)):
                X_pred[i, j] = X[index_predict[i], selected_cols[j]]
        return X_train, y_train, X_pred

    @staticmethod
    @njit(parallel=True, fastmath=True, boundscheck=False, cache=True)
    def _subset(X, rows, columns):
        """
        Extracts a subset from the array X.

        Args:
            X (np.ndarray): Data array.
            rows (np.ndarray): Row indices.
            columns (np.ndarray): Column indices.

        Returns:
            np.ndarray: Corresponding subarray.
        """
        Xs = np.empty((len(rows), len(columns)), dtype=X.dtype)
        for i in prange(len(rows)):
            for j in range(len(columns)):
                Xs[i, j] = X[rows[i], columns[j]]
        return Xs

    def _prepare_train_and_pred_data(self, X, mask_nan, columns, col_to_impute, index_predict):
        """
        Prepares training and prediction data by selecting an optimal subset
        via `OptiMask`.

        Args:
            X (np.ndarray): Data array.
            mask_nan (np.ndarray): Mask of missing values.
            columns (np.ndarray): Columns available for training.
            col_to_impute (int): Column to impute.
            index_predict (np.ndarray): Indices of rows to predict.

        Returns:
            tuple: (X_train, y_train, X_pred, selected_rows, selected_cols)
        """
        trainable_rows = np.flatnonzero(~mask_nan[:, col_to_impute])
        rows, cols = self.optimask.solve(self._subset(X, trainable_rows, columns))
        selected_rows, selected_cols = trainable_rows[rows], columns[cols]
        X_train, y_train, X_pred = self._split(
            X=X,
            index_predict=index_predict,
            selected_rows=selected_rows,
            selected_cols=selected_cols,
            col_to_impute=col_to_impute,
        )
        return X_train, y_train, X_pred, selected_rows, selected_cols

    def _perform_imputation(self, X_train, y_train, X_predict, selected_rows):
        """
        Fits the model and predicts the missing values.

        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Training targets.
            X_predict (np.ndarray): Data to predict.
            selected_rows (np.ndarray): Rows selected for training.

        Returns:
            np.ndarray or tuple: Predictions only or (predictions, confidence intervals)
            if self.alpha is defined.
        """
        if callable(self.weighting_func):
            sample_weight = self.weighting_func(selected_rows)
        else:
            sample_weight = None
        self.estimator.fit(X_train, y_train, sample_weight=sample_weight)
        if self.alpha is not None:
            return self.estimator.predict(X_predict, alpha=self.alpha)
        else:
            return self.estimator.predict(X_predict)

    def _impute(self, X, subset_rows, subset_cols):
        """
        Imputes missing values in the array X for the specified rows and columns.

        Args:
            X (np.ndarray): Data array.
            subset_rows (list): Row indices.
            subset_cols (list): Column indices.

        Returns:
            np.ndarray or tuple: Imputed array and optionally uncertainties
            if self.alpha is defined.
        """
        imputation, mask_nan = X.copy(), np.isnan(X)
        uncertainties = np.full((2 * len(self.alpha),) + imputation.shape, np.nan) if self.alpha else None
        imputable_cols = np.intersect1d(
            np.flatnonzero((0 < mask_nan[subset_rows].sum(axis=0)) & (mask_nan.mean(axis=0) <= self.na_frac_max)),
            subset_cols,
        )
        for col_to_impute in tqdm(imputable_cols, disable=(self.verbose < 1)):
            index_predict, columns = self._prepare_data(mask_nan, col_to_impute, subset_rows)
            for cols, index in zip(columns, index_predict):
                X_train, y_train, X_predict, selected_rows, _ = self._prepare_train_and_pred_data(
                    X, mask_nan, cols, col_to_impute, index
                )
                if len(X_train) >= self.min_samples_train:
                    pred_result = self._perform_imputation(X_train, y_train, X_predict, selected_rows)
                    if self.alpha:
                        imputation[index, col_to_impute], s = pred_result
                        uncertainties[:, index, col_to_impute] = np.sort(s.reshape(-1, 2 * len(self.alpha)), axis=1).T
                    else:
                        imputation[index, col_to_impute] = pred_result
        return (imputation, uncertainties) if self.alpha else imputation

    def __call__(self, X: np.ndarray, subset_rows=None, subset_cols=None):
        """
        Imputes missing data in X.

        Args:
            X (np.ndarray): Data array to impute.
            subset_rows (int, list, np.ndarray, tuple, None, optional): Subset of rows.
            subset_cols (int, list, np.ndarray, tuple, None, optional): Subset of columns.

        Returns:
            np.ndarray or tuple: Imputed array and optionally uncertainties.
        """
        check_params(X, types=np.ndarray)
        check_params(X.dtype.kind, params=("i", "f"))
        subset_rows = self._process_subset(X=X, subset=subset_rows, axis=0)
        subset_cols = self._process_subset(X=X, subset=subset_cols, axis=1)
        return self._impute(X=X, subset_rows=subset_rows, subset_cols=subset_cols)
