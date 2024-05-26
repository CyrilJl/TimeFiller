import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm.auto import tqdm

from ._misc import check_params
from ._multivariate_imputer import ImputeMultiVariate


class TimeSeriesImputer:
    """Classe pour l'imputation de séries temporelles.

    Args:
        estimator (object, optional): Estimation utilisée pour l'imputation.
        preprocessing (callable, optional): Prétraitement des données.
        ar_lags (int, list, numpy.ndarray or tuple, optional): Retards auto-régressifs à considérer.
        multivariate_lags (int or None, optional): Retards multivariés à considérer.
        na_frac_max (float, optional): Fraction maximale de valeurs manquantes autorisées.
        min_samples_train (int, optional): Nombre minimum d'échantillons pour l'apprentissage.
        weighting_func (callable, optional): Fonction de pondération pour l'imputation.
        optimask_n_tries (int, optional): Nombre d'essais pour l'optimisation.
        verbose (bool, optional): Afficher les détails du processus.
        random_state (int or None, optional): État aléatoire pour la reproductibilité.
    """

    def __init__(self, estimator=None, preprocessing=None, ar_lags=None, multivariate_lags=None, na_frac_max=0.33,
                 min_samples_train=50, weighting_func=None, optimask_n_tries=10, verbose=False, random_state=None):

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
            ar_lags = list(range(-abs(ar_lags)-1, -1)) + list(range(1, abs(ar_lags)+1))
            return sorted(ar_lags)
        if isinstance(ar_lags, (tuple, list, np.ndarray)):
            ar_lags = [-k for k in ar_lags if k != 0] + [k for k in ar_lags if k != 0]
            return sorted(list(set(ar_lags)))

    @staticmethod
    def _sample_features(corr, common_samples, col, n_nearest_features, rng):
        s1 = corr[col].drop(col)
        s2 = common_samples[col].drop(col)
        p = np.sqrt(abs(s1.values) * s2.values)
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
        raise TypeError()

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
        """Méthode d'appel pour l'imputation.

        Args:
            X (DataFrame): Données à imputer.
            subset_cols (str, list, tuple or pandas.core.indexes.base.Index, optional): Colonnes à imputer. Par défaut, toutes les colonnes seront imputées.
            before (str or pd.Timestamp or None, optional): Date avant laquelle les données sont imputées. Par défaut, aucune limite temporelle inférieure n'est définie.
            after (str or pd.Timestamp or None, optional): Date après laquelle les données sont imputées. Par défaut, aucune limite temporelle supérieure n'est définie.
            n_nearest_features (int, optional): Nombre de caractéristiques les plus proches à considérer. Une heuristique est utilisée : les caractéristiques
                utilisées sont sélectionnées de manière aléatoire, en fonction de leurs corrélations avec la caractéristique à imputer, ainsi que du nombre
                d'observations temporelles communes avec la caractéristique à imputer.

        Returns:
            DataFrame or tuple: Données imputées.
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

        ret = [pd.Series(index=X.index)]
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
