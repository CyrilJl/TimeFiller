# -*- coding: utf-8 -*-


import numpy as np
from sklearn.base import TransformerMixin

__all__ = ['PositiveOutput']


class PositiveOutput(TransformerMixin):
    def __init__(self, q=10, v=None):
        """
        Initialise un objet PositiveOutput.

        Parameters:
        - q (float, optional): Le quantile utilisé comme seuil pour l'expansion. 
                              Par défaut, q=10, ce qui signifie que le 10e percentile est utilisé comme seuil.
        - v (float, optional): Valeur fixe utilisée comme seuil pour l'expansion négative.
                              Si `v` est spécifié, ce seuil sera utilisé pour toutes les caractéristiques.
                              Par défaut, v=None, ce qui signifie que le seuil est calculé automatiquement à partir des données.

        Raises:
        - ValueError: Si les deux arguments `q` et `v` sont `None`.
        """
        if q is None and v is None:
            raise ValueError("Au moins l'un des arguments 'q' ou 'v' doit être différent de None.")

        self.q = q
        self.v = v
        self.thresholds_ = None

    def fit(self, X, y=None):
        """
        Calcule et enregistre les seuils nécessaires pour l'expansion négative.

        Parameters:
        - X (array-like): Les données d'entraînement.
        - y (array-like, optional): Les étiquettes d'entraînement. Non utilisé ici.

        Returns:
        - self: L'objet PositiveOutput ajusté.
        """
        if np.nanmin(X) < 0:
            raise ValueError("Les données ne doivent pas contenir de valeurs négatives.")

        if self.v is None:
            self.thresholds_ = np.nanpercentile(X, q=self.q, axis=0)
        else:
            self.thresholds_ = np.full(shape=X.shape[1], fill_value=self.v)
        return self

    def transform(self, X, y=None):
        """
        Applique l'expansion négative sur les données.

        Parameters:
        - X (array-like): Les données à transformer.
        - y (array-like, optional): Les étiquettes. Non utilisé ici.

        Returns:
        - array-like: Les données transformées avec l'expansion négative.
        """
        X = np.asarray(X)
        mask = X < self.thresholds_
        return np.where(mask, 2 * X - self.thresholds_, X)

    def inverse_transform(self, X, y=None):
        """
        Inverse l'expansion négative sur les données transformées.

        Parameters:
        - X (array-like): Les données transformées.
        - y (array-like, optional): Les étiquettes. Non utilisé ici.

        Returns:
        - array-like: Les données inversées après l'expansion négative.
        """
        X = np.asarray(X)
        mask = X < self.thresholds_
        return np.maximum(0, np.where(mask, 0.5 * X + self.thresholds_ / 2, X))
