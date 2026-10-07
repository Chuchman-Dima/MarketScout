"""Sklearn-трансформер для LightGBM pipeline (потрібен для joblib.load)."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalCaster(BaseEstimator, TransformerMixin):
    """Переводить вказані колонки в pandas 'category' dtype з фіксованим словником."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.categories_ = {c: pd.Categorical(X[c]).categories for c in self.columns}
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columns:
            X[c] = pd.Categorical(X[c], categories=self.categories_[c])
        return X
