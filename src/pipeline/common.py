"""
Спільні речі для всіх train_*.py та cross-validation.py:
- константи (RANDOM_SEED, CAT_FEATURES)
- завантаження train/test parquet
- підрахунок метрик + запис результату в _results.json
- CategoricalCaster - sklearn-трансформер для LightGBM/XGBoost, який
  переводить задані колонки в pandas 'category' dtype і ЗАПАМ'ЯТОВУЄ
  категорії, побачені на fit() (train), щоб на transform() (test/прод)
  узгоджувати їх з тим самим словником, а не плодити розсинхронізацію.
"""

import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

RANDOM_SEED = 42
CAT_FEATURES = ["Mark", "Model", "Gearbox", "Fuel_Type"]

DATA_DIR = "../data/parquet"
RESULTS_PATH = "../data/_results.json"


def load_train_test():
    """Повертає (X_train, X_test, y_train_raw, y_test_raw) з parquet,
    підготовлених у prepare.py. y_* - у СИРИХ доларах (не логарифм)."""
    X_train = pd.read_parquet(f"{DATA_DIR}/X_train.parquet")
    X_test = pd.read_parquet(f"{DATA_DIR}/X_test.parquet")
    y_train_raw = pd.read_parquet(f"{DATA_DIR}/y_train.parquet")["Price_USD"]
    y_test_raw = pd.read_parquet(f"{DATA_DIR}/y_test.parquet")["Price_USD"]
    return X_train, X_test, y_train_raw, y_test_raw


def evaluate(y_true_raw, pred_log) -> dict:
    """pred_log - прогноз моделі в лог-шкалі (усі моделі тут навчені на
    np.log1p(Price_USD)); функція сама робить expm1 і рахує метрики
    в реальних доларах."""
    pred = np.expm1(pred_log)
    return {
        "MAE": round(float(mean_absolute_error(y_true_raw, pred)), 1),
        "MAPE": round(float(mean_absolute_percentage_error(y_true_raw, pred) * 100), 2),
        "R2": round(float(r2_score(y_true_raw, pred)), 4),
    }


def save_result(model_name: str, metrics: dict) -> None:
    """Дописує результат у _results.json (один рядок = один запуск) і
    одразу друкує його в консоль."""
    row = {"model": model_name, **metrics}
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"{model_name}: " + ", ".join(f"{k}={v}" for k, v in metrics.items()))


class CategoricalCaster(BaseEstimator, TransformerMixin):
    """Переводить вказані колонки в pandas 'category' dtype.

    LightGBM і XGBoost автоматично трактують такі колонки як категоріальні
    (перевірено: LightGBM - через pandas_categorical_, XGBoost - через
    feature_types='c' при enable_categorical=True) - жодних додаткових
    параметрів при .fit() не потрібно.

    Категорії запам'ятовуються на fit() (з train) і застосовуються на
    transform() - тому нове/рідкісне значення в test/проді не створює
    нову категорію "з повітря", а трактується як NaN у межах словника,
    побаченого під час навчання.
    """

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