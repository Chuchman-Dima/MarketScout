"""
Пошук найкращих гіперпараметрів для LightGBM за допомогою Optuna.
Оптимізує MAE на основі 3-Fold крос-валідації.
"""

import optuna
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from common import (
    CAT_FEATURES,
    RANDOM_SEED,
    CategoricalCaster,
    load_train_test
)


def objective(trial):
    # Завантаження даних
    X_train, _, y_train_raw, _ = load_train_test()
    y_train = np.log1p(y_train_raw)

    # 1. Визначаємо простір пошуку гіперпараметрів
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": RANDOM_SEED,
        "verbose": -1,
    }

    # 2. Налаштовуємо крос-валідацію
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    fold_mae = []

    for tr_idx, val_idx in kf.split(X_train):
        Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr, yval_raw = y_train.iloc[tr_idx], y_train_raw.iloc[val_idx]

        pipe = Pipeline([
            ("categorize", CategoricalCaster(CAT_FEATURES)),
            ("model", LGBMRegressor(**params))
        ])

        pipe.fit(Xtr, ytr)

        # Повертаємо прогноз у реальні долари для підрахунку MAE
        pred = np.expm1(pipe.predict(Xval))
        fold_mae.append(mean_absolute_error(yval_raw, pred))

    # Optuna буде мінімізувати середнє MAE по всіх фолдах
    return float(np.mean(fold_mae))


def main():
    # Створюємо study з напрямком 'minimize' (бо чим менше MAE, тим краще)
    study = optuna.create_study(direction="minimize", study_name="LightGBM_Tuning")

    print("Починаємо пошук гіперпараметрів...")
    study.optimize(objective, n_trials=100)  # Кількість спроб можна збільшити

    print("\n" + "=" * 50)
    print(f"Найкраще CV MAE: {study.best_value:.1f}")
    print("Найкращі гіперпараметри:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()