"""
K-Fold перевірка стабільності для трьох бустингів (DecisionTree тут не
беремо - це baseline, а не кандидат на деплой).

Було: `from train_catboost import CatBoostRegressor` - через відсутність
`if __name__ == "__main__":` у train_catboost.py цей імпорт запускав усе
тренування + запис у _results.json як побічний ефект. Тепер кожен
train_*.py має main-guard і просто надає build_pipeline() - тут імпортуємо
саме цю функцію, а не модель напряму.
"""

import json

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from common import PROJECT_ROOT, RANDOM_SEED, load_train_test
from train_catboost import build_pipeline as build_catboost
from train_lightgbm import build_pipeline as build_lightgbm
from train_xgboost import build_pipeline as build_xgboost

MODEL_BUILDERS = {
    "CatBoost": build_catboost,
    "LightGBM": build_lightgbm,
    "XGBoost": build_xgboost,
}


def main():
    X_train, _, y_train_raw, _ = load_train_test()
    y_train = np.log1p(y_train_raw)

    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    cv_results = {}

    for name, build_fn in MODEL_BUILDERS.items():
        fold_mae = []
        for tr_idx, val_idx in kf.split(X_train):
            Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            ytr, yval_raw = y_train.iloc[tr_idx], y_train_raw.iloc[val_idx]

            pipe = build_fn()  # свіжий пайплайн на кожен фолд - без витоку стану між фолдами
            pipe.fit(Xtr, ytr)
            pred = np.expm1(pipe.predict(Xval))
            fold_mae.append(mean_absolute_error(yval_raw, pred))

        cv_results[name] = {"mean": round(float(np.mean(fold_mae)), 1), "std": round(float(np.std(fold_mae)), 1)}
        print(f"{name}: CV MAE = {cv_results[name]['mean']} ± {cv_results[name]['std']}")

    cv_path = PROJECT_ROOT / "data" / "_cv_results.json"
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump(cv_results, f)
    print(f"Збережено {cv_path}")


if __name__ == "__main__":
    main()
