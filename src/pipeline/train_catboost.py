import numpy as np
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline

from common import CAT_FEATURES, RANDOM_SEED, evaluate, load_train_test, save_result


def build_pipeline() -> Pipeline:
    """CatBoost сам вміє працювати з сирими строковими категоріями -
    тому тут Pipeline складається з одного кроку. Він однаково потрібен
    заради єдиного інтерфейсу fit/predict з рештою моделей (і щоб
    cross-validation.py міг тренувати всі чотири моделі однаково)."""
    return Pipeline([
        ("model", CatBoostRegressor(
            iterations=600, depth=8, learning_rate=0.08,
            cat_features=CAT_FEATURES, random_seed=RANDOM_SEED, verbose=False,
        )),
    ])


def main():
    X_train, X_test, y_train_raw, y_test_raw = load_train_test()
    y_train = np.log1p(y_train_raw)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    pred_log = pipe.predict(X_test)
    metrics = evaluate(y_test_raw, pred_log)
    save_result("CatBoost", metrics)

    pipe.named_steps["model"].save_model("../../data/_tmp_catboost.cbm")


if __name__ == "__main__":
    main()
