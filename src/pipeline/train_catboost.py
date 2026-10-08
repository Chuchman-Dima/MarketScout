import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from common import CAT_FEATURES, PROJECT_ROOT, RANDOM_SEED, UNKNOWN, evaluate, load_train_test, save_result


class FillCategorical(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X[CAT_FEATURES] = X[CAT_FEATURES].astype(object).fillna(UNKNOWN).astype(str)
        return X


def build_pipeline() -> Pipeline:
    """CatBoost сам вміє працювати з сирими строковими категоріями."""
    return Pipeline([
        ("fill_cats", FillCategorical()),
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

    out_dir = PROJECT_ROOT / "models_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe.named_steps["model"].save_model(str(out_dir / "catboost_model.cbm"))
    print(f"Збережено модель: {out_dir / 'catboost_model.cbm'}")


if __name__ == "__main__":
    main()
