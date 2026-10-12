import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from common import (
    CAT_FEATURES,
    PROJECT_ROOT,
    RANDOM_SEED,
    CategoricalCaster,
    evaluate,
    load_train_test,
    save_result,
)


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("categorize", CategoricalCaster(CAT_FEATURES)),
        ("model", XGBRegressor(
            n_estimators=600, max_depth=8, learning_rate=0.08,
            tree_method="hist", enable_categorical=True,
            random_state=RANDOM_SEED, verbosity=0,
        )),
    ])


def main():
    X_train, X_test, y_train_raw, y_test_raw = load_train_test()
    y_train = np.log1p(y_train_raw)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    pred_log = pipe.predict(X_test)
    metrics = evaluate(y_test_raw, pred_log)
    save_result("XGBoost", metrics)

    out = PROJECT_ROOT / "models_results" / "xgboost_pipeline.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out)
    print(f"Збережено модель: {out}")


if __name__ == "__main__":
    main()
