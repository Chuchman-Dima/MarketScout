import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from common import CAT_FEATURES, PROJECT_ROOT, RANDOM_SEED, evaluate, load_train_test, save_result


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ], remainder=SimpleImputer(strategy="median", keep_empty_features=True))

    return Pipeline([
        ("prep", preprocessor),
        ("model", DecisionTreeRegressor(max_depth=14, min_samples_leaf=5, random_state=RANDOM_SEED)),
    ])


def main():
    X_train, X_test, y_train_raw, y_test_raw = load_train_test()
    y_train = np.log1p(y_train_raw)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    pred_log = pipe.predict(X_test)
    metrics = evaluate(y_test_raw, pred_log)
    save_result("DecisionTree", metrics)

    out = PROJECT_ROOT / "models_results" / "decision_tree_pipeline.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out)
    print(f"Збережено модель: {out}")


if __name__ == "__main__":
    main()
