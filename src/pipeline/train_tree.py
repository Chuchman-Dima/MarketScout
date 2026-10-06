import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from common import CAT_FEATURES, RANDOM_SEED, evaluate, load_train_test, save_result


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ], remainder="passthrough")

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


if __name__ == "__main__":
    main()
