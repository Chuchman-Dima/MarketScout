"""
Збирає докупи все, що написали train_*.py (_results.json) і
cross-validation.py (_cv_results.json), і показує одну зведену таблицю.

Запускати ПІСЛЯ того, як прогнав усі train_*.py і cross-validation.py:
    python prepare.py
    python train_catboost.py
    python train_lightgbm.py
    python train_xgboost.py
    python train_tree.py
    python cross-validation.py
    python compare_results.py
"""

import json

import pandas as pd

RESULTS_PATH = "../../models_results/_results.json"
CV_RESULTS_PATH = "../../data/_cv_results.json"
LOG_PATH = "../../models_results/results_log.csv"


def main():
    rows = []
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    test_df = pd.DataFrame(rows).drop_duplicates(subset="model", keep="last")

    try:
        with open(CV_RESULTS_PATH, "r", encoding="utf-8") as f:
            cv = json.load(f)
        cv_df = pd.DataFrame([
            {"model": name, "CV_MAE_mean": v["mean"], "CV_MAE_std": v["std"]}
            for name, v in cv.items()
        ])
        combined = test_df.merge(cv_df, on="model", how="left")
    except FileNotFoundError:
        print("_cv_results.json не знайдено - показую лише test-метрики "
              "(запусти cross-validation.py для повної картини).")
        combined = test_df

    combined = combined.sort_values("MAE").reset_index(drop=True)

    print("\n" + "=" * 72)
    print("  ПОРІВНЯННЯ МОДЕЛЕЙ (відсортовано за MAE на test, краще зверху)")
    print("=" * 72)
    print(combined.to_string(index=False))
    print("=" * 72)

    winner = combined.iloc[0]
    print(f"\nПереможець за MAE: {winner['model']} "
          f"(MAE={winner['MAE']}, MAPE={winner['MAPE']}%, R²={winner['R2']})")

    if "CV_MAE_std" in combined.columns and pd.notna(winner.get("CV_MAE_std")):
        max_std = combined["CV_MAE_std"].max()
        if winner["CV_MAE_std"] == max_std and len(combined) > 1:
            print("У переможця найбільший розкид по фолдах CV - "
                  "перевір стабільність перед тим, як остаточно обирати цю модель.")

    combined.to_csv(LOG_PATH, index=False)
    print(f"\nЗбережено: {LOG_PATH}")


if __name__ == "__main__":
    main()
