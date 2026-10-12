"""
Фінальне навчання LightGBM для AUREA.

Що змінено порівняно з попередньою 50-ознаковою моделлю (причини — у аудиті):
  1. Лише характеристики АВТО (18 ознак). Прибрано те, що в даних порожнє/константне
     (ДТП, перший власник, торг, терміновість, VIN, дилер… — 100% нулі) і те, що є
     властивістю ОГОЛОШЕННЯ, а не авто (кількість фото, обмін, аукціон, відео, місто):
     форма цього не збирає, а модель на них «підглядала» в train.
  2. Монотонність за пробігом: ціна не може зростати разом із пробігом.
     LightGBM гарантує це на рівні дерев (monotone_constraints).
  3. Поруч із моделлю пишеться model_meta.json: метрики, реальний діапазон похибки
     (замість фіксованого «±5%») і прапорець monotone_mileage для бекенду.

Запуск — з папки, де лежить common.py (CategoricalCaster береться звідти, як і в
поточному пайплайні, тому бекенд розпакує pickle без змін):
    python train_final_model.py
Результат: <корінь проєкту>/models_results/lightgbm_pipeline.pkl + model_meta.json
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from common import CategoricalCaster

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in [HERE, *HERE.parents] if (p / "data" / "new_data").is_dir()), None)
if ROOT is None:
    raise SystemExit("Не знайшов папку data/new_data вище за цей скрипт — перевір розташування.")

CSV_PATH = ROOT / "data" / "new_data" / "new_cars_dataset_2.csv"
OUT_DIR = ROOT / "models_results"

RANDOM_SEED = 42
CURRENT_YEAR = 2026
MIN_COUNT = 10
UNKNOWN = "Не вказано"

LUXURY_MARKS = {
    'Aston Martin', 'BMW-Alpina', 'Lamborghini', 'Rolls-Royce', 'Ferrari',
    'Bentley', 'Porsche', 'Maserati', 'Lexus', 'Land Rover', 'Jaguar',
    'Mercedes-Benz', 'BMW', 'Audi', 'Lucid', 'Genesis',
}
AUTOMATIC_LIKE = {'Автомат', 'Типтронік', 'Варіатор', 'Робот'}

FEATURES = [
    "Mark", "Model", "Mileage", "Gearbox", "Age", "Fuel_Type", "Engine_Capacity", "Drive_Name",
    "Km_per_Year", "is_EV", "is_suspicious_mileage", "is_new", "is_luxury_brand",
    "Engine_missing", "log_Mileage", "Age_x_Mileage", "Decade", "is_automatic_gearbox",
]
CAT_FEATURES = ["Mark", "Model", "Gearbox", "Fuel_Type", "Drive_Name"]

# -1: ціна не зростає разом з ознакою. Усі ці ознаки ростуть разом із пробігом.
# +1 для is_suspicious_mileage: при зростанні пробігу прапорець перемикається 1→0,
# тож щоб ціна не стрибала вгору, «підозріло малий пробіг» не може бути дешевшим.
MONOTONE = {"Mileage": -1, "log_Mileage": -1, "Km_per_Year": -1, "Age_x_Mileage": -1,
            "is_suspicious_mileage": 1}


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[df["CategoryId"].isin([0, 1])].copy()

    df["Engine_Capacity"] = df["Fuel_Name"].str.extract(r"(\d+\.?\d*)").astype(float)
    df["Fuel_Type"] = (df["Fuel_Name"].str.replace(r",?\s*\d+\.?\d*\s*л\.?", "", regex=True)
                       .str.strip().replace("", np.nan))
    df["Age"] = CURRENT_YEAR - df["Year"]
    df = df.rename(columns={"Mileage_K": "Mileage", "Gearbox_Name": "Gearbox"})
    df = df[df["Mark"] != "Причеп"].copy()

    df["Mark"] = df["Mark"].fillna("Other")
    df["Model"] = df["Model"].fillna("Other")
    df["Gearbox"] = df["Gearbox"].fillna("Unknown")
    df["Fuel_Type"] = df["Fuel_Type"].fillna(UNKNOWN)
    df["Drive_Name"] = df["Drive_Name"].fillna(UNKNOWN)
    df["Engine_Capacity"] = df["Engine_Capacity"].fillna(0)

    df = df[(df["Price_USD"] >= 1000) & (df["Price_USD"] <= 250000)
            & (df["Age"] >= 0) & (df["Age"] <= 46) & (df["Engine_Capacity"] <= 10)].copy()

    df["Km_per_Year"] = df["Mileage"] / (df["Age"] + 1)
    df["is_EV"] = (df["Fuel_Type"] == "Електро").astype(int)
    df["is_suspicious_mileage"] = ((df["Age"] > 10) & (df["Mileage"] < 50)).astype(int)
    df["is_new"] = (df["Age"] <= 3).astype(int)
    df["is_luxury_brand"] = df["Mark"].isin(LUXURY_MARKS).astype(int)
    df["is_automatic_gearbox"] = df["Gearbox"].isin(AUTOMATIC_LIKE).astype(int)
    df["Engine_missing"] = (df["Engine_Capacity"] == 0).astype(int)
    df["log_Mileage"] = np.log1p(df["Mileage"])
    df["Age_x_Mileage"] = df["Age"] * df["Mileage"]
    df["Decade"] = ((CURRENT_YEAR - df["Age"]) // 10 * 10).astype(int)
    return df


def check_monotonic(pipe: Pipeline, X: pd.DataFrame, n: int = 300) -> int:
    """Скільки авто мають хоч одне «зростання ціни при більшому пробігу» (має бути 0)."""
    sample = X.sample(min(n, len(X)), random_state=1).reset_index(drop=True)
    grid = np.arange(0, 305, 5.0)
    bad = 0
    for i in range(len(sample)):
        rows = pd.concat([sample.iloc[[i]]] * len(grid), ignore_index=True)
        rows["Mileage"] = grid
        rows["log_Mileage"] = np.log1p(grid)
        rows["Km_per_Year"] = grid / (rows["Age"] + 1)
        rows["Age_x_Mileage"] = rows["Age"] * grid
        rows["is_suspicious_mileage"] = ((rows["Age"] > 10) & (grid < 50)).astype(int)
        if (np.diff(np.expm1(pipe.predict(rows[FEATURES]))) > 1).any():
            bad += 1
    return bad


def main():
    df = load_clean()
    X, y_raw = df[FEATURES].copy(), df["Price_USD"]
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y_raw, test_size=0.2, random_state=RANDOM_SEED)

    # Рідкісні марки/моделі → "Other"; поріг рахується лише з train
    for col in ("Mark", "Model"):
        counts = X_train[col].value_counts()
        valid = counts[counts >= MIN_COUNT].index
        X_train[col] = np.where(X_train[col].isin(valid), X_train[col], "Other")
        X_test[col] = np.where(X_test[col].isin(valid), X_test[col], "Other")

    pipe = Pipeline([
        ("categorize", CategoricalCaster(CAT_FEATURES)),
        ("model", LGBMRegressor(
            n_estimators=600, max_depth=8, learning_rate=0.08,
            monotone_constraints=[MONOTONE.get(c, 0) for c in FEATURES],
            random_state=RANDOM_SEED, verbose=-1,
        )),
    ])
    pipe.fit(X_train, np.log1p(y_train_raw))

    pred = np.expm1(pipe.predict(X_test))
    mae = mean_absolute_error(y_test_raw, pred)
    mape = mean_absolute_percentage_error(y_test_raw, pred) * 100
    r2 = r2_score(y_test_raw, pred)
    ratio = y_test_raw.values / pred           # факт / прогноз
    q10, q90 = np.quantile(ratio, [0.10, 0.90])
    bad = check_monotonic(pipe, X_train)

    print(f"Рядків train/test: {len(X_train)}/{len(X_test)}")
    print(f"MAE={mae:.1f}  MAPE={mape:.2f}%  R2={r2:.4f}")
    print(f"80% фактичних цін лежать у [{q10:.2f}×; {q90:.2f}×] від прогнозу")
    print(f"Порушень монотонності за пробігом: {bad}/300 (очікується 0)")

    OUT_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, OUT_DIR / "lightgbm_pipeline.pkl")
    with open(OUT_DIR / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": "LightGBM",
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "features": FEATURES,
            "rows_train": int(len(X_train)),
            "mae": round(float(mae), 1), "mape": round(float(mape), 2), "r2": round(float(r2), 4),
            "rel_error_q10": round(float(q10), 4), "rel_error_q90": round(float(q90), 4),
            "monotone_mileage": bad == 0,
        }, f, ensure_ascii=False, indent=2)
    print(f"Збережено в {OUT_DIR}")


if __name__ == "__main__":
    main()