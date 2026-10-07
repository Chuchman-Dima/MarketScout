"""
Будує valid_categories.json для UI (марки, моделі, мапінги КПП/паливо/об'єм)
з того ж CSV, що й prepare.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42  # noqa: F401 — для узгодженості з prepare.py
CURRENT_YEAR = 2026
MIN_COUNT = 10

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "new_data" / "new_cars_dataset_2.csv"
OUT_PATH = PROJECT_ROOT / "models_results" / "valid_categories.json"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["CategoryId"].isin([0, 1])].copy()
    df["Engine_Volume"] = df["Fuel_Name"].str.extract(r"(\d+\.?\d*)").astype(float)
    df["Fuel_Type"] = (
        df["Fuel_Name"]
        .str.replace(r",?\s*\d+\.?\d*\s*л\.?", "", regex=True)
        .str.strip()
    )
    df["Fuel_Type"] = df["Fuel_Type"].replace("", np.nan)
    df["Age"] = CURRENT_YEAR - df["Year"]
    df = df.rename(
        columns={
            "Mileage_K": "Mileage",
            "Gearbox_Name": "Gearbox",
            "Engine_Volume": "Engine_Capacity",
        }
    )
    df = df[df["Mark"] != "Причеп"].copy()
    df["Mark"] = df["Mark"].fillna("Other")
    df["Model"] = df["Model"].fillna("Other")
    df["Gearbox"] = df["Gearbox"].fillna("Unknown")
    df["Fuel_Type"] = df["Fuel_Type"].fillna("Не вказано")
    df["Engine_Capacity"] = df["Engine_Capacity"].fillna(0)
    return df


def build_categories(df: pd.DataFrame) -> dict:
    mark_counts = df["Mark"].value_counts()
    model_counts = df["Model"].value_counts()
    valid_marks = mark_counts[mark_counts >= MIN_COUNT].index.tolist()
    valid_models = model_counts[model_counts >= MIN_COUNT].index.tolist()

    mark_model_mapping: dict[str, list[str]] = {}
    engine_mapping: dict[str, dict[str, list[float]]] = {}
    fuel_mapping: dict[str, dict[str, list[str]]] = {}
    gearbox_mapping: dict[str, dict[str, list[str]]] = {}

    for mark in valid_marks:
        sub = df[df["Mark"] == mark]
        models = sorted(sub["Model"].unique().tolist())
        mark_model_mapping[mark] = models

        engine_mapping[mark] = {}
        fuel_mapping[mark] = {}
        gearbox_mapping[mark] = {}

        for model in models:
            msub = sub[sub["Model"] == model]
            caps = sorted({round(float(x), 1) for x in msub["Engine_Capacity"].unique() if x > 0})
            fuels = sorted(msub["Fuel_Type"].unique().tolist())
            gearboxes = sorted(msub["Gearbox"].unique().tolist())
            engine_mapping[mark][model] = caps
            fuel_mapping[mark][model] = fuels
            gearbox_mapping[mark][model] = gearboxes

    return {
        "valid_marks": sorted(valid_marks),
        "valid_models": sorted(valid_models),
        "mark_model_mapping": mark_model_mapping,
        "engine_mapping": engine_mapping,
        "fuel_mapping": fuel_mapping,
        "gearbox_mapping": gearbox_mapping,
    }


def main() -> None:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Не знайдено датасет: {CSV_PATH}")

    raw = pd.read_csv(CSV_PATH, low_memory=False)
    df = _clean(raw)
    payload = build_categories(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Збережено {OUT_PATH} ({len(payload['valid_marks'])} марок)")


if __name__ == "__main__":
    main()
