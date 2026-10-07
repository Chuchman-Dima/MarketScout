"""
Будує valid_categories.json для UI (марки, моделі, мапінги КПП/паливо/об'єм)
з того ж CSV, що й prepare.py.
"""

import json
from pathlib import Path

import pandas as pd

from prepare import MIN_COUNT, UNKNOWN, clean_raw_dataframe, resolve_csv_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "models_results" / "valid_categories.json"


def _uniq_list(series, extra_other: str | None = None) -> list[str]:
    vals = []
    for x in series.dropna().unique().tolist():
        s = str(x).strip()
        if not s or s in (UNKNOWN, "Unknown", "Other", "nan", "None"):
            continue
        if s not in vals:
            vals.append(s)
    vals = sorted(vals)
    out = [UNKNOWN] + vals
    if extra_other and extra_other not in out:
        out.append(extra_other)
    return out


def build_categories(df) -> dict:
    mark_counts = df["Mark"].value_counts()
    model_counts = df["Model"].value_counts()
    valid_marks = [m for m in mark_counts[mark_counts >= MIN_COUNT].index.tolist() if m not in (UNKNOWN, "Other")]
    valid_models = [m for m in model_counts[model_counts >= MIN_COUNT].index.tolist() if m not in (UNKNOWN, "Other")]

    mark_model_mapping: dict[str, list[str]] = {}
    engine_mapping: dict[str, dict[str, list[float]]] = {}
    fuel_mapping: dict[str, dict[str, list[str]]] = {}
    gearbox_mapping: dict[str, dict[str, list[str]]] = {}

    for mark in valid_marks:
        sub = df[df["Mark"] == mark]
        models = sorted(
            m for m in sub["Model"].unique().tolist()
            if m not in (UNKNOWN, "Other")
        )
        mark_model_mapping[mark] = models

        engine_mapping[mark] = {}
        fuel_mapping[mark] = {}
        gearbox_mapping[mark] = {}

        for model in models:
            msub = sub[sub["Model"] == model]
            caps = sorted({
                round(float(x), 1)
                for x in msub["Engine_Capacity"].dropna().unique()
                if float(x) > 0
            })
            fuels = _uniq_list(msub["Fuel_Type"])
            gearboxes = _uniq_list(msub["Gearbox"])
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
        "body_types": _uniq_list(df["Body_Name"], extra_other="Інше"),
        "drive_types": _uniq_list(df["Drive_Name"]),
        "color_names": _uniq_list(df["Color_Name"], extra_other="Інший"),
    }


def main() -> None:
    csv_path = resolve_csv_path()
    print(f"Читаємо {csv_path}")
    raw = pd.read_csv(csv_path, low_memory=False)
    df = clean_raw_dataframe(raw)
    payload = build_categories(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Збережено {OUT_PATH} ({len(payload['valid_marks'])} марок)")


if __name__ == "__main__":
    main()
