"""
Діагностика: чи реагує навчена модель на прапорці «ДТП / розмитнене / перший власник…».

Запуск (поруч із train_lightgbm.py, тобто там, де лежить common.py):
    python diagnose_flags.py
"""
import joblib
import numpy as np
import pandas as pd

from common import CAT_FEATURES, PROJECT_ROOT, UNKNOWN, load_train_test

FLAGS = ["Is_Crashed", "Custom", "First_Owner", "Exchange_Possible", "Is_Bargain", "Is_Urgent"]

# Колонки, які реально збирає форма в застосунку + похідні від них.
UI_COLS = {
    "Mark", "Model", "Mileage", "Gearbox", "Age", "Fuel_Type", "Engine_Capacity",
    "Body_Name", "Drive_Name", "Color_Name",
    "Km_per_Year", "is_EV", "is_suspicious_mileage", "is_new", "is_luxury_brand",
    "is_automatic_gearbox", "Engine_missing", "log_Mileage", "Age_x_Mileage", "Decade",
} | set(FLAGS)


def price(pipe, X):
    return np.expm1(pipe.predict(X))


def flag_effect(pipe, X, flag):
    """Середня зміна ціни, %, коли прапорець змінюється з 0 на 1 / з NaN на 1 / з NaN на 0."""
    out = {}
    preds = {}
    for label, val in (("NaN", np.nan), ("0", 0.0), ("1", 1.0)):
        d = X.copy()
        d[flag] = val
        preds[label] = price(pipe, d)
    out["0 → 1"] = np.mean(preds["1"] / preds["0"] - 1) * 100
    out["NaN → 1"] = np.mean(preds["1"] / preds["NaN"] - 1) * 100
    out["NaN → 0"] = np.mean(preds["0"] / preds["NaN"] - 1) * 100
    return out


def to_ui_like(X):
    """Лишає тільки те, що вводить користувач у формі; решта — як робить бекенд (NaN / «Не вказано»)."""
    d = X.copy()
    for c in d.columns:
        if c in UI_COLS:
            continue
        d[c] = UNKNOWN if c in CAT_FEATURES else np.nan
    return d


def main():
    X_train, X_test, y_train, _ = load_train_test()
    pipe = joblib.load(PROJECT_ROOT / "models_results" / "lightgbm_pipeline.pkl")
    model = pipe.named_steps["model"]

    print("=" * 70)
    print("1) Розподіл прапорців у train (частка рядків)")
    print("=" * 70)
    for f in FLAGS:
        vc = X_train[f].value_counts(dropna=False, normalize=True).round(4).to_dict()
        print(f"{f:20s} {vc}")

    print("\n" + "=" * 70)
    print("2) Важливість ознак за gain, % (місце серед усіх ознак)")
    print("=" * 70)
    names = list(model.feature_name_)
    gain = pd.Series(model.booster_.feature_importance("gain"), index=names)
    gain = (gain / gain.sum() * 100).sort_values(ascending=False)
    print("Топ-10:", {k: round(v, 2) for k, v in gain.head(10).items()})
    for f in FLAGS:
        print(f"{f:20s} {gain[f]:6.3f}%   місце {list(gain.index).index(f) + 1} з {len(gain)}")

    sample = X_test.sample(min(2000, len(X_test)), random_state=0)
    ui_sample = to_ui_like(sample)

    print("\n" + "=" * 70)
    print("3) Контрфактичний тест: на скільки % змінюється ціна від прапорця")
    print("   (A) реальні рядки тесту   (B) рядки «як у застосунку» — лише поля форми")
    print("=" * 70)
    rows = []
    for f in FLAGS:
        a = flag_effect(pipe, sample, f)
        b = flag_effect(pipe, ui_sample, f)
        rows.append({
            "Прапорець": f,
            "A: 0→1, %": round(a["0 → 1"], 2),
            "B: 0→1, %": round(b["0 → 1"], 2),
            "B: NaN→1, %": round(b["NaN → 1"], 2),
            "B: NaN→0, %": round(b["NaN → 0"], 2),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 70)
    print("4) Наскільки рядки «як у застосунку» відрізняються від реальних")
    print("=" * 70)
    p_real, p_ui = price(pipe, sample), price(pipe, ui_sample)
    print(f"Медіана прогнозу: реальні рядки = {np.median(p_real):,.0f} $, "
          f"рядки з форми = {np.median(p_ui):,.0f} $")
    print(f"Середня різниця: {np.mean(p_ui / p_real - 1) * 100:+.1f}%")
    print(f"Ознак, яких форма не збирає: {len(X_test.columns) - len(UI_COLS & set(X_test.columns))} "
          f"з {len(X_test.columns)}")

    print("\nПідказка до читання результатів:")
    print(" • майже 0% у колонці A  → модель сама майже не використовує прапорець (дані/ознаки);")
    print(" • помітно в A, але ~0 у B → на «порожніх» рядках з форми ефект гаситься (розрив train/inference);")
    print(" • 'NaN→0' ≈ 0 → для цього прапорця «Не вказано» і «Ні» для моделі те саме.")


if __name__ == "__main__":
    main()
