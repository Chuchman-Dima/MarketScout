"""
Auto Price Predictor — FastAPI Backend

Як рахується ціна (щоб не було «магії»):
  1. БАЗОВА ЦІНА — тільки модель LightGBM (lightgbm_pipeline.pkl).
     Їй передаються лише характеристики авто: марка, модель, пробіг, вік,
     коробка, пальне, об'єм, привід (+ похідні від них).
  2. КОРЕКТИВИ ЗА ПРАВИЛАМИ — ДТП, розмитнення, перший власник.
     У зібраних даних немає жодного прикладу цих ознак (див. аудит у README),
     тому модель навчитись на них не може. Це єдині «ручні» коефіцієнти в
     системі: вони зібрані в RULE_ADJUSTMENTS, повертаються окремим рядком у
     відповіді й показуються користувачу. Постав 0.0 — і правило вимкнене.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import common  # noqa: F401  — потрібен для joblib.load: у pickle є посилання на common.CategoricalCaster
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent


def _models_dir() -> Path:
    env = os.getenv("MODELS_DIR")
    if env:
        return Path(env)
    docker = Path("/app/models_results")
    if docker.is_dir():
        return docker
    if len(BASE_DIR.parents) > 1:
        return BASE_DIR.parents[1] / "models_results"
    return BASE_DIR / "models_results"


MODELS_DIR = _models_dir()
MODEL_PATH = MODELS_DIR / "lightgbm_pipeline.pkl"
CATEGORIES_PATH = MODELS_DIR / "valid_categories.json"
CATEGORIES_PKL_PATH = MODELS_DIR / "valid_categories.pkl"
DEFAULTS_PATH = MODELS_DIR / "car_defaults.json"
META_PATH = MODELS_DIR / "model_meta.json"  # пише train_final_model.py

# ─────────────────────────────────────────────
# ЛОГУВАННЯ
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("auto_price")

CURRENT_YEAR = 2026
UNKNOWN = "Не вказано"
INVALID_NAMES = {"", UNKNOWN, "Unknown", "None", "nan"}

LUXURY_MARKS = {
    "Aston Martin", "BMW-Alpina", "Lamborghini", "Rolls-Royce", "Ferrari",
    "Bentley", "Porsche", "Maserati", "Lexus", "Land Rover", "Jaguar",
    "Mercedes-Benz", "BMW", "Audi", "Lucid", "Genesis",
}
AUTOMATIC_LIKE = {"Автомат", "Типтронік", "Варіатор", "Робот"}
DRIVE_TYPES = [UNKNOWN, "Передній", "Задній", "Повний"]

# ─────────────────────────────────────────────
# ЄДИНІ «РУЧНІ» КОЕФІЦІЄНТИ (частка від базової ціни моделі)
# ─────────────────────────────────────────────
# Значення — ті самі, що були в попередній версії проєкту. Це експертні
# припущення, а не результат навчання: відкалібруй їх, коли з'являться дані.
RULE_ADJUSTMENTS = {
    "crashed": -0.15,       # після ДТП
    "not_customs": -0.20,   # нерозмитнене авто
    "first_owner": 0.03,    # перший власник
}
RULE_LABELS = {
    "crashed": "Після ДТП",
    "not_customs": "Нерозмитнене",
    "first_owner": "Перший власник",
}

# ─────────────────────────────────────────────
# СУМІСНІСТЬ ЗІ СТАРОЮ 50-ОЗНАКОВОЮ МОДЕЛЛЮ
# ─────────────────────────────────────────────
# Якщо у завантаженій моделі лишились ознаки, яких форма не збирає (кількість
# фото, обмін, аукціон…), ми не лишаємо їх порожніми: порожнє значення для
# LightGBM = «0 фото», і ціна зсувається (на репліці моделі ≈ +20%). Підставляємо
# типове значення. Нова «slim»-модель (train_final_model.py) цих ознак не має —
# тоді цей словник просто не використовується.
LEGACY_NEUTRAL: dict[str, Any] = {
    **{k: 0.0 for k in (
        "Is_Crashed", "Custom", "First_Owner", "Is_Bargain", "Is_Urgent", "Is_Leasing",
        "Has_VIN", "Is_Checked_VIN", "VIN_Has_Restrictions", "Has_Plate", "Is_Checked_Plate",
        "Is_Dealer", "Phone_Verified", "Desc_Ideal", "SeatsNumber", "DoorsNumber",
        "Description_Length", "Options_Count", "Exchange_Possible", "With_Video",
    )},
    "Photos_Count": 16.0,        # медіана new_cars_dataset_2.csv
    "Auction_Possible": 1.0,     # мода (≈60% оголошень)
    "Exchange_Type": "Будь-який",
}

# Фіксована сітка пробігів (тис. км) для захисту «ціна не росте разом із пробігом».
# Сітка НЕ залежить від введеного пробігу — тому для будь-якого авто виходить одна
# спадна крива, з якої ціна береться інтерполяцією (інакше різні пробіги давали б
# різні криві, і між ними ціна все одно могла б стрибнути вгору).
MILEAGE_GRID = np.unique(np.r_[
    np.arange(0, 100, 5), np.arange(100, 300, 10), np.arange(300, 1000, 50), np.arange(1000, 2001, 100),
]).astype(float)

# Запасна «типова похибка» (MAPE LightGBM з results_log.csv), якщо немає model_meta.json
FALLBACK_MAPE_PCT = 16.4

# ─────────────────────────────────────────────
# СТАН ЗАСТОСУНКУ
# ─────────────────────────────────────────────
model: Pipeline | None = None
explainer: shap.TreeExplainer | None = None
categories: dict = {}
car_defaults: dict = {}
model_meta: dict = {}

FEATURES: list[str] = []              # у тому ж порядку, що й при навчанні
CAT_COLS: list[str] = []
KNOWN_CATS: dict[str, set[str]] = {}  # категорії, які бачила модель
MODEL_IS_MONOTONE = False


def _strip_trailer(categories_data: dict) -> dict:
    if "valid_marks" in categories_data:
        categories_data["valid_marks"] = [
            m for m in categories_data["valid_marks"] if m != "Причеп"
        ]
    for key in ("mark_model_mapping", "engine_mapping", "fuel_mapping", "gearbox_mapping"):
        if key in categories_data:
            categories_data[key].pop("Причеп", None)
    return categories_data


def load_categories_data() -> dict:
    if CATEGORIES_PATH.is_file():
        with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    if not data.get("mark_model_mapping") and CATEGORIES_PKL_PATH.is_file():
        legacy = joblib.load(CATEGORIES_PKL_PATH)
        for key in (
                "valid_marks", "valid_models", "mark_model_mapping",
                "engine_mapping", "fuel_mapping", "gearbox_mapping",
        ):
            data.setdefault(key, legacy.get(key, {} if "mapping" in key else []))

    return _strip_trailer(data)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, explainer, categories, car_defaults, model_meta
    global FEATURES, CAT_COLS, KNOWN_CATS, MODEL_IS_MONOTONE

    log.info("Завантаження моделі…")
    try:
        model = joblib.load(MODEL_PATH)
        explainer = shap.TreeExplainer(model.named_steps["model"])
        # Список ознак беремо З САМОЇ МОДЕЛІ — бекенд працює і зі старою
        # 50-ознаковою, і з новою slim-моделлю без правок коду.
        FEATURES = list(model.named_steps["model"].feature_name_)
        caster = model.named_steps["categorize"]
        CAT_COLS = list(caster.columns)
        KNOWN_CATS = {c: {str(x) for x in cats} for c, cats in caster.categories_.items()}
        log.info(f"Модель завантажена: {len(FEATURES)} ознак.")
    except Exception as e:
        log.error(f"Помилка завантаження моделі: {e}")
        model = None
        explainer = None

    log.info("Завантаження категорій…")
    try:
        categories = load_categories_data()
    except Exception as e:
        log.error(f"Помилка завантаження категорій: {e}")
        categories = {}

    try:
        if DEFAULTS_PATH.is_file():
            with open(DEFAULTS_PATH, "r", encoding="utf-8") as f:
                car_defaults = json.load(f)
            log.info("Значення за замовчуванням (car_defaults) успішно завантажено.")
    except Exception as e:
        log.error(f"Помилка завантаження car_defaults: {e}")

    try:
        if META_PATH.is_file():
            with open(META_PATH, "r", encoding="utf-8") as f:
                model_meta = json.load(f)
            MODEL_IS_MONOTONE = bool(model_meta.get("monotone_mileage"))
            log.info(f"model_meta.json завантажено (монотонна за пробігом: {MODEL_IS_MONOTONE}).")
        else:
            log.warning("model_meta.json не знайдено — діапазон ціни буде приблизним, "
                        "а монотонність за пробігом забезпечуватиме бекенд.")
    except Exception as e:
        log.error(f"Помилка завантаження model_meta: {e}")

    yield

    log.info("Зупинка сервера.")


app = FastAPI(
    title="Auto Price Predictor API",
    version="2.2.0",
    description="ML-бекенд для прогнозу ринкової ціни автомобілів.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Необроблена помилка: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Внутрішня помилка сервера."})


# ─────────────────────────────────────────────
# СХЕМИ ДАНИХ
# ─────────────────────────────────────────────
class CarFeatures(BaseModel):
    """Лише те, що користувач реально може ввести у формі.
    Зайві поля зі старих клієнтів (кузов, колір, обмін, торг…) ігноруються."""
    Mark: str = Field(..., min_length=1)
    Model: str = Field(..., min_length=1)
    Mileage: float = Field(..., ge=0, le=2_000)
    Age: int = Field(..., ge=0, le=60)

    Gearbox: str | None = Field(default=None)
    Fuel_Type: str | None = Field(default=None)
    Engine_Capacity: float | None = Field(default=None, ge=0, le=20)
    Drive_Name: str | None = Field(default=None)

    # Лише для правил-корективів (в модель НЕ потрапляють)
    Is_Crashed: bool | None = Field(default=None)
    Custom: bool | None = Field(default=None)
    First_Owner: bool | None = Field(default=None)

    @field_validator("Engine_Capacity", "Is_Crashed", "Custom", "First_Owner", mode="before")
    @classmethod
    def preprocess_unknowns(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() in (UNKNOWN, "", "Unknown", "NaN", "nan", "None"):
            return None
        return v

    @field_validator("Mileage", "Engine_Capacity")
    @classmethod
    def must_be_finite(cls, v: float | None) -> float | None:
        if v is not None and not np.isfinite(v):
            raise ValueError("Значення має бути скінченним числом.")
        return v


class DepreciationRequest(BaseModel):
    car: CarFeatures
    annual_mileage: float = Field(..., ge=0, le=500)
    years: int = Field(default=5, ge=1, le=20)


# ─────────────────────────────────────────────
# ДОПОМІЖНІ ФУНКЦІЇ
# ─────────────────────────────────────────────
def process_prediction(raw_value: float) -> float:
    price = np.expm1(raw_value) if raw_value < 50 else float(raw_value)
    if not np.isfinite(price) or price <= 0:
        return 0.0
    return round(float(price), 2)


def _as_cat(v, *, mark_or_model: bool = False) -> str:
    if v is None:
        return UNKNOWN
    s = str(v).strip()
    if s in ("", "None", "nan", "NaN", "Unknown"):
        return UNKNOWN
    if mark_or_model and s in ("Інша", "Інше", "Інший"):
        return "Other"
    return s


def _to_known(col: str, value: str) -> str:
    """Рідкісне значення, якого модель не бачила на навчанні, у train було
    згруповане в «Other» — робимо те саме й тут (а не віддаємо NaN)."""
    known = KNOWN_CATS.get(col)
    if known and value not in known and "Other" in known:
        return "Other"
    return value


def _ensure_make_model(car: CarFeatures) -> None:
    """Ввічливе нагадування замість технічної помилки."""
    if car.Mark.strip() in INVALID_NAMES or car.Model.strip() in INVALID_NAMES:
        raise HTTPException(
            status_code=422,
            detail="Будь ласка, оберіть марку та модель авто — без цього ми не зможемо зробити оцінку.",
        )


def build_row(car: dict, assumed: dict | None = None) -> dict:
    """Один рядок ознак для моделі. Усе, що користувач не вказав, береться
    з типових значень для цієї моделі авто (car_defaults.json) — і фіксується
    в `assumed`, щоб інтерфейс міг це показати."""
    raw_mark = _as_cat(car.get("Mark"), mark_or_model=True)
    raw_model = _as_cat(car.get("Model"), mark_or_model=True)
    by_model = car_defaults.get("by_model", {}).get(f"{raw_mark}___{raw_model}")
    defaults = by_model or car_defaults.get("global", {})
    global_defaults = car_defaults.get("global", {})
    assumed = assumed if assumed is not None else {}

    out: dict[str, Any] = {
        "Mark": _to_known("Mark", raw_mark),
        "Model": _to_known("Model", raw_model),
    }

    for col, label in (("Fuel_Type", "Пальне"), ("Gearbox", "Коробка"), ("Drive_Name", "Привід")):
        val = _as_cat(car.get(col))
        if val == UNKNOWN:
            val = _as_cat(defaults.get(col, global_defaults.get(col)))
            if val != UNKNOWN:
                assumed[label] = val
        out[col] = val

    is_ev = out["Fuel_Type"] == "Електро"
    engine = car.get("Engine_Capacity")
    engine = float(engine) if engine is not None and np.isfinite(engine) else 0.0
    if is_ev:
        engine = 0.0
    elif engine <= 0:
        fallback = defaults.get("Engine_Capacity", global_defaults.get("Engine_Capacity"))
        if fallback and fallback > 0:
            engine = float(fallback)
            assumed["Об'єм двигуна"] = f"{engine:g} л"
    out["Engine_Capacity"] = engine
    # Як у навчанні (prepare): Engine_missing = 1 лише коли об'єм справді 0
    out["Engine_missing"] = int(engine <= 0)

    mileage = float(car["Mileage"])  # 0 км — коректне значення, а не «пропуск»
    age = int(car["Age"])
    out["Mileage"] = mileage
    out["Age"] = age
    out["Km_per_Year"] = mileage / (age + 1)
    out["is_EV"] = int(is_ev)
    out["is_suspicious_mileage"] = int(age > 10 and mileage < 50)
    out["is_new"] = int(age <= 3)
    out["is_luxury_brand"] = int(raw_mark in LUXURY_MARKS)
    out["is_automatic_gearbox"] = int(out["Gearbox"] in AUTOMATIC_LIKE)
    out["log_Mileage"] = float(np.log1p(mileage))
    out["Age_x_Mileage"] = age * mileage
    out["Decade"] = int((CURRENT_YEAR - age) // 10 * 10)

    # Усе, чого модель очікує, а форма не збирає (лише для старої моделі)
    for feat in FEATURES:
        if feat not in out:
            out[feat] = LEGACY_NEUTRAL.get(feat, UNKNOWN if feat in CAT_COLS else np.nan)
    return out


def _predict_prices(cars: list[dict]) -> np.ndarray:
    df = pd.DataFrame([build_row(c) for c in cars])[FEATURES]
    return np.array([process_prediction(r) for r in model.predict(df)])


def base_price(car: dict) -> float:
    """Ціна моделі. Ціна не може рости разом із пробігом: якщо модель цього сама
    не гарантує (стара модель), рахуємо її на фіксованій сітці пробігів,
    вирівнюємо ізотонічною регресією (спадна функція) і беремо значення
    для введеного пробігу інтерполяцією."""
    if MODEL_IS_MONOTONE:
        return float(_predict_prices([car])[0])

    prices = _predict_prices([{**car, "Mileage": float(g)} for g in MILEAGE_GRID])
    if (prices <= 0).any():
        return float(_predict_prices([car])[0])
    fitted = IsotonicRegression(increasing=False).fit_transform(MILEAGE_GRID, np.log(prices))
    return round(float(np.exp(np.interp(float(car["Mileage"]), MILEAGE_GRID, fitted))), 2)


def apply_rules(car: dict, price: float) -> tuple[float, dict[str, float]]:
    """Прозорі корективи за правилами. Повертає (фінальна ціна, {назва: $})."""
    adjustments: dict[str, float] = {}
    if car.get("Is_Crashed") is True:
        adjustments[RULE_LABELS["crashed"]] = round(price * RULE_ADJUSTMENTS["crashed"], 1)
    if car.get("Custom") is False:
        adjustments[RULE_LABELS["not_customs"]] = round(price * RULE_ADJUSTMENTS["not_customs"], 1)
    if car.get("First_Owner") is True:
        adjustments[RULE_LABELS["first_owner"]] = round(price * RULE_ADJUSTMENTS["first_owner"], 1)
    adjustments = {k: v for k, v in adjustments.items() if v != 0}
    final = max(round(price + sum(adjustments.values()), 2), 0.0)
    return final, adjustments


def price_range(price: float) -> dict:
    """Діапазон з реальної похибки моделі на test-вибірці (а не фіксований ±5%)."""
    q10, q90 = model_meta.get("rel_error_q10"), model_meta.get("rel_error_q90")
    if q10 and q90:
        return {"price_low_usd": round(price * q10, 2), "price_high_usd": round(price * q90, 2),
                "range_kind": "interval80"}
    mape = model_meta.get("mape", FALLBACK_MAPE_PCT) / 100
    return {"price_low_usd": round(price * (1 - mape), 2), "price_high_usd": round(price * (1 + mape), 2),
            "range_kind": "mape", "range_pct": round(mape * 100, 1)}


SHAP_LABELS = {
    "Age": "Вік авто", "Mileage": "Пробіг", "Engine_Capacity": "Об'єм двигуна",
    "Km_per_Year": "Км на рік", "Fuel_Type": "Тип пального", "Gearbox": "Коробка передач",
    "Mark": "Марка", "Model": "Модель", "Drive_Name": "Привід",
    "is_EV": "Електро", "is_suspicious_mileage": "Підозрілий пробіг", "is_new": "Нове авто (≤3 р.)",
    "is_luxury_brand": "Преміум-марка", "Engine_missing": "Об'єм не вказано",
    "log_Mileage": "Пробіг (log)", "Age_x_Mileage": "Вік × пробіг", "Decade": "Десятиліття випуску",
    "is_automatic_gearbox": "Автоматична КПП",
}


def compute_shap(car: dict, predicted_price: float) -> dict:
    try:
        df = pd.DataFrame([build_row(car)])[FEATURES]
        X = model.named_steps["categorize"].transform(df)
        shap_row = explainer.shap_values(X)[0]
        # Ознаки, яких користувач не вводив (підставлені типові значення старої
        # моделі), у поясненні не показуємо — це не характеристики його авто.
        shap_dict = {
            SHAP_LABELS.get(f, f): round(float(v) * predicted_price, 1)
            for f, v in zip(X.columns, shap_row) if f not in LEGACY_NEUTRAL
        }
        return dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:6])
    except Exception as e:
        log.warning(f"SHAP недоступний: {e}")
        return {}


def _model_ready() -> None:
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не готова. Спробуйте пізніше.")


# ─────────────────────────────────────────────
# ЕНДПОІНТИ
# ─────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "ok",
        "model": "loaded" if model is not None else "not_loaded",
        "categories": "loaded" if categories else "empty",
    }


@app.get("/categories", tags=["Data"])
def get_categories():
    if not categories:
        raise HTTPException(status_code=404, detail="Категорії не знайдені.")
    return {
        **categories,
        "drive_types": categories.get("drive_types") or DRIVE_TYPES,
        "rule_adjustments": RULE_ADJUSTMENTS,
    }


@app.post("/predict", tags=["Prediction"])
def predict_price(car: CarFeatures):
    _model_ready()
    _ensure_make_model(car)

    try:
        car_dict = car.model_dump()
        base = base_price(car_dict)
        if base == 0.0:
            raise HTTPException(status_code=422, detail="Не вдалося обчислити ціну для цих параметрів.")

        final, adjustments = apply_rules(car_dict, base)
        assumed: dict = {}
        build_row(car_dict, assumed)

        log.info(f"Predict: {car.Mark} {car.Model} {car.Age}р {car.Mileage}тис → "
                 f"модель ${base:,.0f}, з корективами ${final:,.0f}")

        return {
            "predicted_price_usd": final,
            "base_ml_price_usd": base,
            "price_adjustments": adjustments,
            "assumed_defaults": assumed,
            "shap_values": compute_shap(car_dict, base),
            **price_range(final),
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Помилка predict: {e}")
        raise HTTPException(status_code=500, detail=f"Помилка обчислення: {e}")


@app.post("/predict_depreciation", tags=["Prediction"])
def predict_depreciation(req: DepreciationRequest):
    _model_ready()
    _ensure_make_model(req.car)

    try:
        base = req.car.model_dump()
        predictions = []

        for year_offset in range(req.years + 1):
            current = base.copy()
            current["Age"] += year_offset
            current["Mileage"] += req.annual_mileage * year_offset

            price, _ = apply_rules(base, base_price(current))

            # Ціна не може рости з часом
            if predictions and price > predictions[-1]["Price"]:
                price = predictions[-1]["Price"]

            predictions.append({"Year": year_offset, "Price": price})

        total_loss = predictions[0]["Price"] - predictions[-1]["Price"]
        return {"depreciation": predictions, "total_loss_usd": round(total_loss, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка розрахунку: {e}")


@app.post("/predict_batch", tags=["Prediction"])
def predict_batch(cars: list[CarFeatures]):
    _model_ready()
    if len(cars) > 20:
        raise HTTPException(status_code=400, detail="Максимум 20 авто за запит.")

    results = []
    for car in cars:
        try:
            _ensure_make_model(car)
            car_dict = car.model_dump()
            final, adjustments = apply_rules(car_dict, base_price(car_dict))
            results.append({"mark": car.Mark, "model": car.Model,
                            "predicted_price_usd": final, "price_adjustments": adjustments})
        except HTTPException as e:
            results.append({"mark": car.Mark, "model": car.Model, "error": e.detail})
        except Exception as e:
            results.append({"mark": car.Mark, "model": car.Model, "error": str(e)})

    return {"results": results}