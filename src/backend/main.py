"""
Auto Price Predictor — FastAPI Backend
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import common  # noqa: F401
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
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

LUXURY_MARKS = {
    "Aston Martin", "BMW-Alpina", "Lamborghini", "Rolls-Royce", "Ferrari",
    "Bentley", "Porsche", "Maserati", "Lexus", "Land Rover", "Jaguar",
    "Mercedes-Benz", "BMW", "Audi", "Lucid", "Genesis",
}
AUTOMATIC_LIKE = {"Автомат", "Типтронік", "Варіатор", "Робот"}

BODY_TYPES = [
    "Не вказано", "Седан", "Позашляховик / Кросовер", "Хетчбек", "Універсал",
    "Мінівен", "Купе", "Пікап", "Інше",
]
DRIVE_TYPES = ["Не вказано", "Передній", "Задній", "Повний"]
COLOR_NAMES = [
    "Не вказано", "Чорний", "Білий", "Сірий", "Сріблястий", "Синій",
    "Червоний", "Зелений", "Інший",
]

MODEL_FEATURE_ORDER = [
    "Mark", "Model", "Modification", "Mileage", "Gearbox", "Age",
    "Fuel_Type", "Engine_Capacity", "Km_per_Year",
    "Body_Name", "Drive_Name", "Color_Name", "Wheel_Name",
    "SeatsNumber", "DoorsNumber",
    "Is_Crashed", "Custom", "First_Owner", "Is_Leasing",
    "Has_VIN", "Is_Checked_VIN", "VIN_Has_Restrictions",
    "Has_Plate", "Is_Checked_Plate",
    "Country_Origin_Id", "ConditionId", "State_Name", "City",
    "Is_Dealer", "Seller_Type", "Phone_Verified",
    "Exchange_Possible", "Exchange_Type",
    "Auction_Possible", "Is_Bargain", "Is_Urgent",
    "Photos_Count", "With_Video", "Description_Length", "Options_Count",
    "Desc_Ideal",
    "is_EV", "is_suspicious_mileage", "is_new",
    "is_luxury_brand", "Engine_missing", "log_Mileage", "Age_x_Mileage",
    "Decade", "is_automatic_gearbox",
]
MODEL_CAT_FEATURES = [
    "Mark", "Model", "Modification", "Gearbox", "Fuel_Type",
    "Body_Name", "Drive_Name", "Color_Name", "Wheel_Name",
    "Country_Origin_Id", "ConditionId",
    "State_Name", "City", "Seller_Type", "Exchange_Type",
]
BOOL_FEATURES = [
    "Is_Crashed", "Custom", "First_Owner", "Is_Leasing",
    "Has_VIN", "Is_Checked_VIN", "VIN_Has_Restrictions",
    "Has_Plate", "Is_Checked_Plate", "Is_Dealer", "Phone_Verified",
    "Exchange_Possible", "Auction_Possible", "Is_Bargain", "Is_Urgent",
    "With_Video", "Desc_Ideal",
]
NUM_OPTIONAL = [
    "SeatsNumber", "DoorsNumber", "Photos_Count", "Description_Length", "Options_Count",
]

# ─────────────────────────────────────────────
# ЗАВАНТАЖЕННЯ МОДЕЛІ / КАТЕГОРІЙ
# ─────────────────────────────────────────────
model: Pipeline | None = None
explainer: shap.TreeExplainer | None = None
categories: dict = {}
car_defaults: dict = {}


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
    global model, explainer, categories, car_defaults

    log.info("Завантаження моделі…")
    try:
        model = joblib.load(MODEL_PATH)
        explainer = shap.TreeExplainer(model.named_steps["model"])
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

    yield

    log.info("Зупинка сервера.")


app = FastAPI(
    title="Auto Price Predictor API",
    version="2.1.0",
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
    Mark: str = Field(..., min_length=1)
    Model: str = Field(..., min_length=1)
    Mileage: float = Field(..., ge=0, le=2_000)
    Age: int = Field(..., ge=0, le=60)

    Gearbox: str | None = Field(default=None)
    Fuel_Type: str | None = Field(default=None)
    Engine_Capacity: float | None = Field(default=None, ge=0, le=20)
    Km_per_Year: float | None = Field(default=None, ge=0)
    is_EV: int | None = Field(default=None, ge=0, le=1)
    is_suspicious_mileage: int | None = Field(default=None, ge=0, le=1)
    is_new: int | None = Field(default=None, ge=0, le=1)

    Modification: str | None = Field(default=None)
    Body_Name: str | None = Field(default=None)
    Drive_Name: str | None = Field(default=None)
    Color_Name: str | None = Field(default=None)
    Wheel_Name: str | None = Field(default=None)
    State_Name: str | None = Field(default=None)
    City: str | None = Field(default=None)
    Seller_Type: str | None = Field(default=None)
    Exchange_Type: str | None = Field(default=None)
    Country_Origin_Id: str | None = Field(default=None)
    ConditionId: str | None = Field(default=None)

    SeatsNumber: float | None = Field(default=None)
    DoorsNumber: float | None = Field(default=None)
    Photos_Count: float | None = Field(default=None)
    Description_Length: float | None = Field(default=None)
    Options_Count: float | None = Field(default=None)

    Is_Crashed: bool | None = Field(default=None)
    Custom: bool | None = Field(default=None)
    First_Owner: bool | None = Field(default=None)
    Is_Leasing: bool | None = Field(default=None)
    Has_VIN: bool | None = Field(default=None)
    Is_Checked_VIN: bool | None = Field(default=None)
    VIN_Has_Restrictions: bool | None = Field(default=None)
    Has_Plate: bool | None = Field(default=None)
    Is_Checked_Plate: bool | None = Field(default=None)
    Is_Dealer: bool | None = Field(default=None)
    Phone_Verified: bool | None = Field(default=None)
    Exchange_Possible: bool | None = Field(default=None)
    Auction_Possible: bool | None = Field(default=None)
    Is_Bargain: bool | None = Field(default=None)
    Is_Urgent: bool | None = Field(default=None)
    With_Video: bool | None = Field(default=None)
    Desc_Ideal: bool | None = Field(default=None)

    @field_validator(
        "Engine_Capacity", "Km_per_Year", "SeatsNumber", "DoorsNumber",
        "Photos_Count", "Description_Length", "Options_Count",
        "is_EV", "is_suspicious_mileage", "is_new",
        "Is_Crashed", "Custom", "First_Owner", "Is_Leasing",
        "Has_VIN", "Is_Checked_VIN", "VIN_Has_Restrictions",
        "Has_Plate", "Is_Checked_Plate", "Is_Dealer", "Phone_Verified",
        "Exchange_Possible", "Auction_Possible", "Is_Bargain", "Is_Urgent",
        "With_Video", "Desc_Ideal",
        mode="before"
    )
    @classmethod
    def preprocess_unknowns(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() in ("Не вказано", "", "Unknown", "NaN", "nan", "None"):
            return None
        return v

    @field_validator("Mileage", "Engine_Capacity", "Km_per_Year", "SeatsNumber", "DoorsNumber")
    @classmethod
    def must_be_finite(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if not np.isfinite(v):
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


def _as_num(v, *, zero_as_missing: bool = False):
    if v is None or v == "":
        return np.nan
    try:
        x = float(v)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(x):
        return np.nan
    if zero_as_missing and x <= 0:
        return np.nan
    return x


def _as_bool01(v):
    """Надійна конвертація булевих значень з урахуванням пустот."""
    if v is None:
        return np.nan
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("", "не вказано", "unknown", "nan", "none"):
            return np.nan
        if s in ("0", "false", "ні", "no"):
            return 0.0
        if s in ("1", "true", "так", "yes"):
            return 1.0
    # Якщо це справжній bool (True/False)
    return float(int(bool(v)))


def engineer_features(car_dict: dict) -> dict:
    out = dict(car_dict)
    out["Mark"] = _as_cat(out.get("Mark"), mark_or_model=True)
    out["Model"] = _as_cat(out.get("Model"), mark_or_model=True)

    key = f"{out['Mark']}___{out['Model']}"
    defaults = car_defaults.get("by_model", {}).get(key, car_defaults.get("global", {}))
    global_defaults = car_defaults.get("global", {})

    # Об'єм двигуна + логіка Engine_missing
    raw_engine = _as_num(out.get("Engine_Capacity"))
    out["Engine_missing"] = int(pd.isna(raw_engine) or raw_engine <= 0)

    if pd.isna(raw_engine) or raw_engine <= 0:
        out["Engine_Capacity"] = defaults.get("Engine_Capacity", global_defaults.get("Engine_Capacity", 2.0))
    else:
        out["Engine_Capacity"] = raw_engine

    # Фізична комплектація береться з дефолтів, якщо пусто
    for col in ["Gearbox", "Fuel_Type", "Drive_Name", "Body_Name"]:
        val = _as_cat(out.get(col))
        if val == UNKNOWN:
            out[col] = defaults.get(col, global_defaults.get(col, UNKNOWN))
        else:
            out[col] = val

    # Інші категорії (місто, область) - без автозаповнення
    for col in MODEL_CAT_FEATURES:
        if col in ("Mark", "Model", "Gearbox", "Fuel_Type", "Drive_Name", "Body_Name"):
            continue
        out[col] = _as_cat(out.get(col))

    # Прапорці історії/стану авто: ЗАЛИШАЄМО NaN ЯКЩО "НЕ ВКАЗАНО"
    for col in BOOL_FEATURES:
        out[col] = _as_bool01(out.get(col))

    # Додаткові цифри
    for col in ["SeatsNumber", "DoorsNumber"]:
        val = _as_num(out.get(col), zero_as_missing=True)
        if pd.isna(val):
            out[col] = defaults.get(col, global_defaults.get(col, 0.0))
        else:
            out[col] = val

    for col in ["Photos_Count", "Description_Length", "Options_Count"]:
        out[col] = _as_num(out.get(col))

    # Похідні ознаки
    mileage = _as_num(out.get("Mileage"), zero_as_missing=True)
    age = int(out.get("Age") or 0)

    out["Mileage"] = mileage
    out["Age"] = age
    out["is_EV"] = int(out.get("Fuel_Type") == "Електро")
    out["is_suspicious_mileage"] = int(bool(age > 10 and pd.notna(mileage) and mileage < 50))
    out["is_new"] = int(age <= 3)
    out["Km_per_Year"] = (mileage / (age + 1)) if pd.notna(mileage) else np.nan
    out["is_luxury_brand"] = int(out["Mark"] in LUXURY_MARKS)
    out["is_automatic_gearbox"] = int(out.get("Gearbox") in AUTOMATIC_LIKE)
    out["log_Mileage"] = float(np.log1p(mileage)) if pd.notna(mileage) else np.nan
    out["Age_x_Mileage"] = (age * mileage) if pd.notna(mileage) else np.nan
    out["Decade"] = int((CURRENT_YEAR - age) // 10 * 10)

    return out


def build_dataframe(car_dict: dict) -> pd.DataFrame:
    enriched = engineer_features(car_dict)
    df = pd.DataFrame([enriched])
    return df[MODEL_FEATURE_ORDER]


def compute_shap(car: CarFeatures, predicted_price: float) -> dict:
    try:
        df = build_dataframe(car.model_dump())
        X_transformed = model.named_steps["categorize"].transform(df)
        shap_row = explainer.shap_values(X_transformed)[0]
        feature_names = X_transformed.columns.tolist()

        label_map = {
            "Age": "Вік авто", "Mileage": "Пробіг", "Engine_Capacity": "Об'єм двигуна",
            "Km_per_Year": "Км на рік", "Fuel_Type": "Тип пального", "Gearbox": "Коробка передач",
            "Mark": "Марка", "Model": "Модель", "Body_Name": "Кузов", "Drive_Name": "Привід",
            "Is_Crashed": "ДТП", "Custom": "Розмитнення", "First_Owner": "Перший власник",
        }
        shap_dict = {
            label_map.get(f, f): round(float(v) * predicted_price, 1)
            for f, v in zip(feature_names, shap_row)
        }
        top6 = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:6])
        return top6

    except Exception:
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
        "body_types": categories.get("body_types") or BODY_TYPES,
        "drive_types": categories.get("drive_types") or DRIVE_TYPES,
        "color_names": categories.get("color_names") or COLOR_NAMES,
    }


@app.post("/predict", tags=["Prediction"])
def predict_price(car: CarFeatures):
    _model_ready()

    invalid_inputs = ("Не вказано", "", "Інша", "Unknown")
    if car.Mark in invalid_inputs or car.Model in invalid_inputs:
        raise HTTPException(
            status_code=400,
            detail="Для отримання оцінки необхідно обов'язково обрати Марку та Модель авто."
        )

    try:
        df = build_dataframe(car.model_dump())
        raw = model.predict(df)[0]
        base_price = process_prediction(raw)

        if base_price == 0.0:
            raise HTTPException(status_code=422, detail="Не вдалося обчислити ціну.")

        shap_values = compute_shap(car, base_price)

        log.info(f"Predict: {car.Mark} {car.Model} {car.Age}р {car.Mileage}тис → ${base_price:,.0f}")

        return {
            "predicted_price_usd": base_price,
            "shap_values": shap_values,
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Помилка predict: {e}")
        raise HTTPException(status_code=500, detail=f"Помилка обчислення: {e}")


@app.post("/predict_depreciation", tags=["Prediction"])
def predict_depreciation(req: DepreciationRequest):
    _model_ready()

    invalid_inputs = ("Не вказано", "", "Інша", "Unknown")
    if req.car.Mark in invalid_inputs or req.car.Model in invalid_inputs:
        raise HTTPException(status_code=400, detail="Вкажіть Марку та Модель.")

    try:
        base = req.car.model_dump()
        predictions = []

        for year_offset in range(req.years + 1):
            current = base.copy()
            current["Age"] += year_offset
            current["Mileage"] += req.annual_mileage * year_offset
            current["Km_per_Year"] = current["Mileage"] / (current["Age"] + 1)
            current["is_new"] = int(current["Age"] <= 3)
            current["is_suspicious_mileage"] = int(
                current["Age"] > 10 and current["Mileage"] is not None and current["Mileage"] < 50
            )

            df = build_dataframe(current)
            raw = model.predict(df)[0]
            price = process_prediction(raw)

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
            df = build_dataframe(car.model_dump())
            raw = model.predict(df)[0]
            price = process_prediction(raw)
            results.append({"mark": car.Mark, "model": car.Model, "predicted_price_usd": price})
        except Exception as e:
            results.append({"mark": car.Mark, "model": car.Model, "error": str(e)})

    return {"results": results}