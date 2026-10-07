"""
Auto Price Predictor — FastAPI Backend
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import common  # noqa: F401 — CategoricalCaster для joblib.load pipeline
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
    # Якщо локально в src/backend, беремо батьківську директорію проєкту на 2 рівні вище
    if len(BASE_DIR.parents) > 1:
        return BASE_DIR.parents[1] / "models_results"
    return BASE_DIR / "models_results"


MODELS_DIR = _models_dir()
MODEL_PATH = MODELS_DIR / "lightgbm_pipeline.pkl"
CATEGORIES_PATH = MODELS_DIR / "valid_categories.json"
CATEGORIES_PKL_PATH = MODELS_DIR / "valid_categories.pkl"

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

# Статичний список преміум/люкс марок — має ЗБІГАТИСЯ зі списком у prepare.py,
# інакше ознака is_luxury_brand на інференсі не відповідатиме тренуванню.
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

# Порядок і склад колонок = MODEL_FEATURE_ORDER / CAT_FEATURES у src/pipeline/common.py
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
# ЗАВАНТАЖЕННЯ МОДЕЛІ / КАТЕГОРІЙ (один раз)
# ─────────────────────────────────────────────
model: Pipeline | None = None          # Pipeline(categorize → LGBMRegressor), див. train_lightgbm.py
explainer: shap.TreeExplainer | None = None  # будується один раз при старті - дорого створювати на кожен запит
categories: dict = {}


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
    """JSON з мапінгами; якщо їх немає — доповнює з legacy valid_categories.pkl."""
    if CATEGORIES_PATH.is_file():
        with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    if not data.get("mark_model_mapping") and CATEGORIES_PKL_PATH.is_file():
        log.info("Доповнюємо категорії з valid_categories.pkl…")
        legacy = joblib.load(CATEGORIES_PKL_PATH)
        for key in (
            "valid_marks",
            "valid_models",
            "mark_model_mapping",
            "engine_mapping",
            "fuel_mapping",
            "gearbox_mapping",
        ):
            data.setdefault(key, legacy.get(key, {} if "mapping" in key else []))

    return _strip_trailer(data)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Завантажує модель, SHAP-explainer та категорії при старті."""
    global model, explainer, categories

    log.info("Завантаження моделі…")
    try:
        model = joblib.load(MODEL_PATH)
        # TreeExplainer будуємо один раз тут (ініціалізація ~0.3-0.5с) і
        # перевикористовуємо в кожному /predict - сам прогноз SHAP потім
        # займає мілісекунди, на відміну від повторної ініціалізації.
        explainer = shap.TreeExplainer(model.named_steps["model"])
        log.info("Модель та SHAP-explainer завантажені успішно.")
    except Exception as e:
        log.error(f"Помилка завантаження моделі: {e}")
        model = None
        explainer = None

    log.info("Завантаження категорій…")
    try:
        categories = load_categories_data()
        if categories.get("mark_model_mapping"):
            log.info(
                "Категорії завантажені (%d марок, мапінг моделей OK).",
                len(categories.get("valid_marks", [])),
            )
        elif categories:
            log.warning(
                "Категорії без mark_model_mapping — у UI буде лише «Інша» для моделей. "
                "Запустіть src/pipeline/export_valid_categories.py"
            )
        else:
            log.warning("Категорії порожні.")
    except Exception as e:
        log.error(f"Помилка завантаження категорій: {e}")
        categories = {}

    yield  # ← сервер працює тут

    log.info("Зупинка сервера.")


# ─────────────────────────────────────────────
# ЗАСТОСУНОК
# ─────────────────────────────────────────────
app = FastAPI(
    title="Auto Price Predictor API",
    version="2.1.0",
    description="ML-бекенд для прогнозу ринкової ціни автомобілів",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# ГЛОБАЛЬНИЙ ОБРОБНИК ПОМИЛОК
# ─────────────────────────────────────────────
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
    """
    Якщо модель повернула логарифм ціни (raw < 50) — застосовуємо expm1.
    Інакше — вже готова ціна в USD.
    """
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
    if v is None:
        return np.nan
    if isinstance(v, str) and v.strip() in ("", UNKNOWN, "None"):
        return np.nan
    return float(int(bool(v)))


def engineer_features(car_dict: dict) -> dict:
    """Нормалізує unknown і додає похідні ознаки, як у prepare.py."""
    out = dict(car_dict)
    out["Mark"] = _as_cat(out.get("Mark"), mark_or_model=True)
    out["Model"] = _as_cat(out.get("Model"), mark_or_model=True)
    for col in MODEL_CAT_FEATURES:
        if col in ("Mark", "Model"):
            continue
        out[col] = _as_cat(out.get(col))

    mileage = _as_num(out.get("Mileage"), zero_as_missing=True)
    age = int(out.get("Age") or 0)
    engine = _as_num(out.get("Engine_Capacity"))
    if pd.isna(engine):
        engine = 0.0

    out["Mileage"] = mileage
    out["Age"] = age
    out["Engine_Capacity"] = engine
    out["is_EV"] = int(out["Fuel_Type"] == "Електро")
    out["is_suspicious_mileage"] = int(bool(age > 10 and pd.notna(mileage) and mileage < 50))
    out["is_new"] = int(age <= 3)
    out["Km_per_Year"] = (mileage / (age + 1)) if pd.notna(mileage) else np.nan
    out["is_luxury_brand"] = int(out["Mark"] in LUXURY_MARKS)
    out["is_automatic_gearbox"] = int(out["Gearbox"] in AUTOMATIC_LIKE)
    out["Engine_missing"] = int(float(engine) == 0)
    out["log_Mileage"] = float(np.log1p(mileage)) if pd.notna(mileage) else np.nan
    out["Age_x_Mileage"] = (age * mileage) if pd.notna(mileage) else np.nan
    out["Decade"] = int((CURRENT_YEAR - age) // 10 * 10)

    for col in BOOL_FEATURES:
        out[col] = _as_bool01(out.get(col))
    for col in NUM_OPTIONAL:
        val = _as_num(out.get(col), zero_as_missing=(col in ("SeatsNumber", "DoorsNumber")))
        out[col] = val
    return out


def build_dataframe(car_dict: dict) -> pd.DataFrame:
    """Формує DataFrame з одного словника авто, з похідними ознаками,
    у тому самому порядку колонок, що й на тренуванні."""
    enriched = engineer_features(car_dict)
    df = pd.DataFrame([enriched])
    return df[MODEL_FEATURE_ORDER]


def compute_shap(car: CarFeatures, predicted_price: float) -> dict:
    """
    Обчислює SHAP-внески кожної характеристики через shap.TreeExplainer
    (побудований один раз у lifespan, не на кожен запит).
    Якщо SHAP з якоїсь причини недоступний — евристична апроксимація.
    """
    try:
        df = build_dataframe(car.model_dump())
        # categorize-крок переводить Mark/Model/Gearbox/Fuel_Type у pandas
        # 'category' dtype з тим самим словником категорій, що бачила модель
        # на тренуванні (CategoricalCaster.transform, див. common.py).
        X_transformed = model.named_steps["categorize"].transform(df)
        shap_row = explainer.shap_values(X_transformed)[0]
        feature_names = X_transformed.columns.tolist()

        # Переводимо в зручні Ukrainian-назви та залишаємо топ-6 за abs
        label_map = {
            "Age": "Вік авто",
            "Mileage": "Пробіг",
            "Engine_Capacity": "Об'єм двигуна",
            "Km_per_Year": "Км на рік",
            "Fuel_Type": "Тип пального",
            "Gearbox": "Коробка передач",
            "Mark": "Марка",
            "Model": "Модель",
            "Modification": "Модифікація",
            "Body_Name": "Кузов",
            "Drive_Name": "Привід",
            "Color_Name": "Колір",
            "Wheel_Name": "Кермо",
            "SeatsNumber": "К-сть місць",
            "DoorsNumber": "К-сть дверей",
            "Is_Crashed": "ДТП",
            "Custom": "Розмитнення",
            "First_Owner": "Перший власник",
            "Is_Leasing": "Лізинг",
            "Has_VIN": "Є VIN",
            "Is_Checked_VIN": "VIN перевірено",
            "VIN_Has_Restrictions": "Обмеження VIN",
            "State_Name": "Область",
            "City": "Місто",
            "Is_Dealer": "Дилер",
            "Seller_Type": "Тип продавця",
            "Exchange_Possible": "Обмін",
            "Is_Bargain": "Торг",
            "Is_Urgent": "Терміново",
            "Photos_Count": "К-сть фото",
            "With_Video": "Є відео",
            "Description_Length": "Довжина опису",
            "Options_Count": "К-сть опцій",
            "Desc_Ideal": "«Ідеальний» в описі",
            "is_EV": "Електро",
            "is_suspicious_mileage": "Підозр. пробіг",
            "is_new": "Нове авто (≤3р)",
            "is_luxury_brand": "Преміум-марка",
            "Engine_missing": "Об'єм не вказано",
            "log_Mileage": "Пробіг (log)",
            "Age_x_Mileage": "Вік × Пробіг",
            "Decade": "Десятиліття випуску",
            "is_automatic_gearbox": "Тип КПП (спрощ.)",
        }
        shap_dict = {
            label_map.get(f, f): round(float(v) * predicted_price, 1)
            for f, v in zip(feature_names, shap_row)
        }
        # Топ-6 за абсолютним значенням
        top6 = dict(
            sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
        )
        return top6

    except Exception:
        # Евристична апроксимація (якщо SHAP не підтримується)
        base = predicted_price
        age_effect  = -base * 0.04 * car.Age if car.Age > 3 else base * 0.03
        mile_effect = -base * 0.003 * car.Mileage if car.Mileage > 100 else base * 0.015 * (100 - car.Mileage) / 100
        fuel_effect = base * 0.03 if car.Fuel_Type in ("Дизель", "Гібрид (HEV)") else -base * 0.01
        gear_effect = base * 0.02 if car.Gearbox == "Автомат" else -base * 0.01
        eng_effect  = base * 0.01 * (car.Engine_Capacity - 1.6) if car.Engine_Capacity > 0 else 0
        luxury_effect = base * 0.08 if car.Mark in LUXURY_MARKS else 0.0
        return {
            "Вік авто":       round(age_effect, 1),
            "Пробіг":         round(mile_effect, 1),
            "Тип пального":   round(fuel_effect, 1),
            "Коробка передач": round(gear_effect, 1),
            "Об'єм двигуна":  round(eng_effect, 1),
            "Преміум-марка":  round(luxury_effect, 1),
        }


def apply_heuristic_adjustments(car: CarFeatures, base_price: float) -> tuple[float, dict]:
    """
    Застосовує rule-based корективи поверх ML-ціни для ознак, яких ще немає
    в тренувальних даних моделі (див. коментар біля ADJ_* констант вище).
    Повертає (фінальна_ціна, {назва_коректива: сума_у_$}).
    """
    adjustments: dict[str, float] = {}

    if car.Is_Crashed:
        adjustments["Після ДТП"] = round(base_price * ADJ_CRASHED, 1)
    if not car.Custom:
        adjustments["Нерозмитнене"] = round(base_price * ADJ_NOT_CUSTOMS, 1)
    if car.First_Owner:
        adjustments["Перший власник"] = round(base_price * ADJ_FIRST_OWNER, 1)
    if car.Drive_Name == "Повний":
        adjustments["Повний привід"] = round(base_price * ADJ_FULL_DRIVE, 1)
    if car.Body_Name in DEMAND_BODY_TYPES:
        adjustments["Затребуваний кузов"] = round(base_price * ADJ_DEMAND_BODY, 1)

    final_price = base_price + sum(adjustments.values())
    final_price = max(round(final_price, 2), 0.0)
    return final_price, adjustments


def _model_ready() -> None:
    """Кидає 503 якщо модель не завантажена."""
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не готова. Спробуйте пізніше.")


# ─────────────────────────────────────────────
# ЕНДПОІНТИ
# ─────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Перевірка стану сервера."""
    return {
        "status":   "ok",
        "model":    "loaded" if model is not None else "not_loaded",
        "categories": "loaded" if categories else "empty",
    }


@app.get("/categories", tags=["Data"])
def get_categories():
    """Повертає всі допустимі значення для форми вводу."""
    if not categories:
        raise HTTPException(status_code=404, detail="Категорії не знайдені.")
    # Довідники для нових (не-ML) полів — тримаємо в бекенді, щоб UI
    # не хардкодив списки окремо від логіки корективів.
    return {
        **categories,
        "body_types":  BODY_TYPES,
        "drive_types": DRIVE_TYPES,
        "color_names": COLOR_NAMES,
    }


@app.post("/predict", tags=["Prediction"])
def predict_price(car: CarFeatures):
    """
    Прогнозує ринкову ціну автомобіля.
    Повертає ціну в USD і SHAP-внески характеристик.
    """
    _model_ready()

    try:
        df = build_dataframe(car.model_dump())
        raw = model.predict(df)[0]
        base_price = process_prediction(raw)

        if base_price == 0.0:
            raise HTTPException(status_code=422, detail="Не вдалося обчислити ціну для цих параметрів.")

        shap_values = compute_shap(car, base_price)
        final_price, adjustments = apply_heuristic_adjustments(car, base_price)

        log.info(
            f"Predict: {car.Mark} {car.Model} {car.Age}р {car.Mileage}тис → "
            f"ML=${base_price:,.0f} корективи={adjustments} → ${final_price:,.0f}"
        )

        return {
            "predicted_price_usd": final_price,   # фінальна ціна (ML + корективи)
            "base_ml_price_usd":   base_price,     # чиста ML-ціна без корективів
            "shap_values":         shap_values,
            "price_adjustments":   adjustments,    # rule-based корективи, $ (не з моделі)
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Помилка predict: {e}")
        raise HTTPException(status_code=500, detail=f"Помилка обчислення: {e}")


@app.post("/predict_depreciation", tags=["Prediction"])
def predict_depreciation(req: DepreciationRequest):
    """
    Прогнозує ціну авто на кілька років вперед,
    враховуючи старіння та збільшення пробігу.
    """
    _model_ready()

    try:
        base = req.car.model_dump()
        predictions = []

        for year_offset in range(req.years + 1):
            current = base.copy()
            current["Age"]      += year_offset
            current["Mileage"]  += req.annual_mileage * year_offset
            current["Km_per_Year"] = current["Mileage"] / (current["Age"] + 1)
            current["is_new"]   = int(current["Age"] <= 3)
            current["is_suspicious_mileage"] = int(current["Age"] > 10 and current["Mileage"] < 50)

            df  = build_dataframe(current)
            raw = model.predict(df)[0]
            price = process_prediction(raw)
            # Ознаки на кшталт ДТП/розмитнення не змінюються з роками, тому
            # застосовуємо ті самі корективи, що й для req.car, до кожного року.
            price, _ = apply_heuristic_adjustments(req.car, price)

            # Ціна не може рости з часом
            if predictions and price > predictions[-1]["Price"]:
                price = predictions[-1]["Price"]

            predictions.append({"Year": year_offset, "Price": price})

        total_loss = predictions[0]["Price"] - predictions[-1]["Price"]
        log.info(
            f"Depreciation: {req.car.Mark} {req.car.Model} "
            f"→ втрата ${total_loss:,.0f} за {req.years} р."
        )

        return {"depreciation": predictions, "total_loss_usd": round(total_loss, 2)}

    except Exception as e:
        log.error(f"Помилка depreciation: {e}")
        raise HTTPException(status_code=500, detail=f"Помилка розрахунку знецінення: {e}")


@app.post("/predict_batch", tags=["Prediction"])
def predict_batch(cars: list[CarFeatures]):
    """
    Пакетний прогноз для кількох авто одразу (до 20).
    Зручно для порівняння варіантів.
    """
    _model_ready()

    if len(cars) > 20:
        raise HTTPException(status_code=400, detail="Максимум 20 авто за один запит.")

    results = []
    for car in cars:
        try:
            df    = build_dataframe(car.model_dump())
            raw   = model.predict(df)[0]
            price = process_prediction(raw)
            price, adjustments = apply_heuristic_adjustments(car, price)
            results.append({
                "mark":  car.Mark,
                "model": car.Model,
                "predicted_price_usd": price,
                "price_adjustments":   adjustments,
            })
        except Exception as e:
            results.append({
                "mark":  car.Mark,
                "model": car.Model,
                "error": str(e),
            })

    return {"results": results}