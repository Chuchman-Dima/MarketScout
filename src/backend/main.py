"""
Auto Price Predictor — FastAPI Backend
"""

import logging
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

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

# Статичний список преміум/люкс марок — має ЗБІГАТИСЯ зі списком у main.ipynb,
# інакше ознака is_luxury_brand на інференсі не відповідатиме тренуванню.
LUXURY_MARKS = {
    "Aston Martin", "BMW-Alpina", "Lamborghini", "Rolls-Royce", "Ferrari",
    "Bentley", "Porsche", "Maserati", "Lexus", "Land Rover", "Jaguar",
    "Mercedes-Benz", "BMW", "Audi", "Lucid", "Genesis",
}
AUTOMATIC_LIKE = {"Автомат", "Типтронік", "Варіатор", "Робот"}

# ─────────────────────────────────────────────
# ЕВРИСТИЧНІ КОРЕКТИВИ (нові поля з API.py / app.py)
# ─────────────────────────────────────────────
# ВАЖЛИВО: catboost_car_price_model.cbm НЕ навчена на цих ознаках (їх немає
# в cars_dataset.csv, на якому тренується main.ipynb). Тому вони НЕ йдуть
# у MODEL_FEATURE_ORDER і не передаються в model.predict() — CatBoost вимагає
# точно той самий набір і порядок фіч, що й на тренуванні, інакше впаде.
#
# Замість цього застосовуємо їх як прозорий (rule-based) шар поверх ML-ціни:
# фінальна_ціна = ML_ціна * (1 + сума_корективів). Кожен коректив повертається
# окремо в price_adjustments, щоб було видно, що порахувала модель, а що —
# бізнес-правило. Коли ці ознаки з'являться в тренувальному датасеті й модель
# буде перенавчена — цей шар можна прибрати.
BODY_TYPES = [
    "Седан", "Позашляховик / Кросовер", "Хетчбек", "Універсал",
    "Мінівен", "Купе", "Пікап", "Інше",
]
DRIVE_TYPES = ["Передній", "Задній", "Повний", "Не вказано"]
COLOR_NAMES = [
    "Чорний", "Білий", "Сірий", "Сріблястий", "Синій",
    "Червоний", "Зелений", "Інший",
]

# Кожен коректив — частка (+/-) від базової ML-ціни.
ADJ_CRASHED = -0.15          # після ДТП
ADJ_NOT_CUSTOMS = -0.20      # нерозмитнене авто
ADJ_FIRST_OWNER = 0.03       # перший власник
ADJ_FULL_DRIVE = 0.04        # повний привід
ADJ_DEMAND_BODY = 0.03       # затребувані кузови (кросовер/пікап)
DEMAND_BODY_TYPES = {"Позашляховик / Кросовер", "Пікап"}

# Порядок ознак має ЗБІГАТИСЯ з `features` у main.ipynb на момент навчання моделі
MODEL_FEATURE_ORDER = [
    "Mark", "Model", "Mileage", "Gearbox", "Age",
    "Fuel_Type", "Engine_Capacity", "Km_per_Year",
    "is_EV", "is_suspicious_mileage", "is_new",
    "is_luxury_brand", "Engine_missing", "log_Mileage", "Age_x_Mileage", "Decade",
    "Gearbox_simple",
]
MODEL_CAT_FEATURES = ["Mark", "Model", "Gearbox", "Fuel_Type"]


# ─────────────────────────────────────────────
# ЗАВАНТАЖЕННЯ МОДЕЛІ / КАТЕГОРІЙ (один раз)
# ─────────────────────────────────────────────
model: CatBoostRegressor | None = None
categories: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Завантажує модель та категорії при старті."""
    global model, categories

    log.info("Завантаження моделі…")
    try:
        model = CatBoostRegressor()
        model.load_model("models/catboost_car_price_model.cbm")
        log.info("Модель завантажена успішно.")
    except Exception as e:
        log.error(f"Помилка завантаження моделі: {e}")
        model = None

    log.info("Завантаження категорій…")
    try:
        categories = joblib.load("models/valid_categories.pkl")
        # Прибираємо «Причеп» зі всіх маппінгів
        categories.get("valid_marks", [])
        if "valid_marks" in categories:
            categories["valid_marks"] = [m for m in categories["valid_marks"] if m != "Причеп"]
        for key in ("mark_model_mapping", "engine_mapping", "fuel_mapping", "gearbox_mapping"):
            if key in categories:
                categories[key].pop("Причеп", None)
        log.info("Категорії завантажені успішно.")
    except FileNotFoundError:
        log.warning("valid_categories.pkl не знайдено — категорії порожні.")
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
    Gearbox: str = Field(..., min_length=1)
    Age: int = Field(..., ge=0, le=60)
    Fuel_Type: str = Field(..., min_length=1)
    Engine_Capacity: float = Field(..., ge=0, le=20)
    Km_per_Year: float = Field(..., ge=0)
    is_EV: int = Field(..., ge=0, le=1)
    is_suspicious_mileage: int = Field(..., ge=0, le=1)
    is_new: int = Field(..., ge=0, le=1)

    # ── Нові поля (не йдуть у ML-модель, лише в евристичний шар корективів) ──
    Body_Name: str | None = Field(default=None)
    Drive_Name: str | None = Field(default=None)
    Color_Name: str | None = Field(default=None)
    Is_Crashed: bool = Field(default=False)
    Custom: bool = Field(default=True)          # чи розмитнене
    First_Owner: bool = Field(default=False)
    Exchange_Possible: bool = Field(default=False)  # інформаційне, на ціну не впливає
    Is_Bargain: bool = Field(default=False)          # інформаційне, на ціну не впливає
    Is_Urgent: bool = Field(default=False)           # інформаційне, на ціну не впливає

    @field_validator("Mileage", "Engine_Capacity", "Km_per_Year")
    @classmethod
    def must_be_finite(cls, v: float) -> float:
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


def engineer_features(car_dict: dict) -> dict:
    """
    Додає похідні ознаки, якими навчалась оновлена модель.
    """
    out = dict(car_dict)
    out["is_luxury_brand"] = int(out["Mark"] in LUXURY_MARKS)

    # ВИПРАВЛЕНО: тепер повертаємо 1 (якщо автомат) або 0 (якщо механіка)
    out["Gearbox_simple"] = 1 if out["Gearbox"] in AUTOMATIC_LIKE else 0

    out["Engine_missing"] = int(out["Engine_Capacity"] == 0)
    out["log_Mileage"] = float(np.log1p(out["Mileage"]))
    out["Age_x_Mileage"] = out["Age"] * out["Mileage"]
    year = CURRENT_YEAR - out["Age"]
    out["Decade"] = int(year // 10 * 10)
    return out


def build_dataframe(car_dict: dict) -> pd.DataFrame:
    """Формує DataFrame з одного словника авто, з похідними ознаками,
    у тому самому порядку колонок, що й на тренуванні."""
    enriched = engineer_features(car_dict)
    df = pd.DataFrame([enriched])
    return df[MODEL_FEATURE_ORDER]

def compute_shap(car: CarFeatures, predicted_price: float) -> dict:
    """
    Обчислює SHAP-подібні внески кожної характеристики.
    Якщо модель підтримує get_feature_importance(type='ShapValues') — використовуємо її.
    Інакше — евристична апроксимація.
    """
    try:
        df = build_dataframe(car.model_dump())
        pool = Pool(df, cat_features=MODEL_CAT_FEATURES)
        shap_matrix = model.get_feature_importance(
            data=pool,
            type="ShapValues",
        )
        # shap_matrix: (1, n_features+1) — остання колонка — базове значення
        shap_row = shap_matrix[0, :-1]
        feature_names = df.columns.tolist()

        # Переводимо в зручні Ukrainian-назви та залишаємо топ-6 за abs
        label_map = {
            "Age":                  "Вік авто",
            "Mileage":              "Пробіг",
            "Engine_Capacity":      "Об'єм двигуна",
            "Km_per_Year":          "Км на рік",
            "Fuel_Type":            "Тип пального",
            "Gearbox":              "Коробка передач",
            "Mark":                 "Марка",
            "Model":                "Модель",
            "is_EV":                "Електро",
            "is_suspicious_mileage": "Підозр. пробіг",
            "is_new":               "Нове авто (≤3р)",
            "is_luxury_brand":      "Преміум-марка",
            "Engine_missing":       "Об'єм не вказано",
            "log_Mileage":          "Пробіг (log)",
            "Age_x_Mileage":        "Вік × Пробіг",
            "Decade":               "Десятиліття випуску",
            "Gearbox_simple":       "Тип КПП (спрощ.)",
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