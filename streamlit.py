import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import joblib
import datetime

# --- 1. НАЛАШТУВАННЯ СТОРІНКИ ---
st.set_page_config(page_title="Прогноз ціни авто", page_icon="🚗", layout="centered")


# --- 2. ЗАВАНТАЖЕННЯ МОДЕЛІ ТА ДАНИХ ---
@st.cache_resource
def load_assets():
    model = CatBoostRegressor()
    # Шлях до моделі
    model.load_model('models/catboost_car_price_model.cbm')
    # Завантаження словників
    valid_categories = joblib.load('models/valid_categories.pkl')
    return model, valid_categories


model, valid_categories = load_assets()

# Дістаємо словники з файлу
valid_marks = valid_categories['valid_marks']
valid_models = valid_categories['valid_models']
mark_model_mapping = valid_categories.get('mark_model_mapping', {})
engine_mapping = valid_categories.get('engine_mapping', {})
fuel_mapping = valid_categories.get('fuel_mapping', {})
gearbox_mapping = valid_categories.get('gearbox_mapping', {})  # Нова фіча: мапінг коробок

# Дефолтні списки (якщо вибрано "Інша")
default_fuels = ["Бензин", "Дизель", "Електро", "Газ", "Гібрид (HEV)"]
default_capacities = [1.0, 1.4, 1.6, 2.0, 2.5, 3.0]
default_gearboxes = ["Автомат", "Ручна / Механіка", "Робот", "Варіатор", "Тіптронік", "Редуктор"]

# --- 3. ІНТЕРФЕЙС КОРИСТУВАЧА ---
st.title("🚗 Калькулятор вартості вживаного авто")
st.write("Введіть параметри автомобіля, і штучний інтелект спрогнозує його ціну на основі реальних даних з ринку.")

col1, col2 = st.columns(2)

with col1:
    mark = st.selectbox("Марка автомобіля", sorted(valid_marks) + ['Інша'])

    if mark == 'Інша':
        available_models = ['Інша']
    else:
        available_models = mark_model_mapping.get(mark, []) + ['Інша']

    model_name = st.selectbox("Модель автомобіля", available_models)

    # Використовуємо 2026 рік як поточний згідно з інструкцією
    year = st.number_input("Рік випуску", min_value=1990, max_value=2026, value=2023, step=1)
    mileage = st.number_input("Пробіг (тис. км)", min_value=0, max_value=1000, value=50, step=5)

with col2:
    # --- ДИНАМІЧНА КОРОБКА ПЕРЕДАЧ ---
    available_gearboxes = default_gearboxes
    if mark != 'Інша' and model_name != 'Інша':
        available_gearboxes = gearbox_mapping.get(mark, {}).get(model_name, default_gearboxes)
        if not available_gearboxes:
            available_gearboxes = default_gearboxes

    # Даємо користувачу можливість змінити коробку, навіть якщо знайдено лише 1 варіант
    gearbox = st.selectbox("Коробка передач", available_gearboxes)

    # --- ДИНАМІЧНИЙ ТИП ПАЛЬНОГО ---
    available_fuels = default_fuels
    if mark != 'Інша' and model_name != 'Інша':
        available_fuels = fuel_mapping.get(mark, {}).get(model_name, default_fuels)
        available_fuels = [f for f in available_fuels if f not in ['Не вказано', 'Other', '']]
        if not available_fuels:
            available_fuels = ["Бензин"]

    fuel_type = st.selectbox("Тип пального", available_fuels)

    # --- ДИНАМІЧНИЙ ОБ'ЄМ ДВИГУНА ---
    if fuel_type == 'Електро':
        st.text_input("Об'єм двигуна (л)", value="0.0 (Електро)", disabled=True)
        engine_capacity = 0.0
    else:
        # Шукаємо об'єми в базі для обраної марки та моделі
        if mark != 'Інша' and model_name != 'Інша':
            available_capacities = engine_mapping.get(mark, {}).get(model_name, default_capacities)
            # Якщо раптом список порожній, підставляємо дефолтний
            if not available_capacities:
                available_capacities = default_capacities
        else:
            # Для категорії "Інша" показуємо дефолтний список
            available_capacities = default_capacities

        # Виводимо виключно випадаючий список (selectbox)
        engine_capacity = st.selectbox("Об'єм двигуна (л)", available_capacities)

# Кнопка та логіка (зміни лише в передачі "Other")
st.markdown("---")
if st.button("💰 Розрахувати орієнтовну ціну", use_container_width=True):
    current_year = 2026
    age = current_year - year
    km_per_year = mileage / (age + 1)
    is_ev = 1 if fuel_type == 'Електро' else 0
    is_suspicious_mileage = 1 if (age > 10 and mileage < 50) else 0

    if km_per_year < 5 and is_ev == 0 and age > 2:
        st.warning(
            "⚠️ Зверніть увагу: вказаний пробіг є аномально низьким для віку цього авто. "
            "На реальному ринку такі автомобілі часто продаються за іншою ціною через підозру на скручений пробіг.")

    # Гарантуємо, що 'Інша' передається як 'Other'
    final_mark = 'Other' if mark == 'Інша' else mark
    final_model = 'Other' if model_name == 'Інша' else model_name

    features = ['Mark', 'Model', 'Mileage', 'Gearbox', 'Age', 'Fuel_Type', 'Engine_Capacity', 'Km_per_Year', 'is_EV',
                'is_suspicious_mileage']

    input_data = pd.DataFrame([[
        final_mark, final_model, mileage, gearbox, age, fuel_type, engine_capacity, km_per_year, is_ev,
        is_suspicious_mileage
    ]], columns=features)

    pred_log = model.predict(input_data)
    pred_price = np.expm1(pred_log)[0]

    st.success(f"### Орієнтовна ринкова вартість: **${pred_price:,.0f}**")