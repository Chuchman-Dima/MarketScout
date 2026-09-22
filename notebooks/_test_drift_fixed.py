import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from catboost import CatBoostRegressor
import joblib

CURRENT_YEAR = 2026
LUXURY_MARKS = {
    'Aston Martin', 'BMW-Alpina', 'Lamborghini', 'Rolls-Royce', 'Ferrari',
    'Bentley', 'Porsche', 'Maserati', 'Lexus', 'Land Rover', 'Jaguar',
    'Mercedes-Benz', 'BMW', 'Audi', 'Lucid', 'Genesis',
}
AUTOMATIC_LIKE = {'Автомат', 'Типтронік', 'Варіатор', 'Робот'}

model = CatBoostRegressor()
model.load_model("../src/backend/models/catboost_car_price_model.cbm")

train_valid = joblib.load("../data/train_valid_categories.pkl")
valid_marks = set(train_valid['valid_marks'])
valid_models = set(train_valid['valid_models'])
print(f"Завантажено поріг: {len(valid_marks)} марок, {len(valid_models)} моделей")

data = pd.read_csv("../data/new_clean_data_20.csv", low_memory=False)
df = data.drop(columns=['Unnamed: 0'], errors='ignore').copy()

df = df.drop(columns=['ID', 'Engine'], errors='ignore')
df = df[df['Price_USD'] > 0].copy()
df = df[df['Mark'] != 'Причеп'].copy()

df['Age'] = CURRENT_YEAR - df['Year']
df = df[(df['Age'] >= 0) & (df['Age'] <= 46)].copy()

split_data = df['Fuel'].str.split(', ', expand=True)
raw_fuel_type = split_data[0]
engine_from_split = pd.to_numeric(
    split_data[1].str.replace(' л.', '', regex=False), errors='coerce'
) if 1 in split_data.columns else pd.Series(np.nan, index=df.index)

looks_like_engine = raw_fuel_type.str.match(r'^\d+(\.\d+)?\s*л\.?$', na=False)
df['Fuel_Type'] = np.where(looks_like_engine, 'Не вказано', raw_fuel_type)
engine_from_type = pd.to_numeric(raw_fuel_type.str.replace(' л.', '', regex=False), errors='coerce')
df['Engine_Capacity'] = np.where(looks_like_engine, engine_from_type, engine_from_split)
df['Engine_Capacity'] = df['Engine_Capacity'].fillna(0)

df['Km_per_Year'] = df['Mileage'] / (df['Age'] + 1)
df['is_EV'] = (df['Fuel_Type'] == 'Електро').astype(int)
df['is_suspicious_mileage'] = ((df['Age'] > 10) & (df['Mileage'] < 50)).astype(int)
df['is_new'] = (df['Age'] <= 3).astype(int)

df['Mark'] = df['Mark'].fillna('Other')
df['Model'] = df['Model'].fillna('Other')
df['Gearbox'] = df['Gearbox'].fillna('Unknown')

df = df[
    (df['Price_USD'] >= 1000) &
    (df['Price_USD'] <= 250000) &
    (df['Engine_Capacity'] <= 10)
].copy()

df['is_luxury_brand'] = df['Mark'].isin(LUXURY_MARKS).astype(int)
df['is_automatic_gearbox'] = df['Gearbox'].isin(AUTOMATIC_LIKE).astype(int)
df['Engine_missing'] = (df['Engine_Capacity'] == 0).astype(int)
df['log_Mileage'] = np.log1p(df['Mileage'])
df['Age_x_Mileage'] = df['Age'] * df['Mileage']
df['Decade'] = (df['Year'] // 10 * 10).astype(int)

# --- НОВЕ: те саме "Other"-групування, яке реально бачила модель у травні ---
before_other = (df['Mark'] == 'Other').sum()
df['Mark'] = np.where(df['Mark'].isin(valid_marks), df['Mark'], 'Other')
df['Model'] = np.where(df['Model'].isin(valid_models), df['Model'], 'Other')
after_other = (df['Mark'] == 'Other').sum()
print(f"Марок замінено на 'Other': {after_other - before_other} з {len(df)} ({(after_other-before_other)/len(df)*100:.1f}%)")

features = [
    'Mark', 'Model', 'Mileage', 'Gearbox', 'Age',
    'Fuel_Type', 'Engine_Capacity', 'Km_per_Year',
    'is_EV', 'is_suspicious_mileage', 'is_new',
    'is_luxury_brand', 'Engine_missing', 'log_Mileage', 'Age_x_Mileage', 'Decade',
    'is_automatic_gearbox',
]
X_new = df[features]
y_new = df['Price_USD']

y_pred_log = model.predict(X_new)
y_pred = np.expm1(y_pred_log)

mae = mean_absolute_error(y_new, y_pred)
mape = mean_absolute_percentage_error(y_new, y_pred) * 100
r2 = r2_score(y_new, y_pred)
median_ae = np.median(np.abs(y_new.values - y_pred))

print(f"\n{'='*50}")
print(f"  ІЗ ПРАВИЛЬНИМ ПРЕПРОЦЕСИНГОМ (Other-групування)")
print(f"{'-'*50}")
print(f"  MAE     : {mae:,.2f} USD")
print(f"  MAPE    : {mape:.2f}%")
print(f"  R²      : {r2:.4f}")
print(f"  MedianAE: {median_ae:,.2f} USD")
print(f"{'='*50}")

OLD_MAE, OLD_MAPE, OLD_R2, OLD_MEDIAN = 2967.34, 22.65, 0.8713, 1360.60
NAIVE_MAE, NAIVE_MAPE, NAIVE_R2, NAIVE_MEDIAN = 3140.16, 23.27, 0.8370, 1374.20

print(f"\n{'Метрика':<10} {'Травень':>12} {'Наївний тест':>15} {'З фіксом':>12}")
print(f"{'MAE':<10} {OLD_MAE:>9,.0f} USD {NAIVE_MAE:>12,.0f} USD {mae:>9,.0f} USD")
print(f"{'MAPE':<10} {OLD_MAPE:>11.2f}% {NAIVE_MAPE:>14.2f}% {mape:>11.2f}%")
print(f"{'R²':<10} {OLD_R2:>12.4f} {NAIVE_R2:>15.4f} {r2:>12.4f}")
print(f"{'MedianAE':<10} {OLD_MEDIAN:>9,.0f} USD {NAIVE_MEDIAN:>12,.0f} USD {median_ae:>9,.0f} USD")
