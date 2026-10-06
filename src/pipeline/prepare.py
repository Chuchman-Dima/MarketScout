import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
CURRENT_YEAR = 2026
MIN_COUNT = 10

LUXURY_MARKS = {
    'Aston Martin', 'BMW-Alpina', 'Lamborghini', 'Rolls-Royce', 'Ferrari',
    'Bentley', 'Porsche', 'Maserati', 'Lexus', 'Land Rover', 'Jaguar',
    'Mercedes-Benz', 'BMW', 'Audi', 'Lucid', 'Genesis',
}
AUTOMATIC_LIKE = {'Автомат', 'Типтронік', 'Варіатор', 'Робот'}


def main():
    data = pd.read_csv("../../data/new_data/new_cars_dataset_2.csv", low_memory=False)
    df = data.copy()

    missing_summary = df.isnull().mean() * 100
    df = df.drop(columns=missing_summary[missing_summary > 90].index, errors='ignore')
    df = df.dropna(how='all', axis=1)
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=constant_cols, errors='ignore')
    df = df.drop(columns=['Ad_Link', 'Main_Photo_URL'], errors='ignore')

    df = df[df['CategoryId'].isin([0, 1])].copy()

    df['Engine_Volume'] = df['Fuel_Name'].str.extract(r'(\d+\.?\d*)').astype(float)
    df['Fuel_Type'] = df['Fuel_Name'].str.replace(r',?\s*\d+\.?\d*\s*л\.?', '', regex=True).str.strip()
    df['Fuel_Type'] = df['Fuel_Type'].replace('', np.nan)

    df['Age'] = CURRENT_YEAR - df['Year']

    df = df.drop(columns=['Price_UAH', 'Price_EUR', 'ID', 'CategoryId', 'Fuel_Name', 'Year'], errors='ignore')
    df = df.rename(columns={'Mileage_K': 'Mileage', 'Gearbox_Name': 'Gearbox', 'Engine_Volume': 'Engine_Capacity'})

    df = df[df['Mark'] != 'Причеп'].copy()

    df['Mark'] = df['Mark'].fillna('Other')
    df['Model'] = df['Model'].fillna('Other')
    df['Gearbox'] = df['Gearbox'].fillna('Unknown')
    df['Fuel_Type'] = df['Fuel_Type'].fillna('Не вказано')
    df['Engine_Capacity'] = df['Engine_Capacity'].fillna(0)

    df = df[
        (df['Price_USD'] >= 1000) & (df['Price_USD'] <= 250000) &
        (df['Age'] >= 0) & (df['Age'] <= 46) &
        (df['Engine_Capacity'] <= 10)
    ].copy()

    df['Km_per_Year'] = df['Mileage'] / (df['Age'] + 1)
    df['is_EV'] = (df['Fuel_Type'] == 'Електро').astype(int)
    df['is_suspicious_mileage'] = ((df['Age'] > 10) & (df['Mileage'] < 50)).astype(int)
    df['is_new'] = (df['Age'] <= 3).astype(int)
    df['is_luxury_brand'] = df['Mark'].isin(LUXURY_MARKS).astype(int)
    df['is_automatic_gearbox'] = df['Gearbox'].isin(AUTOMATIC_LIKE).astype(int)
    df['Engine_missing'] = (df['Engine_Capacity'] == 0).astype(int)
    df['log_Mileage'] = np.log1p(df['Mileage'])
    df['Age_x_Mileage'] = df['Age'] * df['Mileage']
    df['Decade'] = ((CURRENT_YEAR - df['Age']) // 10 * 10).astype(int)

    print(f"Після очищення: {df.shape}")

    features = [
        'Mark', 'Model', 'Mileage', 'Gearbox', 'Age',
        'Fuel_Type', 'Engine_Capacity', 'Km_per_Year',
        'is_EV', 'is_suspicious_mileage', 'is_new',
        'is_luxury_brand', 'Engine_missing', 'log_Mileage', 'Age_x_Mileage', 'Decade',
        'is_automatic_gearbox',
    ]

    X = df[features].copy()
    y_raw = df['Price_USD'].copy()

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y_raw, test_size=0.2, random_state=RANDOM_SEED
    )

    # Рідкісні Mark/Model (< MIN_COUNT входжень У TRAIN) -> "Other".
    # Поріг рахується ЛИШЕ з train, щоб не було витоку з test, і
    # застосовується до обох - так модель на інференсі ніколи не
    # побачить категорію, якої не бачила на навчанні.
    mark_counts = X_train['Mark'].value_counts()
    model_counts = X_train['Model'].value_counts()
    valid_marks = mark_counts[mark_counts >= MIN_COUNT].index
    valid_models = model_counts[model_counts >= MIN_COUNT].index

    for col, valid in [('Mark', valid_marks), ('Model', valid_models)]:
        X_train[col] = np.where(X_train[col].isin(valid), X_train[col], 'Other')
        X_test[col] = np.where(X_test[col].isin(valid), X_test[col], 'Other')

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Валідних марок: {len(valid_marks)}, моделей: {len(valid_models)}")

    X_train.to_parquet("../../data/parquet/X_train.parquet")
    X_test.to_parquet("../../data/parquet/X_test.parquet")
    y_train_raw.to_frame('Price_USD').to_parquet("../../data/parquet/y_train.parquet")
    y_test_raw.to_frame('Price_USD').to_parquet("../../data/parquet/y_test.parquet")
    print("Збережено parquet-файли.")


if __name__ == "__main__":
    main()
