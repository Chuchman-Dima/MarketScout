import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (
    CAT_FEATURES,
    DATA_DIR,
    MODEL_FEATURE_ORDER,
    PROJECT_ROOT,
    RANDOM_SEED,
    UNKNOWN,
)

CURRENT_YEAR = 2026
MIN_COUNT = 10

LUXURY_MARKS = {
    "Aston Martin", "BMW-Alpina", "Lamborghini", "Rolls-Royce", "Ferrari",
    "Bentley", "Porsche", "Maserati", "Lexus", "Land Rover", "Jaguar",
    "Mercedes-Benz", "BMW", "Audi", "Lucid", "Genesis",
}
AUTOMATIC_LIKE = {"Автомат", "Типтронік", "Варіатор", "Робот"}

BOOL_TRUE = {"1", "true", "True", "yes", "Yes", True, 1}

DROP_ALWAYS = [
    "Ad_Link", "Main_Photo_URL", "Main_Photo_Path", "VIN", "Damage_Description",
    "ID", "MarkId", "ModelId", "Fuel_Id", "Gearbox_Id", "Body_Id", "Color_Id",
    "Drive_Id", "UserId", "PartnerId", "Dealer_Name", "Price_UAH", "Price_EUR",
    "Main_Currency", "Add_Date", "Update_Date", "Expire_Date",
    "Is_Sold", "From_Archive", "On_Moderation",
    "Views_Total", "Views_Today", "Bookmarks_Count", "Chips_Count",
    "Top_Level", "Top_Label", "Hot_Type", "Badges_Count",
    "CategoryId", "Fuel_Name", "Year", "State_Id", "City_Id", "StatusId",
    "Mileage_Per_Year",
]

BOOL_COLS = [
    "Is_Crashed", "Custom", "Is_Leasing", "Has_VIN", "Is_Checked_VIN",
    "VIN_Has_Restrictions", "Has_Plate", "Is_Checked_Plate", "Is_Dealer",
    "Phone_Verified", "Exchange_Possible", "Auction_Possible", "With_Video",
    "Desc_Urgent", "Desc_Bargain", "Desc_First_Owner", "Desc_Ideal",
]


def resolve_csv_path():
    candidates = [
        PROJECT_ROOT / "data" / "new_cars_dataset_2.csv",
        PROJECT_ROOT / "data" / "new_data" / "new_cars_dataset_2.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Не знайдено new_cars_dataset_2.csv. Очікувані шляхи:\n"
        + "\n".join(f"  - {p}" for p in candidates)
    )


def to_bool01(s: pd.Series) -> pd.Series:
    return s.map(
        lambda x: np.nan if pd.isna(x) or x == "" else int(x in BOOL_TRUE or x == 1)
    ).astype(float)


def _fill_cat(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    return out.replace({"": UNKNOWN, "nan": UNKNOWN, "None": UNKNOWN, "Unknown": UNKNOWN}).fillna(UNKNOWN)


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "CategoryId" in df.columns:
        df = df[df["CategoryId"].isin([0, 1])].copy()
    if "Mark" in df.columns:
        df = df[df["Mark"] != "Причеп"].copy()

    if "Engine_Volume" in df.columns:
        eng = pd.to_numeric(df["Engine_Volume"], errors="coerce")
    else:
        eng = pd.Series(np.nan, index=df.index)

    if "Fuel_Name" in df.columns:
        from_fuel = df["Fuel_Name"].astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)
        fuel_type = (
            df["Fuel_Name"].astype(str)
            .str.replace(r",?\s*\d+\.?\d*\s*л\.?", "", regex=True)
            .str.strip()
            .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        )
    else:
        from_fuel = pd.Series(np.nan, index=df.index)
        fuel_type = pd.Series(np.nan, index=df.index)

    df["Engine_Capacity"] = eng.where(eng.fillna(0) > 0, from_fuel)
    df["Fuel_Type"] = fuel_type

    year = pd.to_numeric(df["Year"], errors="coerce") if "Year" in df.columns else pd.Series(np.nan, index=df.index)
    df["Age"] = CURRENT_YEAR - year

    if "Mileage_K" in df.columns:
        df = df.rename(columns={"Mileage_K": "Mileage"})
    if "Gearbox_Name" in df.columns:
        df = df.rename(columns={"Gearbox_Name": "Gearbox"})

    for c in CAT_FEATURES:
        if c in ("Country_Origin_Id", "ConditionId"):
            continue
        if c in df.columns:
            df[c] = _fill_cat(df[c])
        else:
            df[c] = UNKNOWN

    for c in ("Country_Origin_Id", "ConditionId"):
        if c in df.columns:
            raw = pd.to_numeric(df[c], errors="coerce")
            df[c] = np.where(raw.isna() | (raw == 0), UNKNOWN, raw.astype("Int64").astype(str))
        else:
            df[c] = UNKNOWN
        df[c] = _fill_cat(df[c])

    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = to_bool01(df[c])
        else:
            df[c] = np.nan

    df["Mileage"] = pd.to_numeric(df.get("Mileage"), errors="coerce")
    df.loc[df["Mileage"] <= 0, "Mileage"] = np.nan
    df["Engine_Capacity"] = pd.to_numeric(df["Engine_Capacity"], errors="coerce")
    df.loc[df["Engine_Capacity"] <= 0, "Engine_Capacity"] = np.nan

    for num_col in ("SeatsNumber", "DoorsNumber", "Photos_Count", "Description_Length", "Options_Count"):
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
        else:
            df[num_col] = np.nan
    df.loc[df["SeatsNumber"] <= 0, "SeatsNumber"] = np.nan
    df.loc[df["DoorsNumber"] <= 0, "DoorsNumber"] = np.nan

    df["First_Owner"] = df["Desc_First_Owner"]
    df["Is_Urgent"] = df["Desc_Urgent"]
    df["Is_Bargain"] = df["Desc_Bargain"]

    drop_cols = [c for c in DROP_ALWAYS if c in df.columns and c not in MODEL_FEATURE_ORDER]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df


def build_and_apply_global_imputer(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Рахує моду/медіану ТІЛЬКИ для базової комплектації (об'єм, кузов, коробка).
    Прапорці стану залишаються NaN, щоб модель їх не ігнорувала!
    """
    impute_cols_num = ["Engine_Capacity", "SeatsNumber", "DoorsNumber"]
    impute_cols_cat = ["Gearbox", "Fuel_Type", "Drive_Name", "Body_Name"]

    global_defs = {}
    for c in impute_cols_num:
        if c in X_train.columns:
            val = X_train[c].dropna().median()
            global_defs[c] = float(val) if pd.notna(val) else 0.0

    for c in impute_cols_cat:
        if c in X_train.columns:
            valid = X_train[c][X_train[c] != UNKNOWN]
            global_defs[c] = valid.mode().iloc[0] if not valid.empty else UNKNOWN

    model_defs = {}
    grouped = X_train.groupby(["Mark", "Model"])
    for (mark, model), group in grouped:
        key = f"{mark}___{model}"
        model_defs[key] = {}

        for c in impute_cols_num:
            if c in group.columns:
                valid_num = group[c].dropna()
                val = valid_num.median() if not valid_num.empty else np.nan
                model_defs[key][c] = float(val) if pd.notna(val) else global_defs.get(c, 0.0)

        for c in impute_cols_cat:
            if c in group.columns:
                valid = group[c][group[c] != UNKNOWN]
                model_defs[key][c] = valid.mode().iloc[0] if not valid.empty else global_defs.get(c, UNKNOWN)

    def apply_imputation(df):
        df_out = df.copy()
        keys = df_out["Mark"] + "___" + df_out["Model"]

        for c in impute_cols_num:
            if c in df_out.columns:
                mapped = keys.map(lambda k: model_defs.get(k, global_defs).get(c, global_defs.get(c, 0.0)))
                mask = df_out[c].isna() | (df_out[c] <= 0)
                df_out.loc[mask, c] = mapped[mask]

        for c in impute_cols_cat:
            if c in df_out.columns:
                mapped = keys.map(lambda k: model_defs.get(k, global_defs).get(c, global_defs.get(c, UNKNOWN)))
                mask = df_out[c].isna() | (df_out[c] == UNKNOWN) | (df_out[c] == "")
                df_out.loc[mask, c] = mapped[mask]

        return df_out

    X_train_imp = apply_imputation(X_train)
    X_test_imp = apply_imputation(X_test)

    export_dict = {"global": global_defs, "by_model": model_defs}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "car_defaults.json", "w", encoding="utf-8") as f:
        json.dump(export_dict, f, ensure_ascii=False, indent=2)
    print(f"Збережено car_defaults.json у {DATA_DIR}")

    return X_train_imp, X_test_imp


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Km_per_Year"] = df["Mileage"] / (df["Age"] + 1)
    df["is_EV"] = (df["Fuel_Type"] == "Електро").astype(int)
    df["is_suspicious_mileage"] = ((df["Age"] > 10) & (df["Mileage"] < 50)).fillna(False).astype(int)
    df["is_new"] = (df["Age"] <= 3).astype(int)
    df["is_luxury_brand"] = df["Mark"].isin(LUXURY_MARKS).astype(int)
    df["is_automatic_gearbox"] = df["Gearbox"].isin(AUTOMATIC_LIKE).astype(int)
    df["Engine_missing"] = df["Engine_Capacity"].isna().astype(int)
    df["log_Mileage"] = np.log1p(df["Mileage"])
    df["Age_x_Mileage"] = df["Age"] * df["Mileage"]
    df["Decade"] = ((CURRENT_YEAR - df["Age"]) // 10 * 10).astype(int)
    return df


def group_rare_categories(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    for col in CAT_FEATURES:
        counts = X_train[col].value_counts()
        valid = set(counts[counts >= MIN_COUNT].index)
        valid.add(UNKNOWN)
        valid.add("Other")
        X_train[col] = np.where(X_train[col].isin(valid), X_train[col], "Other")
        X_test[col] = np.where(X_test[col].isin(valid), X_test[col], "Other")


def main():
    csv_path = resolve_csv_path()
    print(f"Читаємо {csv_path}")
    raw = pd.read_csv(csv_path, low_memory=False)
    df = clean_raw_dataframe(raw)

    df = df[
        (df["Price_USD"] >= 1000) & (df["Price_USD"] <= 250000)
        & (df["Age"] >= 0) & (df["Age"] <= 46)
        ].copy()
    df = df[(df["Engine_Capacity"].isna()) | (df["Engine_Capacity"] <= 10)]

    df = add_derived_features(df)

    missing = [c for c in MODEL_FEATURE_ORDER if c not in df.columns]
    if missing:
        raise KeyError(f"Немає колонок для моделі: {missing}")

    print(f"Після очищення: {df.shape}")

    X = df[MODEL_FEATURE_ORDER].copy()
    y_raw = df["Price_USD"].copy()

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y_raw, test_size=0.2, random_state=RANDOM_SEED
    )

    X_train, X_test = build_and_apply_global_imputer(X_train, X_test)

    X_train["is_EV"] = (X_train["Fuel_Type"] == "Електро").astype(int)
    X_train["is_automatic_gearbox"] = X_train["Gearbox"].isin(AUTOMATIC_LIKE).astype(int)
    X_test["is_EV"] = (X_test["Fuel_Type"] == "Електро").astype(int)
    X_test["is_automatic_gearbox"] = X_test["Gearbox"].isin(AUTOMATIC_LIKE).astype(int)

    group_rare_categories(X_train, X_test)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(DATA_DIR / "X_train.parquet")
    X_test.to_parquet(DATA_DIR / "X_test.parquet")
    y_train_raw.to_frame("Price_USD").to_parquet(DATA_DIR / "y_train.parquet")
    y_test_raw.to_frame("Price_USD").to_parquet(DATA_DIR / "y_test.parquet")
    print(f"Збережено parquet у {DATA_DIR}")


if __name__ == "__main__":
    main()