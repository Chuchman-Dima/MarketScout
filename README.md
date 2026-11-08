# MarketScout — Used Car Market Price Estimation (AUTO.RIA)

An end-to-end ML product covering vehicle listing collection through the official AUTO.RIA API, data cleaning, model comparison, a FastAPI backend, and a Streamlit interface.

The application predicts the **fair market value** of a used passenger car in USD (with conversion to UAH and EUR). It also provides practical tools for buyers: listing price evaluation, a loan calculator, depreciation forecasts, and the ability to save and compare multiple vehicles.

### Try the Application

- **[Open the Application (Streamlit)](https://car-price-predictor-autoria-as5etqdr7uhjyugcgbcxmy.streamlit.app/)**

> If the page does not load immediately, wait 60–120 seconds for the free server to wake up, then refresh the page.

<img width="1339" height="1144" alt="image" src="https://github.com/user-attachments/assets/83e96f0e-2cab-42ae-99e2-b0fe922e2098" />

### Analytical Dashboard (Power BI)

- **[Open Interactive Dashboard](https://app.powerbi.com/view?r=eyJrIjoiMGM0ZGVlMDEtZWI1Mi00ZTFmLWJkMTUtMzc5N2IyNDkyZmU5IiwidCI6ImI3NjdhNzJlLTQwYjYtNDc2ZC1iMTQ0LTI0OGRhMDdjNzFiNCIsImMiOjl9)**

The dashboard explores the used car market based on AUTO.RIA listings, including price distributions, popular makes and models, and the impact of vehicle age and mileage on market value.

<img width="1675" height="941" alt="MarketScout" src="https://github.com/user-attachments/assets/03ffd7fa-6459-4975-94e6-e476aec558d1" />

## Technology Stack

### Programming Language

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### Data Collection

![AUTO.RIA API](https://img.shields.io/badge/AUTO.RIA_API-E4002B?style=for-the-badge)
![REST API](https://img.shields.io/badge/REST_API-02569B?style=for-the-badge)

### Data Processing and Analysis

![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Apache Parquet](https://img.shields.io/badge/Parquet-50ABF1?style=for-the-badge&logo=apacheparquet&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

### Machine Learning

![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02A88E?style=for-the-badge)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=for-the-badge&logoColor=black)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge)
![Optuna](https://img.shields.io/badge/Optuna-2980B9?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-FF0051?style=for-the-badge)
![joblib](https://img.shields.io/badge/joblib-5C6BC0?style=for-the-badge)

### Backend

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-499848?style=for-the-badge)

### Frontend and Visualization

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Altair](https://img.shields.io/badge/Altair-1F77B4?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)

### Infrastructure and Deployment

![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Docker Compose](https://img.shields.io/badge/Docker_Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Docker Hub](https://img.shields.io/badge/Docker_Hub-0db7ed?style=for-the-badge&logo=docker&logoColor=white)
![Streamlit Community Cloud](https://img.shields.io/badge/Streamlit_Community_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

### Development Tools

![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![PyCharm](https://img.shields.io/badge/PyCharm-000000?style=for-the-badge&logo=pycharm&logoColor=white)

## Application Features

- **Market Value Estimation.** The model predicts a vehicle's price in USD. The interface displays an estimated price range based on model error and converts the result into UAH and EUR.
- **Optional Input Fields.** Body type, drivetrain, color, transmission, fuel type, engine displacement, accident history, and other fields can be left as "Not specified". Missing values are not replaced with zero: numerical features use typical values (median or mode by make and model), while categorical features use a separate "Not specified" category.
- **Listing Price Evaluator.** Compares the seller's asking price with the estimated market value.
- **Loan Calculator.** Calculates the down payment, loan term, interest rate, monthly payment, and total overpayment.
- **Depreciation Forecast.** Estimates vehicle value over the next several years, accounting for aging and mileage growth.
- **Saved Vehicles and Comparisons.** Supports batch price estimation, vehicle comparisons, and session history.

## Model Development

The project initially relied on **CatBoost**, with data collection, EDA, a single price regression model, and a FastAPI + Streamlit + Docker application. Later, the dataset was expanded and multiple algorithms were compared on the larger dataset. **LightGBM** was ultimately selected for deployment.

### May–September: CatBoost

**May.** The initial working dataset contained approximately **20,000** passenger car listings. The model was a `CatBoostRegressor` trained on `log1p(Price_USD)`, with predictions transformed back into USD using `expm1`.

CatBoost was selected because it natively handles categorical features (make, model, transmission, and fuel type) without requiring one-hot encoding.

However, this sample did not fully represent the market, and outliers (older vehicles and unusual prices) had a significant impact on MAE. Rare-category grouping and outlier thresholds were initially calculated using the entire training set before cross-validation, slightly inflating the metrics. These calculations were moved inside each fold to prevent data leakage.

At this stage, the application already included price estimation, currency conversion, prediction explanations, listing evaluation, fuel type information, depreciation forecasts, vehicle comparisons, and deployment.

**September.** The original dataset was no longer sufficient. A new parser (`src/parser/parser.py`) collected approximately **100,000** listings (`data/new_data/new_cars_dataset_2.csv`), adding features such as body type, drivetrain, color, customs clearance status, accident history, location, seller type, and sale conditions.

The original CatBoost model was evaluated on fresh data and then retrained on the expanded dataset. Model performance improved significantly due to the increased dataset size and diversity, rather than a change in algorithm.

This made it clear that simply training another CatBoost model was not enough. Multiple algorithms needed to be compared using **identically prepared data**.

### Model Comparison and LightGBM Selection

Data preparation is handled by `src/pipeline/prepare.py` (Parquet format, 80/20 train-test split, and rare-category grouping based on training data only).

The candidate models included Decision Tree (baseline), CatBoost, LightGBM, and XGBoost, along with shared cross-validation, Optuna hyperparameter optimization for LightGBM, and SHAP analysis.

On the same expanded dataset:

| Model | MAE, USD | Median AE, USD | MAPE, % | R² | CV MAE |
|---|---:|---:|---:|---:|---:|
| XGBoost | 1903 | 949 | 16.16 | 0.920 | 1967 ± 11 |
| **LightGBM** | 1921 | **937** | **15.75** | 0.919 | 1984 ± 10 |
| CatBoost | 2152 | 995 | 16.76 | 0.886 | 2180 ± 18 |
| Decision Tree | 2694 | — | 21.92 | 0.850 | — |

The Decision Tree performs worse, confirming the advantage of gradient boosting for this task.

CatBoost underperforms the other boosting algorithms in this pipeline. XGBoost and LightGBM achieve nearly identical MAE (approximately $18 difference).

LightGBM achieves better MAPE and more stable cross-validation results. Optuna provided only marginal improvements over manually selected hyperparameters.

**LightGBM was selected for production** due to its competitive performance compared with XGBoost, convenient categorical feature handling, fast inference, and SHAP compatibility.

The final application model uses vehicle characteristics such as make, model, mileage, age, transmission, fuel type, engine displacement, drivetrain, accident history, customs clearance status, first-owner status, and derived features.

Mileage is modeled as a factor that does not increase vehicle value.

## Model Performance Over Time

The following table shows how model performance evolved as the dataset and feature set expanded. Prices are evaluated in USD after applying `expm1`.

| Period | Dataset | Production Model | Approx. Rows | MAE, USD | Median AE, USD | MAPE, % | R² |
|---|---|---|---:|---:|---:|---:|---:|
| May | Initial dataset, limited feature set | CatBoost | ~20,000 | 2,967 | 1,361 | 22.65 | 0.87 |
| September (previous preprocessing) | New data, May preprocessing | CatBoost | ~20,000 | 3,071 | 1,366 | 23.01 | 0.85 |
| September | New parser, expanded features | CatBoost | ~100,000 | 2,152 | 995 | 16.76 | 0.89 |
| After model comparison | Same expanded dataset | **LightGBM** | ~77,000 train | **1,921** | **937** | **15.75** | **0.92** |

Median AE values for these models are averages across 5-fold cross-validation.

The September evaluation of the original May model is not a hold-out evaluation: the May model was not retrained and was evaluated on September data using the restored May preprocessing and rare-category grouping.

From the initial May CatBoost model to the current LightGBM model, MAE decreased by approximately **1.5x**.

The main improvement came in September, driven by the larger and more recent dataset. The second improvement came from switching algorithms on the same expanded dataset.

## Data and Features

| | May | September and Later |
|---|---|---|
| Data Source | AUTO.RIA API | Same API, new parser |
| Dataset Size | ~20K listings | ~100K listings |
| Train / Test | Smaller sample | ~76,800 / ~19,200 |
| Target Variable | `Price_USD` | `Price_USD` (trained on `log1p`) |

The model uses features including make, model, mileage, vehicle age, transmission, fuel type, engine displacement, drivetrain, body type, color, accident history, customs clearance status, first-owner status, sale conditions, and derived features (`Km_per_Year`, `is_EV`, `is_luxury_brand`, `log_Mileage`, etc.).

Features that could cause target leakage or describe the listing rather than the vehicle itself are excluded, including `Price_UAH` / `Price_EUR`, views, "top" listing indicators, and moderation dates.

If a user does not provide engine displacement, transmission, or similar information, the application fills in a typical market value (median or mode) rather than zero.

## Repository Structure

```text
├── data/
│   ├── cars_dataset_sample.csv
│   ├── new_data/                    # Full dataset is not committed to Git
│   │   └── README_parser.md
│   └── parquet/
├── docs/
│   └── images/                      # Screenshots used in README
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── parser/parser.py
│   ├── pipeline/                    # Preparation, training, CV, and model comparison
│   ├── backend/                     # FastAPI, final training, model weights
│   └── frontend/                    # Streamlit
├── docker-compose.yml
└── README.md
```

## How to Run

The application requires `lightgbm_pipeline.pkl` and `valid_categories.json` in the models directory.

### Docker Compose

```bash
docker compose up --build
```

- API: http://localhost:8080
- UI: http://localhost:8501

### Separate Docker Images

```bash
docker network create auto_net
docker run -d --name backend --network auto_net -p 8080:8000 dmytrochuchman/auto-ria-backend
docker run -d --name frontend --network auto_net -p 8501:8501 dmytrochuchman/auto-ria-frontend
```

### Local Setup

1. Backend: Install dependencies from `src/backend/requirements.txt` and run Uvicorn on port 8080.
2. Frontend: Run the following command:

```bash
BACKEND_URL=http://localhost:8080 streamlit run src/frontend/app.py
```

To use the parser, create a `.env` file in the project root:

```env
MY_API_KEY=your_developers.ria.com_api_key
```

## Retraining the Model

From the `src/pipeline` directory:

```bash
python prepare.py
python export_valid_categories.py
python train_tree.py
python train_catboost.py
python train_lightgbm.py
python train_xgboost.py
python cross-validation.py
python compare_results.py
```

To train the final model used by the application:

```bash
python "src/backend/Train final model.py"
```

Restart the API after training.

## Limitations

1. The estimated price is indicative and does not constitute an official vehicle appraisal.
2. The full dataset is not included in the repository due to the paid API access requirements.
3. The market changes over time. Without collecting new data and retraining the model, prediction drift may occur.

---

## Author

**Dmytro Chuchman**

Computer Mathematics student at Taras Shevchenko National University of Kyiv (Faculty of Mechanics and Mathematics).

Interests: Data Science, Machine Learning, and Analytics.
