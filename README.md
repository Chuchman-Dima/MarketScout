# MarketScout — оцінка ринкової ціни авто (AUTO.RIA)

Повний цикл ML-продукту: збір оголошень з [AUTO.RIA](https://auto.ria.com) через офіційний API, очищення даних, порівняння моделей, API на FastAPI і інтерфейс на Streamlit.

Застосунок прогнозує **справедливу ринкову ціну** вживаного легкового авто в доларах (з конвертацією в гривні та євро), дає інструменти покупця: перевірка ціни оголошення, кредит, прогноз знецінення, збереження й порівняння кількох авто.

### Спробувати застосунок

* **[Основне посилання (Streamlit)](https://car-price-predictor-autoria-as5etqdr7uhjyugcgbcxmy.streamlit.app/)**
> Якщо сторінка не відкривається одразу, зачекайте 60–120 секунд на «прогрів» безкоштовного сервера й оновіть її.

<img width="1339" height="1144" alt="image" src="https://github.com/user-attachments/assets/83e96f0e-2cab-42ae-99e2-b0fe922e2098" />


### Аналітичний дашборд (Power BI)

* **[Відкрити інтерактивний дашборд](https://app.powerbi.com/view?r=eyJrIjoiMGM0ZGVlMDEtZWI1Mi00ZTFmLWJkMTUtMzc5N2IyNDkyZmU5IiwidCI6ImI3NjdhNzJlLTQwYjYtNDc2ZC1iMTQ0LTI0OGRhMDdjNzFiNCIsImMiOjl9)**

Дашборд показує структуру ринку вживаних авто за даними AUTO.RIA: розподіл цін, популярні марки й моделі, вплив віку та пробігу на вартість.

<img width="1675" height="941" alt="MarketScout" src="https://github.com/user-attachments/assets/03ffd7fa-6459-4975-94e6-e476aec558d1" />


## Технологічний стек

### Мова

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### Збір даних

![AUTO.RIA API](https://img.shields.io/badge/AUTO.RIA_API-E4002B?style=for-the-badge)
![REST API](https://img.shields.io/badge/REST_API-02569B?style=for-the-badge)

### Обробка та аналіз даних

![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Apache Parquet](https://img.shields.io/badge/Parquet-50ABF1?style=for-the-badge&logo=apacheparquet&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

### Машинне навчання

![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02A88E?style=for-the-badge)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=for-the-badge&logoColor=black)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge)
![Optuna](https://img.shields.io/badge/Optuna-2980B9?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-FF0051?style=for-the-badge)
![joblib](https://img.shields.io/badge/joblib-5C6BC0?style=for-the-badge)

### Бекенд

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-499848?style=for-the-badge)

### Фронтенд та візуалізація

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Altair](https://img.shields.io/badge/Altair-1F77B4?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)

### Інфраструктура та деплой

![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Docker Compose](https://img.shields.io/badge/Docker_Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Docker Hub](https://img.shields.io/badge/Docker_Hub-0db7ed?style=for-the-badge&logo=docker&logoColor=white)
![Streamlit Community Cloud](https://img.shields.io/badge/Streamlit_Community_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

### Інструменти розробки

![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![PyCharm](https://img.shields.io/badge/PyCharm-000000?style=for-the-badge&logo=pycharm&logoColor=white)


## Що вміє застосунок

* **Оцінка ринкової вартості.** Модель повертає ціну в USD. Інтерфейс показує орієнтовний діапазон навколо неї (за похибкою моделі) і конвертацію в UAH / EUR.
* **Необов’язкові поля.** Кузов, привід, колір, коробка, пальне, об’єм, ДТП тощо можна лишити як «Не вказано». Порожнє поле не підміняється нулем: для чисел береться типове значення (медіана / мода за маркою й моделлю), для категорій — окрема категорія «Не вказано».
* **Детектор ціни оголошення.** Порівняння ціни продавця з ринковою оцінкою.
* **Кредитний калькулятор.** Перший внесок, строк, ставка, платіж і переплата.
* **Прогноз знецінення.** Ціна на кілька років уперед з урахуванням старіння та приросту пробігу.
* **Збережені авто, пакетна оцінка, порівняння, історія сесії.**

## Як розвивалась модель

Спочатку проєкт тримався на **CatBoost**: збір даних, EDA, одна регресія ціни, FastAPI + Streamlit + Docker. Далі змінили обсяг і склад даних і вже на великому корпусі **порівняли кілька алгоритмів** — у застосунок пішов LightGBM.

### Травень–вересень: CatBoost

**Травень.** Робочий датасет — близько **20 000** оголошень легкових авто. Модель — `CatBoostRegressor` на `log1p(Price_USD)` з оберненим `expm1` у долари. CatBoost обрали через нативні категорії (марка, модель, КПП, пальне) без one-hot. На такій вибірці ринок покривався неповно, викиди (старі авто, нетипові ціни) сильно били по MAE. Групування рідкісних марок і межі викидів спочатку рахувались на всьому train до CV — це трохи завищувало метрики; пороги перенесли всередину кожного фолду.

Уже тоді в продукті були оцінка ціни, валюти, пояснення прогнозу, перевірка оголошення, пальне, знецінення, порівняння авто й деплой.

**Вересень.** Старого обсягу стало замало. Новий парсер (`src/parser/parser.py`) зібрав близько **100 000** оголошень (`data/new_data/new_cars_dataset_2.csv`): кузов, привід, колір, розмитнення, ДТП, локація, тип продавця, умови продажу. Перевірили, як стара CatBoost поводиться на свіжих даних, і **перенавчили CatBoost** уже на великому корпусі. Якість помітно виросла саме за рахунок обсягу й різноманіття, а не заміни алгоритму.

Після цього стало зрозуміло, що «ще одна CatBoost» — не відповідь: потрібне порівняння кількох моделей на **однаково підготовлених** даних.

### Порівняння моделей і вибір LightGBM

Дані готує `src/pipeline/prepare.py` (parquet, split 80/20, рідкісні категорії лише на train). Кандидати: Decision Tree (baseline), CatBoost, LightGBM, XGBoost; плюс спільний CV, Optuna для LightGBM, SHAP.

На тому самому великому датасеті:

| Модель | MAE, USD | Median AE, USD | MAPE, % | R² | CV MAE |
|---|---:|---:|---:|---:|---:|
| XGBoost | 1903 | 949 | 16.16 | 0.920 | 1967 ± 11 |
| **LightGBM** | 1921 | **937** | **15.75** | 0.919 | 1984 ± 10 |
| CatBoost | 2152 | 995 | 16.76 | 0.886 | 2180 ± 18 |
| Decision Tree | 2694 | — | 21.92 | 0.850 | — |

Дерево рішень гірше — бустинг виправданий. CatBoost на цьому пайплайні поступається іншим бустингам. XGBoost і LightGBM майже рівні (різниця MAE ~18 USD). LightGBM кращий за MAPE та стабільніший на CV; Optuna майже не покращила ручні параметри. У прод обрано **LightGBM**: близька якість до XGBoost, зручні категорії, швидкий інференс, SHAP.

Фінальна модель у застосунку — характеристики самого авто (марка, модель, пробіг, вік, КПП, пальне, об’єм, привід, ДТП, розмитнення, перший власник та похідні ознаки). Зі зростанням пробігу ціна не зростає.

## Метрики з часом

Як змінювалась якість разом із даними. Ціна в USD після `expm1`.

| Період | Дані | Модель у проді | Рядків (орієнтовно) | MAE, USD | Median AE, USD | MAPE, % | R² |
|---|---|---|---:|---:|---:|---:|---:|
| Травень | перший корпус, вузький набір полів | CatBoost | ~20 000 | 2 967 | 1 361 | 22.65 | 0.87 |
| Вересень (стара обробка) | нові дані, обробка з травня | CatBoost | ~20 000 | 3 071 | 1 366 | 23.01 | 0.85 |
| Вересень | новий парсер, розширені фічі | CatBoost | ~100 000 | 2 152 | 995 | 16.76 | 0.89 |
| Після порівняння | той самий великий корпус | **LightGBM** | ~77 000 train | **1 921** | **937** | **15.75** | **0.92** |

Median AE для цих моделей — середнє по 5-fold CV;

Не hold-out: травнева модель без перенавчання, з відновленим травневим Other-групуванням, оцінена на вересневих даних.

Від травневої CatBoost до поточної LightGBM MAE знизився приблизно **в 1.5 раза**. Основний стрибок — вересень (більше й свіжіші дані). Другий — зміна алгоритму на тому ж корпусі.

## Дані та ознаки

| | Травень | Вересень і далі |
|---|---|---|
| Джерело | AUTO.RIA API | той самий API, новий парсер |
| Обсяг | ~20 тис. оголошень | ~100 тис. |
| Train / test | менша вибірка | ~76 800 / ~19 200 |
| Ціль | `Price_USD` | `Price_USD` (навчання на `log1p`) |

У модель ідуть, зокрема: марка, модель, пробіг, вік, коробка, пальне, об’єм, привід, кузов, колір, ДТП, розмитнення, перший власник, умови продажу та похідні (`Km_per_Year`, `is_EV`, `is_luxury_brand`, `log_Mileage` тощо).

Не беруться поля, які витікають ціль або описують оголошення, а не авто: `Price_UAH` / `Price_EUR`, перегляди, «топ», дати модерації.

Якщо користувач не вказав об’єм, КПП чи подібне, підставляється типове значення з ринку (медіана або мода), а не нуль.


## Структура репозиторію

```text
├── data/
│   ├── cars_dataset_sample.csv
│   ├── new_data/                    # великий CSV у git не комітиться
│   │   └── README_parser.md
│   └── parquet/
├── docs/
│   └── images/                      # скриншоти для README
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── parser/parser.py
│   ├── pipeline/                    # prepare, train_*, CV, порівняння
│   ├── backend/                     # FastAPI, фінальне навчання, ваги
│   └── frontend/                    # Streamlit
├── docker-compose.yml
└── README.md
```


## Як запустити

Потрібні `lightgbm_pipeline.pkl` і `valid_categories.json` у директорії моделей.

### Docker Compose

```bash
docker compose up --build
```

* API: http://localhost:8080
* UI: http://localhost:8501

### Окремі образи

```bash
docker network create auto_net
docker run -d --name backend --network auto_net -p 8080:8000 dmytrochuchman/auto-ria-backend
docker run -d --name frontend --network auto_net -p 8501:8501 dmytrochuchman/auto-ria-frontend
```

### Локально

1. Бекенд: залежності з `src/backend/requirements.txt`, uvicorn на 8080.
2. Фронтенд: `BACKEND_URL=http://localhost:8080 streamlit run src/frontend/app.py`

Для парсера в корені потрібен `.env`:

```env
MY_API_KEY=ваш_ключ_developers.ria.com
```


## Як перенавчити модель

З каталогу `src/pipeline`:

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

Модель для застосунку:

```bash
python "src/backend/Train final model.py"
```

Після цього перезапустіть API.

## Обмеження

1. Оцінка орієнтовна, не офіційна експертиза.
2. Повний датасет у репозиторії немає (платний API).
3. Ринок змінюється: без дозбору й перенавчання з’являється дрифт.

---

## Автор

**Чучман Дмитро**

Студент КНУ ім. Тараса Шевченка (механіко-математичний факультет).

Сфера інтересів: Data Science, Machine Learning, аналітика.
