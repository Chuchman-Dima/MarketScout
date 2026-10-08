import os
import time
from datetime import datetime

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="AUREA · оцінка авто",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
CURRENT_YEAR = datetime.now().year
UNKNOWN = "Не вказано"
TRI = [UNKNOWN, "Ні", "Так"]

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Manrope:wght@400;500;600;700&display=swap');

    :root {
        --ink: #0c0d10;
        --panel: #14161c;
        --line: rgba(232, 197, 132, 0.18);
        --gold: #e8c584;
        --gold-2: #c9a15b;
        --mist: #b8b4ab;
        --ok: #7dcea0;
        --warn: #e8b86d;
        --bad: #e07a6a;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(1200px 600px at 8% -10%, rgba(232,197,132,.08), transparent 50%),
            radial-gradient(900px 500px at 100% 0%, rgba(90,70,40,.18), transparent 45%),
            var(--ink) !important;
        color: #efece6;
        font-family: 'Manrope', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    footer { visibility: hidden; }
    .block-container { padding-top: 1.4rem; max-width: 1180px; }

    h1, h2, h3, .hero-title { font-family: 'Cormorant Garamond', serif; }

    .hero-kicker {
        letter-spacing: .28em; text-transform: uppercase; font-size: .72rem;
        color: var(--gold-2); text-align: center; margin-bottom: .2rem;
    }
    .hero-title {
        text-align: center; font-size: 3.1rem; font-weight: 600;
        color: #f4efe6; letter-spacing: -0.03em; line-height: 1.05; margin: 0;
    }
    .hero-sub {
        text-align: center; color: var(--mist); font-size: 1.02rem;
        margin: .45rem 0 1.6rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
        border: 1px solid var(--line) !important;
        border-radius: 18px !important;
    }

    .stSelectbox label, .stNumberInput label, .stRadio label, .stSlider label,
    .stTextInput label, .stCheckbox label {
        font-size: .82rem !important; color: #d9d3c7 !important; font-weight: 600 !important;
    }

    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(90deg, #d4af67, #e8c584);
        color: #1a140c; border: 0; border-radius: 999px;
        font-weight: 700; letter-spacing: .04em; padding: .7rem 1.4rem;
    }
    div[data-testid="stButton"] > button[kind="secondary"] {
        background: transparent; color: var(--gold);
        border: 1px solid var(--line); border-radius: 999px;
    }

    .price-hero {
        text-align: center; padding: 1.2rem 0 .4rem;
    }
    .price-hero .amt {
        font-family: 'Cormorant Garamond', serif;
        font-size: 3.4rem; font-weight: 600; color: var(--gold);
        line-height: 1;
    }
    .price-hero .hint { color: var(--mist); margin-top: .35rem; font-size: .9rem; }

    .score-wrap { display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 8px; }
    .score-label { font-size: 0.78rem; color: var(--mist); }

    .status-bar {
        display: flex; align-items: center; gap: 14px; margin-top: 10px;
        padding: 12px 16px; border-radius: 14px;
        background: rgba(232,197,132,.06); border: 1px solid var(--line);
    }
</style>
""", unsafe_allow_html=True)

_defaults = {
    "prediction_done": False,
    "pred_price": 0.0,
    "payload": {},
    "compare_list": [],
    "shap_data": {},
    "saved_cars": [],
    "history": [],
    "run_batch": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def fmt_money(amount: float, currency: str) -> str:
    return f"{int(amount):,} {currency}".replace(",", "\u202f")


def condition_score(age: int, mileage: float, fuel_type: str, gearbox: str, crashed) -> tuple[int, str, str]:
    score = 100
    if age <= 2:
        score -= 0
    elif age <= 5:
        score -= 10
    elif age <= 10:
        score -= 20
    elif age <= 15:
        score -= 32
    else:
        score -= 48

    if mileage <= 50:
        score -= 0
    elif mileage <= 100:
        score -= 8
    elif mileage <= 200:
        score -= 18
    elif mileage <= 300:
        score -= 28
    else:
        score -= 40

    if fuel_type in ("Електро", "Гібрид (HEV)"):
        score += 4
    if gearbox == "Автомат":
        score += 2
    if crashed is True:
        score = max(0, score - 30)
    score = max(0, min(100, score))

    if crashed is True:
        return score, "#e07a6a", "Після ДТП"
    if score >= 80:
        return score, "#7dcea0", "Відмінний"
    if score >= 60:
        return score, "#e8b86d", "Хороший"
    if score >= 40:
        return score, "#e0a070", "Задовільний"
    return score, "#e07a6a", "Слабкий"


def loan_monthly(principal: float, rate_pct: float, months: int) -> float:
    if rate_pct == 0 or months == 0:
        return principal / max(months, 1)
    r = rate_pct / 100 / 12
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)


def _select_index(options: list, value, default: int = 0) -> int:
    try:
        return options.index(value)
    except ValueError:
        return default


def _with_unknown(values: list, extra: str | None = None) -> list:
    seen = []
    for x in values or []:
        s = str(x).strip()
        if not s or s in seen or s in (UNKNOWN, "Other", "Unknown"):
            continue
        seen.append(s)
    out = [UNKNOWN] + seen
    if extra and extra not in out:
        out.append(extra)
    return out


def tri_to_bool(value: str):
    if value == UNKNOWN:
        return None
    return value == "Так"


def ui_to_mark(mark: str) -> str:
    if mark in (UNKNOWN, ""):
        return UNKNOWN
    if mark in ("Інша", "Інше"):
        return "Other"
    return mark


def preload_from_query_params(valid_marks: list[str], mark_model_map: dict) -> dict | None:
    qp = st.query_params
    mark = qp.get("mark")
    if not mark:
        return None

    mark_list = _with_unknown(sorted(valid_marks), extra="Інша")
    if mark not in mark_list:
        mark = UNKNOWN

    model = qp.get("model", UNKNOWN)
    if mark in (UNKNOWN, "Інша"):
        models = [UNKNOWN, "Інша"]
    else:
        models = _with_unknown(mark_model_map.get(mark, []), extra="Інша")
    if model not in models:
        model = UNKNOWN

    try:
        year = int(qp.get("year", "2020"))
    except ValueError:
        year = 2020
    try:
        mileage = int(float(qp.get("mileage", "100")))
    except ValueError:
        mileage = 100

    return {
        "mark": mark,
        "model": model,
        "year": year,
        "mileage": mileage,
        "gearbox": qp.get("gearbox", UNKNOWN),
        "fuel": qp.get("fuel", UNKNOWN),
        "engine": qp.get("engine"),
        "body": qp.get("body", UNKNOWN),
        "drive": qp.get("drive", UNKNOWN),
        "color": qp.get("color", UNKNOWN),
        "crashed": qp.get("crashed", UNKNOWN),
        "custom": qp.get("custom", UNKNOWN),
        "first_owner": qp.get("first_owner", UNKNOWN),
        "exchange": qp.get("exchange", UNKNOWN),
        "bargain": qp.get("bargain", UNKNOWN),
        "urgent": qp.get("urgent", UNKNOWN),
    }


def score_ring_svg(sc: int, sc_color: str, sc_label: str) -> str:
    circ = 2 * 3.14159 * 28
    dash = circ * sc / 100
    return f"""<div class='score-wrap'>
        <svg width='70' height='70' viewBox='0 0 70 70'>
          <circle cx='35' cy='35' r='28' fill='none' stroke='#2a2c33' stroke-width='8'/>
          <circle cx='35' cy='35' r='28' fill='none' stroke='{sc_color}' stroke-width='8'
            stroke-dasharray='{dash:.1f} {circ:.1f}' stroke-linecap='round'
            transform='rotate(-90 35 35)'/>
          <text x='35' y='40' text-anchor='middle'
                font-size='16' font-weight='700' fill='{sc_color}'>{sc}</text>
        </svg>
        <span class='score-label'>{sc_label}</span>
    </div>"""


def build_payload(saved: dict) -> dict:
    age = CURRENT_YEAR - int(saved["year"])
    mileage = float(saved["mileage"])
    fuel = saved.get("fuel") or UNKNOWN
    engine = saved.get("engine")
    if engine in (None, UNKNOWN, ""):
        engine_val = 0.0 if fuel == "Електро" else 0.0
    else:
        engine_val = float(engine)
    return {
        "Mark": ui_to_mark(saved["mark"]),
        "Model": ui_to_mark(saved["model"]),
        "Mileage": mileage,
        "Gearbox": None if saved.get("gearbox") in (None, UNKNOWN) else saved.get("gearbox"),
        "Age": int(age),
        "Fuel_Type": None if fuel in (None, UNKNOWN) else fuel,
        "Engine_Capacity": float(engine_val),
        "Km_per_Year": mileage / (age + 1),
        "is_EV": 1 if fuel == "Електро" else 0,
        "is_suspicious_mileage": 1 if (age > 10 and mileage < 50) else 0,
        "is_new": 1 if age <= 3 else 0,
        "Body_Name": None if saved.get("body") in (None, UNKNOWN) else saved.get("body"),
        "Drive_Name": None if saved.get("drive") in (None, UNKNOWN) else saved.get("drive"),
        "Color_Name": None if saved.get("color") in (None, UNKNOWN) else saved.get("color"),
        "Is_Crashed": tri_to_bool(saved.get("crashed", UNKNOWN)),
        "Custom": tri_to_bool(saved.get("custom", UNKNOWN)),
        "First_Owner": tri_to_bool(saved.get("first_owner", UNKNOWN)),
        "Exchange_Possible": tri_to_bool(saved.get("exchange", UNKNOWN)),
        "Is_Bargain": tri_to_bool(saved.get("bargain", UNKNOWN)),
        "Is_Urgent": tri_to_bool(saved.get("urgent", UNKNOWN)),
    }


@st.cache_data(show_spinner=False)
def load_categories() -> dict | None:
    for attempt in range(3):
        try:
            r = requests.get(f"{BACKEND_URL}/categories", timeout=120)
            if r.status_code == 200:
                return r.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt < 2:
                time.sleep(5)
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_exchange_rates() -> dict:
    default = {"USD": 1.0, "UAH": 43.89, "EUR": 0.852}
    try:
        r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=5)
        if r.status_code == 200:
            d = r.json()
            return {
                "USD": 1.0,
                "UAH": round(d["rates"].get("UAH", default["UAH"]), 2),
                "EUR": round(d["rates"].get("EUR", default["EUR"]), 4),
            }
    except Exception:
        pass
    return default


if "categories_loaded" not in st.session_state:
    _status_placeholder = st.empty()
    with _status_placeholder.status("З'єднання з сервером…", expanded=True) as _status:
        _cats = load_categories()
        if _cats:
            st.session_state.valid_categories = _cats
            st.session_state.categories_loaded = True
            _status.update(label="Готово", state="complete", expanded=False)
        else:
            _status.update(label="Немає зв'язку", state="error")
            st.error(f"Бекенд не відповідає.\n\nURL: `{BACKEND_URL}`")
            st.stop()
    valid_categories = st.session_state.valid_categories
else:
    valid_categories = st.session_state.valid_categories

valid_marks = [m for m in valid_categories.get("valid_marks", []) if m != "Причеп"]
mark_model_map = valid_categories.get("mark_model_mapping", {})
engine_mapping = valid_categories.get("engine_mapping", {})
fuel_mapping = valid_categories.get("fuel_mapping", {})
gearbox_mapping = valid_categories.get("gearbox_mapping", {})
body_types = valid_categories.get("body_types", [])
drive_types = valid_categories.get("drive_types", [])
color_names = valid_categories.get("color_names", [])

default_fuels = ["Бензин", "Дизель", "Електро", "Газ", "Гібрид (HEV)"]
default_capacities = np.arange(1.0, 8.2, 0.2).round(1).tolist()
default_gearboxes = ["Автомат", "Ручна / Механіка", "Робот", "Варіатор", "Тіптронік", "Редуктор"]

st.markdown("<p class='hero-kicker'>Market intelligence</p>", unsafe_allow_html=True)
st.markdown("<h1 class='hero-title'>AUREA</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='hero-sub'>Ринкова оцінка авто за повним набором ознак — без ручних коефіцієнтів.</p>",
    unsafe_allow_html=True,
)

if st.session_state.saved_cars:
    with st.expander("Збережені авто / пакетна оцінка"):
        saved_labels = [f"{c['mark']} {c['model']} ({c['year']})" for c in st.session_state.saved_cars]
        col_sel, col_del = st.columns([3, 1])
        with col_sel:
            chosen = st.selectbox("Оберіть авто", saved_labels, key="saved_selector")
        with col_del:
            st.write("")
            if st.button("Видалити", key="del_saved", type="secondary"):
                idx = saved_labels.index(chosen)
                st.session_state.saved_cars.pop(idx)
                st.rerun()

        col_load, col_batch = st.columns(2)
        with col_load:
            if st.button("Підставити параметри", key="load_saved"):
                idx = saved_labels.index(chosen)
                st.session_state["_preload"] = st.session_state.saved_cars[idx]
                st.rerun()
        with col_batch:
            if st.button("Оцінити всі", key="batch_eval"):
                st.session_state.run_batch = True

        if st.session_state.get("run_batch"):
            with st.spinner("Пакетна оцінка…"):
                payloads = [build_payload(c) for c in st.session_state.saved_cars]
                try:
                    res_batch = requests.post(f"{BACKEND_URL}/predict_batch", json=payloads, timeout=60)
                    if res_batch.status_code == 200:
                        batch_data = res_batch.json().get("results", [])
                        batch_df = pd.DataFrame(batch_data)
                        if not batch_df.empty:
                            batch_df.rename(columns={
                                "mark": "Марка", "model": "Модель",
                                "predicted_price_usd": "Оцінка (USD)",
                                "error": "Помилка",
                            }, inplace=True)
                            batch_df.insert(2, "Рік", [c["year"] for c in st.session_state.saved_cars])
                            batch_df.insert(3, "Пробіг", [c["mileage"] for c in st.session_state.saved_cars])
                            st.success("Готово")
                            st.dataframe(batch_df, use_container_width=True)
                    else:
                        st.error(f"Помилка сервера: {res_batch.status_code}")
                except Exception as e:
                    st.error(f"Помилка запиту: {e}")
                if st.button("Закрити пакетну оцінку"):
                    st.session_state.run_batch = False
                    st.rerun()

if "url_params_applied" not in st.session_state:
    _from_url = preload_from_query_params(valid_marks, mark_model_map)
    if _from_url:
        st.session_state["_preload"] = _from_url
    st.session_state.url_params_applied = True

preload = st.session_state.pop("_preload", None)


def preload_or(key, fallback):
    if not preload:
        return fallback
    val = preload.get(key, fallback)
    return fallback if val is None else val


with st.container(border=True):
    st.markdown("### Характеристики")
    st.caption("Невідомі поля можна лишити як «Не вказано» — модель навчена на пропусках.")
    col1, col2 = st.columns(2)

    with col1:
        mark_list = _with_unknown(sorted(valid_marks), extra="Інша")
        mark = st.selectbox(
            "Марка",
            mark_list,
            index=_select_index(mark_list, preload_or("mark", UNKNOWN)),
            key="car_mark",
        )
        if mark in (UNKNOWN, "Інша"):
            available_models = [UNKNOWN, "Інша"]
        else:
            available_models = _with_unknown(sorted(mark_model_map.get(mark, [])), extra="Інша")
        model_name = st.selectbox(
            "Модель",
            available_models,
            index=_select_index(available_models, preload_or("model", UNKNOWN)),
            key=f"car_model_{mark}",
        )
        year = st.number_input(
            "Рік випуску",
            min_value=1990, max_value=CURRENT_YEAR, step=1,
            value=int(preload["year"]) if preload and preload.get("year") else 2020,
        )
        mileage = st.number_input(
            "Пробіг, тис. км",
            min_value=0, max_value=1000, step=5,
            value=int(preload["mileage"]) if preload and preload.get("mileage") else 100,
        )

    with col2:
        mapped_gb = gearbox_mapping.get(mark, {}).get(model_name, default_gearboxes)
        available_gearboxes = _with_unknown(mapped_gb or default_gearboxes)
        gearbox = st.selectbox(
            "Коробка передач",
            available_gearboxes,
            index=_select_index(available_gearboxes, preload_or("gearbox", UNKNOWN)),
            key=f"car_gearbox_{mark}_{model_name}",
        )

        mapped_fuel = fuel_mapping.get(mark, {}).get(model_name, default_fuels)
        available_fuels = _with_unknown(mapped_fuel or default_fuels)
        fuel_type = st.selectbox(
            "Тип пального",
            available_fuels,
            index=_select_index(available_fuels, preload_or("fuel", UNKNOWN)),
            key=f"car_fuel_{mark}_{model_name}",
        )

        if fuel_type == "Електро":
            st.text_input("Об'єм двигуна, л", value="не потрібен (електро)", disabled=True)
            engine_capacity = 0.0
            engine_ui = 0.0
        else:
            mapped_caps = engine_mapping.get(mark, {}).get(model_name, default_capacities)
            cap_opts = [UNKNOWN] + (mapped_caps or default_capacities)
            eng_pre = preload.get("engine") if preload else None
            try:
                eng_pre = float(eng_pre) if eng_pre not in (None, UNKNOWN, "") else UNKNOWN
            except (TypeError, ValueError):
                eng_pre = UNKNOWN
            engine_ui = st.selectbox(
                "Об'єм двигуна, л",
                cap_opts,
                index=_select_index(cap_opts, eng_pre if eng_pre in cap_opts else UNKNOWN),
                key=f"car_engine_{mark}_{model_name}",
            )
            engine_capacity = 0.0 if engine_ui == UNKNOWN else float(engine_ui)

    with st.expander("Кузов, привід, колір"):
        c3, c4, c5 = st.columns(3)
        body_opts = _with_unknown(body_types, extra="Інше")
        drive_opts = _with_unknown(drive_types)
        color_opts = _with_unknown(color_names, extra="Інший")
        with c3:
            body_name = st.selectbox(
                "Кузов", body_opts,
                index=_select_index(body_opts, preload_or("body", UNKNOWN)),
            )
        with c4:
            drive_name = st.selectbox(
                "Привід", drive_opts,
                index=_select_index(drive_opts, preload_or("drive", UNKNOWN)),
            )
        with c5:
            color_name = st.selectbox(
                "Колір", color_opts,
                index=_select_index(color_opts, preload_or("color", UNKNOWN)),
            )

    with st.expander("Стан і умови продажу"):
        st.caption("Кожне поле можна не заповнювати.")
        r1, r2, r3 = st.columns(3)
        with r1:
            is_crashed = st.radio("Після ДТП", TRI, horizontal=True, index=_select_index(TRI, preload_or("crashed", UNKNOWN)))
            is_custom = st.radio("Розмитнене", TRI, horizontal=True, index=_select_index(TRI, preload_or("custom", UNKNOWN)))
        with r2:
            first_owner = st.radio("Перший власник", TRI, horizontal=True, index=_select_index(TRI, preload_or("first_owner", UNKNOWN)))
            exchange_possible = st.radio("Можливий обмін", TRI, horizontal=True, index=_select_index(TRI, preload_or("exchange", UNKNOWN)))
        with r3:
            is_bargain = st.radio("Можливий торг", TRI, horizontal=True, index=_select_index(TRI, preload_or("bargain", UNKNOWN)))
            is_urgent = st.radio("Терміновий продаж", TRI, horizontal=True, index=_select_index(TRI, preload_or("urgent", UNKNOWN)))

    age_preview = CURRENT_YEAR - year
    sc0, sc0_color, sc0_label = condition_score(
        age_preview, mileage, fuel_type, gearbox, tri_to_bool(is_crashed),
    )
    st.markdown(
        f"""<div class="status-bar">
            <div style="font-size:1.8rem;font-weight:700;color:{sc0_color}">{sc0}</div>
            <div>
                <div style="font-size:.75rem;color:#b8b4ab;">Умовний бал стану</div>
                <div style="font-weight:600;color:{sc0_color};">{sc0_label}</div>
            </div>
            <div style="flex:1;height:7px;border-radius:4px;background:#2a2c33;overflow:hidden">
                <div style="width:{sc0}%;height:100%;background:{sc0_color}"></div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

st.write("")
save_col, btn_col, _ = st.columns([1, 2, 1])
form_snapshot = {
    "mark": mark, "model": model_name, "year": year, "mileage": mileage,
    "fuel": fuel_type, "gearbox": gearbox, "engine": engine_ui if fuel_type != "Електро" else 0.0,
    "body": body_name, "drive": drive_name, "color": color_name,
    "crashed": is_crashed, "custom": is_custom, "first_owner": first_owner,
    "exchange": exchange_possible, "bargain": is_bargain, "urgent": is_urgent,
}
with save_col:
    if st.button("Зберегти", use_container_width=True, type="secondary"):
        st.session_state.saved_cars.append(form_snapshot)
        st.toast(f"{mark} {model_name} збережено")
with btn_col:
    calculate_btn = st.button("Оцінити ринкову ціну", use_container_width=True, type="primary")

if calculate_btn:
    payload = build_payload(form_snapshot)
    progress = st.progress(0, text="Рахуємо оцінку…")
    try:
        progress.progress(40)
        res = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=60)
        progress.progress(90)
        if res.status_code == 200:
            data = res.json()
            st.session_state.pred_price = data["predicted_price_usd"]
            st.session_state.shap_data = data.get("shap_values", {})
            st.session_state.payload = payload
            st.session_state.prediction_done = True
            st.session_state.history.append({
                "Час": datetime.now().strftime("%H:%M:%S"),
                "Авто": f"{mark} {model_name}",
                "Рік": year,
                "Пробіг (тис.)": mileage,
                "Оцінка (USD)": int(data["predicted_price_usd"]),
            })
            progress.empty()
            st.rerun()
        elif res.status_code == 422:
            progress.empty()
            st.error(f"Помилка валідації: {res.json().get('detail', 'Некоректні параметри.')}")
        elif res.status_code == 503:
            progress.empty()
            st.error("Модель ще не готова. Зачекайте кілька секунд.")
        else:
            progress.empty()
            st.error(f"Помилка сервера: {res.status_code} — {res.text[:300]}")
    except requests.exceptions.Timeout:
        progress.empty()
        st.error("Сервер не відповів за 60 секунд.")
    except requests.exceptions.ConnectionError:
        progress.empty()
        st.error(f"Немає зв'язку з бекендом (`{BACKEND_URL}`).")
    except Exception as e:
        progress.empty()
        st.error(f"Несподівана помилка: {e}")

if st.session_state.prediction_done:
    st.markdown("---")
    rates = get_exchange_rates()
    p = st.session_state.payload

    with st.container(border=True):
        col_curr, col_price, col_range, col_score = st.columns([1, 2, 2, 1])
        with col_curr:
            curr = st.radio("Валюта", ["USD", "UAH", "EUR"], key="currency_radio_selector")
            if curr != "USD":
                st.caption(f"1 USD = {rates[curr]:.2f} {curr}")

        price_usd = st.session_state.pred_price
        price_conv = price_usd * rates[curr]
        margin = price_conv * 0.05

        with col_price:
            st.markdown(
                f"<div class='price-hero'><div class='amt'>{fmt_money(price_conv, curr)}</div>"
                f"<div class='hint'>прогноз моделі LightGBM</div></div>",
                unsafe_allow_html=True,
            )
        with col_range:
            st.metric(
                "Орієнтовний діапазон",
                value=fmt_money(price_conv - margin, curr),
                delta=f"до {fmt_money(price_conv + margin, curr)}",
                delta_color="off",
            )
        sc, sc_color, sc_label = condition_score(
            p.get("Age", 0), p.get("Mileage", 0) or 0,
            p.get("Fuel_Type") or "", p.get("Gearbox") or "",
            p.get("Is_Crashed"),
        )
        with col_score:
            st.markdown(score_ring_svg(sc, sc_color, sc_label), unsafe_allow_html=True)

    if p.get("is_suspicious_mileage") == 1:
        st.warning(
            f"Підозрілий пробіг: {p.get('Age')} р. і лише {p.get('Mileage')} тис. км. "
            "Модель врахувала можливе скручування."
        )

    if st.session_state.shap_data:
        with st.expander("Що найбільше вплинуло на ціну"):
            df_shap = pd.DataFrame(
                list(st.session_state.shap_data.items()),
                columns=["Характеристика", "Вплив ($)"],
            ).sort_values("Вплив ($)")
            df_shap["Колір"] = np.where(df_shap["Вплив ($)"] > 0, "#7dcea0", "#e07a6a")
            fig, ax = plt.subplots(figsize=(10, max(3, len(df_shap) * 0.55)))
            fig.patch.set_facecolor("#14161c")
            ax.set_facecolor("#14161c")
            ax.barh(df_shap["Характеристика"], df_shap["Вплив ($)"], color=df_shap["Колір"], height=0.55)
            ax.axvline(0, color="#6f6a62", linewidth=1.0, linestyle="--")
            ax.tick_params(colors="#d9d3c7")
            ax.set_xlabel("Зміна ціни, USD", color="#b8b4ab")
            for spine in ax.spines.values():
                spine.set_color("#3a3d46")
            st.pyplot(fig)

    st.subheader("Інструменти")
    col_tools1, col_tools2 = st.columns(2)
    with col_tools1:
        with st.container(border=True):
            st.markdown("#### Ціна в оголошенні")
            actual_price = st.number_input("USD", min_value=0, value=0, step=100)
            if actual_price > 0:
                diff = actual_price - price_usd
                diff_pct = diff / price_usd * 100
                if diff_pct > 8:
                    st.error(f"Завищена на {diff_pct:.1f}% (${diff:,.0f}).")
                elif diff_pct < -8:
                    st.success(f"Нижче ринку на {abs(diff_pct):.1f}% (${abs(diff):,.0f}).")
                else:
                    st.info("Близько до ринкової оцінки.")
            else:
                st.caption("Введіть ціну продавця для порівняння.")

    with col_tools2:
        with st.container(border=True):
            st.markdown("#### Витрати на пальне")
            annual_mileage = st.slider("Пробіг за рік, тис. км", 1, 100, 15, 1)
            if p.get("is_EV") == 0:
                eng = p.get("Engine_Capacity") or 0
                consumption = eng * 2.5 + 2 if eng > 0 else 8
                yearly_uah = (annual_mileage * 1000 / 100) * consumption * 54
                yearly_usd = int(yearly_uah / rates["UAH"])
                st.info(f"Орієнтовно ${yearly_usd:,}/рік  ·  ~{consumption:.1f} л/100 км".replace(",", " "))
            else:
                ev_uah = (annual_mileage * 1000 / 100) * 18 * 4.32
                ev_usd = int(ev_uah / rates["UAH"])
                st.success(f"Електро · зарядка ~${ev_usd:,}/рік".replace(",", " "))

    with st.container(border=True):
        st.markdown("#### Кредит")
        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            down_pct = st.slider("Перший внесок, %", 0, 80, 20, 5)
        with lc2:
            loan_months = st.selectbox("Термін, міс.", [12, 24, 36, 48, 60, 84], index=2)
        with lc3:
            rate_pct = st.number_input("Ставка, %/рік", 0.0, 50.0, 15.0, 0.5)
        down_usd = price_usd * down_pct / 100
        principal_usd = price_usd - down_usd
        monthly_usd = loan_monthly(principal_usd, rate_pct, loan_months)
        total_pay_usd = monthly_usd * loan_months + down_usd
        overpay_usd = total_pay_usd - price_usd
        lm1, lm2, lm3, lm4 = st.columns(4)
        lm1.metric("Перший внесок", fmt_money(down_usd * rates[curr], curr))
        lm2.metric("Щомісяця", fmt_money(monthly_usd * rates[curr], curr))
        lm3.metric("Всього", fmt_money(total_pay_usd * rates[curr], curr))
        lm4.metric("Переплата", fmt_money(overpay_usd * rates[curr], curr),
                   delta=f"+{overpay_usd / price_usd * 100:.1f}%", delta_color="inverse")

    with st.container(border=True):
        st.markdown(f"#### Знецінення · {annual_mileage} тис. км/рік")
        depr_years = st.slider("Горизонт, років", 1, 20, 5, 1)
        try:
            res_depr = requests.post(
                f"{BACKEND_URL}/predict_depreciation",
                json={"car": p, "annual_mileage": float(annual_mileage), "years": depr_years},
                timeout=30,
            )
            if res_depr.status_code == 200:
                body = res_depr.json()
                depr_data = body.get("depreciation", [])
                if depr_data:
                    df_graph = pd.DataFrame(depr_data)
                    df_graph["Рік"] = df_graph["Year"].apply(lambda x: "Зараз" if x == 0 else f"+{x} р.")
                    chart = (
                        alt.Chart(df_graph)
                        .mark_area(
                            line={"color": "#e8c584"},
                            color=alt.Gradient(
                                gradient="linear",
                                stops=[
                                    alt.GradientStop(color="#e8c584", offset=0),
                                    alt.GradientStop(color="rgba(12,13,16,0)", offset=1),
                                ],
                                x1=1, x2=1, y1=1, y2=0,
                            ),
                        )
                        .encode(
                            x=alt.X("Рік", sort=None, title=""),
                            y=alt.Y("Price", scale=alt.Scale(zero=False), title="Ціна, $"),
                            tooltip=["Рік", "Price"],
                        )
                        .properties(height=280)
                    )
                    st.altair_chart(chart, use_container_width=True)
                    total_loss = body.get("total_loss_usd", depr_data[0]["Price"] - depr_data[-1]["Price"])
                    st.warning(f"Втрата за {depr_years} р.: {fmt_money(total_loss * rates[curr], curr)}")
        except Exception:
            st.caption("Графік знецінення тимчасово недоступний.")

    col_add, _ = st.columns([1, 2])
    with col_add:
        if st.button("Додати до порівняння", use_container_width=True):
            st.session_state.compare_list.append({
                "Марка/Модель": f"{p['Mark']} {p['Model']}",
                "Рік": CURRENT_YEAR - p["Age"],
                "Оцінка (USD)": int(price_usd),
                "Оголошення (USD)": actual_price if actual_price > 0 else "—",
                "Бал стану": sc,
            })
            st.toast("Додано до порівняння")
    if st.session_state.compare_list:
        st.markdown("### Порівняння")
        st.dataframe(pd.DataFrame(st.session_state.compare_list), use_container_width=True)
        if st.button("Очистити порівняння", type="secondary"):
            st.session_state.compare_list = []
            st.rerun()

if st.session_state.history:
    st.markdown("---")
    with st.expander("Історія цієї сесії"):
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
        if st.button("Очистити історію", key="clear_history", type="secondary"):
            st.session_state.history = []
            st.rerun()

st.markdown("---")
st.markdown(
    "<p style='text-align:center;font-size:.78rem;color:#8a857c;'>"
    "Оцінка орієнтовна. Базується на ринкових оголошеннях і не є офіційною експертизою."
    "</p>",
    unsafe_allow_html=True,
)
