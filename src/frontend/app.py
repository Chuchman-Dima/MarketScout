import os
import re
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

# ───────────────────────────── ДИЗАЙН ─────────────────────────────
# Картки = st.container(border=True, key="card_..."): Streamlit додає клас st-key-<key>.
CARD_SEL = '[class*="st-key-card_"]'
RES_SEL = '.st-key-result_card'
MAIN = ':is([data-testid="stMain"], section.main)'

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Onest:wght@400;500;600;700&family=Unbounded:wght@500;600;700&display=swap');
:root {
    --bg: #0a0f1e; --card: #121a30; --card-2: #18223d; --text: #eef2ff; --muted: #a6b4d8;
    --line: rgba(255,255,255,.12); --accent: #2ee6a6; --accent-2: #19c48a; --blue: #5b8cff; --coral: #ff7a59;
    --shadow: 0 14px 34px rgba(0,0,0,.38);
}
.stApp, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(900px 480px at 100% -5%, rgba(91,140,255,.20), transparent 60%),
        radial-gradient(700px 420px at -5% 35%, rgba(46,230,166,.09), transparent 60%),
        var(--bg) !important;
    color: var(--text);
}
.stApp, .stApp input, .stApp button, .stApp textarea { font-family: 'Onest', sans-serif; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }
footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px; }
hr { border-color: var(--line) !important; margin: 1.6rem 0 !important; }

/* ── Базовий колір тексту (незалежно від теми Streamlit) ── */
@M [data-testid="stMarkdownContainer"] p, @M [data-testid="stMarkdownContainer"] li { color: var(--text); }
@M [data-testid="stWidgetLabel"] p { color: var(--muted) !important; font-size: .82rem; font-weight: 600; }
@M [data-testid="stCaptionContainer"] { color: var(--muted); }

/* ── Hero ── */
.hero {
    position: relative; overflow: hidden; border-radius: 28px; padding: 1.8rem 2.2rem 1.9rem;
    margin-bottom: 1.5rem; color: #fff;
    background:
        radial-gradient(620px 320px at 92% 0%, rgba(46,230,166,.30), transparent 60%),
        linear-gradient(135deg, #16276a 0%, #0b5563 100%);
    border: 1px solid rgba(255,255,255,.10); box-shadow: var(--shadow);
}
.hero::after {
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background: repeating-linear-gradient(115deg, rgba(255,255,255,.04) 0 2px, transparent 2px 22px);
}
.hero > * { position: relative; z-index: 1; }
.hero-top { display: flex; align-items: center; gap: .7rem; margin-bottom: 1.5rem; }
.brand-mark {
    width: 34px; height: 34px; border-radius: 11px; background: var(--accent); color: #04261b;
    display: grid; place-items: center; font-size: 1rem; font-weight: 700; box-shadow: 0 6px 18px rgba(46,230,166,.45);
}
.brand-name { font-family: 'Unbounded', sans-serif; font-weight: 700; letter-spacing: .14em; font-size: 1rem; color: #fff; }
.brand-pill {
    margin-left: auto; font-size: .68rem; letter-spacing: .2em; text-transform: uppercase;
    padding: .38rem .85rem; border: 1px solid rgba(255,255,255,.3); border-radius: 999px; color: #e3f0ff;
}
.hero-title {
    font-family: 'Unbounded', sans-serif; font-size: 2.45rem; line-height: 1.12; font-weight: 600;
    margin: 0 0 .7rem; color: #fff; letter-spacing: -.02em; max-width: 760px;
}
.hero-sub { color: #dbe8ff !important; font-size: 1.02rem; max-width: 640px; margin: 0 0 1.3rem; }
.chips { display: flex; flex-wrap: wrap; gap: .5rem; }
.chip {
    font-size: .78rem; font-weight: 600; padding: .42rem .85rem; border-radius: 999px;
    background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.2); color: #fff;
}

/* ── Картки ── */
@CARD {
    background: var(--card) !important; border: 1px solid var(--line) !important;
    border-radius: 24px !important; box-shadow: var(--shadow); padding: 1.35rem 1.6rem !important;
}
@RES {
    background:
        radial-gradient(520px 280px at 100% 0%, rgba(46,230,166,.26), transparent 60%),
        linear-gradient(135deg, #1b3490 0%, #0c6370 100%) !important;
    border: 1px solid rgba(255,255,255,.14) !important; border-radius: 26px !important;
    padding: 1.5rem 1.8rem !important; box-shadow: 0 24px 50px rgba(0,0,0,.45);
}
.card-title { font-family: 'Unbounded', sans-serif; font-weight: 600; font-size: 1.05rem; margin-bottom: .2rem; color: var(--text); }
.card-sub { color: var(--muted); font-size: .86rem; margin-bottom: 1rem; }
.sec { display: flex; align-items: center; gap: .75rem; margin: 1.1rem 0 .7rem; }
.sec-n {
    font-family: 'Unbounded', sans-serif; font-size: .7rem; font-weight: 600; color: var(--accent);
    background: rgba(46,230,166,.14); border: 1px solid rgba(46,230,166,.35); border-radius: 10px; padding: .38rem .52rem;
}
.sec-t { font-weight: 700; font-size: 1rem; color: var(--text); line-height: 1.2; }
.sec-s { font-size: .8rem; color: var(--muted); }

/* ── Поля ── */
div[data-baseweb="select"] > div:first-child,
div[data-baseweb="input"] {
    background: #0c1326 !important; border: 1px solid rgba(255,255,255,.16) !important; border-radius: 12px !important;
    min-height: 44px; box-shadow: none !important; transition: border-color .15s, box-shadow .15s;
}
div[data-baseweb="base-input"] { background: transparent !important; border: 0 !important; }
div[data-baseweb="select"] > div:first-child:hover, div[data-baseweb="input"]:hover { border-color: rgba(255,255,255,.32) !important; }
div[data-baseweb="select"] > div:first-child:focus-within, div[data-baseweb="input"]:focus-within {
    border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(46,230,166,.18) !important;
}
div[data-baseweb="select"] *, div[data-baseweb="input"] input { color: var(--text) !important; }
div[data-baseweb="input"] input:disabled { -webkit-text-fill-color: var(--muted); }
[data-testid="stNumberInput"] button { background: rgba(255,255,255,.07) !important; color: var(--text) !important; border: 0 !important; }
div[data-baseweb="slider"] div[role="slider"] { background: var(--accent) !important; box-shadow: 0 0 0 5px rgba(46,230,166,.22); }
@M [data-testid="stSliderThumbValue"] { color: var(--accent) !important; font-weight: 700; }
@M [data-testid="stSliderTickBarMin"], @M [data-testid="stSliderTickBarMax"] { color: var(--muted) !important; }

/* ── Радіо як сегментовані кнопки ── */
@M [data-testid="stRadio"] [role="radiogroup"] { gap: .4rem; flex-wrap: wrap; }
@M [data-testid="stRadio"] label {
    background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14); border-radius: 999px;
    padding: .3rem .9rem; margin: 0 !important; transition: background .15s; cursor: pointer;
}
@M [data-testid="stRadio"] label:hover { background: rgba(255,255,255,.16); }
@M [data-testid="stRadio"] label > div:first-child:not(:has([data-testid="stMarkdownContainer"])) { display: none !important; }
@M [data-testid="stRadio"] label > div:last-child { margin-left: 0 !important; padding-left: 0 !important; }
@M [data-testid="stRadio"] label p { font-size: .85rem; font-weight: 600; color: #e6ecff !important; }
@M [data-testid="stRadio"] label:has(input:checked) { background: var(--accent); border-color: var(--accent); }
@M [data-testid="stRadio"] label:has(input:checked) p { color: #04261b !important; }
@RES [data-testid="stCaptionContainer"] { color: #cfe3ff !important; }

/* ── Кнопки ── */
button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #5af2bf, var(--accent-2)) !important; color: #04261b !important;
    border: 0 !important; border-radius: 14px !important; font-weight: 700; letter-spacing: .01em;
    padding: .8rem 1.4rem; box-shadow: 0 10px 26px rgba(46,230,166,.30); transition: transform .15s, box-shadow .15s;
}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-1px); box-shadow: 0 14px 32px rgba(46,230,166,.42);
}
button[kind="primary"] p, button[data-testid="stBaseButton-primary"] p { color: #04261b !important; font-weight: 700; }
button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
    background: rgba(255,255,255,.07) !important; color: var(--text) !important; border: 1px solid rgba(255,255,255,.2) !important;
    border-radius: 14px !important; font-weight: 600; padding: .8rem 1.2rem; transition: border-color .15s, background .15s;
}
button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--accent) !important; background: rgba(46,230,166,.10) !important;
}
button[kind="secondary"] p, button[data-testid="stBaseButton-secondary"] p { color: var(--text) !important; }

/* ── Вкладки ── */
@M [role="tablist"] { gap: .35rem; background: var(--card); border: 1px solid var(--line); padding: .3rem; border-radius: 16px; }
@M [role="tab"] { border-radius: 12px !important; height: 44px; padding: 0 1.2rem; background: transparent !important; }
@M [role="tab"] p { color: var(--muted) !important; font-weight: 600; }
@M [role="tab"][aria-selected="true"] { background: var(--accent) !important; }
@M [role="tab"][aria-selected="true"] p { color: #04261b !important; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"], [role="tablist"] div:empty { display: none !important; }
[data-testid="stTabs"] { margin-top: .4rem; }

/* ── Експандери, метрики, алерти, таблиці ── */
[data-testid="stExpander"] details {
    border: 1px solid var(--line) !important; border-radius: 16px !important; background: var(--card-2);
}
@M [data-testid="stExpander"] summary, @M [data-testid="stExpander"] summary p { color: var(--text) !important; font-weight: 600; }
[data-testid="stMetric"] { background: var(--card-2); border: 1px solid var(--line); border-radius: 16px; padding: .8rem 1rem; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color: var(--muted) !important; font-size: .78rem; font-weight: 600; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div { color: var(--text) !important; font-weight: 700; font-size: 1.3rem; }
[data-testid="stAlert"] { border-radius: 14px; border: 1px solid var(--line); }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid var(--line); }

/* ── Бал стану ── */
.status-bar {
    display: flex; align-items: center; gap: 1rem; margin-top: .9rem; padding: .9rem 1.2rem;
    border-radius: 18px; background: var(--card-2); border: 1px solid var(--line);
}
.sb-num { font-family: 'Unbounded', sans-serif; font-size: 1.7rem; font-weight: 700; line-height: 1; }
.sb-lbl { font-size: .68rem; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; }
.sb-val { font-weight: 700; }
.sb-track { flex: 1; height: 9px; border-radius: 99px; background: rgba(255,255,255,.12); overflow: hidden; }
.sb-fill { height: 100%; border-radius: 99px; }

/* ── Результат ── */
.res-eyebrow { font-size: .74rem; letter-spacing: .2em; text-transform: uppercase; color: #8ff5d0; font-weight: 700; padding-top: .55rem; }
.res-label { font-size: .8rem; color: #cfe3ff; font-weight: 600; margin-bottom: .35rem; }
.res-price { font-family: 'Unbounded', sans-serif; font-size: 2.4rem; font-weight: 600; line-height: 1.08; color: #fff; letter-spacing: -.02em; }
.res-hint { color: #cfe3ff; font-size: .85rem; margin-top: .5rem; }
.res-range { display: flex; align-items: center; gap: .7rem; font-family: 'Unbounded', sans-serif; font-size: 1.02rem; font-weight: 500; color: #fff; }
.res-range i { flex: 1; min-width: 22px; height: 4px; border-radius: 99px; background: linear-gradient(90deg, var(--accent), #8fb4ff); }
.score-wrap { display: flex; flex-direction: column; align-items: center; gap: 4px; }
.score-label { font-size: .82rem; font-weight: 700; color: #fff !important; }

/* ── Структура кредиту ── */
.split-bar { display: flex; height: 14px; border-radius: 99px; overflow: hidden; background: rgba(255,255,255,.1); margin: .4rem 0 .6rem; }
.split-legend { display: flex; flex-wrap: wrap; gap: 1.1rem; font-size: .82rem; color: var(--muted); }
.split-legend b { color: var(--text); }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: .4rem; }

.foot { text-align: center; font-size: .8rem; color: var(--muted) !important; padding: .4rem 0 0; }

@media (max-width: 760px) {
    .hero { padding: 1.4rem 1.2rem; border-radius: 22px; }
    .hero-title { font-size: 1.7rem; }
    .brand-pill { display: none; }
    .res-price { font-size: 1.8rem; }
}
"""
CSS = CSS.replace("@CARD", CARD_SEL).replace("@RES", RES_SEL).replace("@M", MAIN)
st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)


def md_html(html: str) -> None:
    """HTML без переносів рядків — markdown не сприйме відступи як код."""
    st.markdown(re.sub(r"\n\s*", "", html.strip()), unsafe_allow_html=True)


def card(name: str, result: bool = False):
    """Картка зі стабільним CSS-класом st-key-<key> (потрібен свіжий Streamlit)."""
    key = "result_card" if result else f"card_{name}"
    try:
        return st.container(border=True, key=key)
    except TypeError:
        return st.container(border=True)


def card_head(title: str, sub: str = "") -> None:
    md_html(f"<div class='card-title'>{title}</div>" + (f"<div class='card-sub'>{sub}</div>" if sub else ""))


def section(num: str, title: str, sub: str = "") -> None:
    md_html(
        f"<div class='sec'><span class='sec-n'>{num}</span><div><div class='sec-t'>{title}</div>"
        + (f"<div class='sec-s'>{sub}</div>" if sub else "")
        + "</div></div>"
    )


# ───────────────────────────── СТАН ─────────────────────────────
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


# ───────────────────────────── ЛОГІКА ─────────────────────────────
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
        return score, "#f87171", "Після ДТП"
    if score >= 80:
        return score, "#34d399", "Відмінний"
    if score >= 60:
        return score, "#fbbf24", "Хороший"
    if score >= 40:
        return score, "#fb923c", "Задовільний"
    return score, "#f87171", "Слабкий"


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


def score_ring_svg(sc: int, sc_color: str, sc_label: str, dark: bool = False) -> str:
    r = 40
    circ = 2 * 3.14159 * r
    dash = circ * sc / 100
    color = sc_color
    track = "rgba(255,255,255,.16)"
    return f"""<div class='score-wrap'>
        <svg width='104' height='104' viewBox='0 0 100 100'>
          <circle cx='50' cy='50' r='{r}' fill='none' stroke='{track}' stroke-width='9'/>
          <circle cx='50' cy='50' r='{r}' fill='none' stroke='{color}' stroke-width='9'
            stroke-dasharray='{dash:.1f} {circ:.1f}' stroke-linecap='round'
            transform='rotate(-90 50 50)'/>
          <text x='50' y='58' text-anchor='middle' font-family='Unbounded, sans-serif'
                font-size='24' font-weight='700' fill='{color}'>{sc}</text>
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


# ───────────────────────────── ЗАВАНТАЖЕННЯ КАТЕГОРІЙ ─────────────────────────────
if "categories_loaded" not in st.session_state:
    with st.status("З'єднання з сервером…", expanded=True) as _status:
        _cats = load_categories()
        if _cats:
            st.session_state.valid_categories = _cats
            st.session_state.categories_loaded = True
            if _status:
                _status.update(label="Готово", state="complete", expanded=False)
        else:
            if _status:
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

# ───────────────────────────── HERO ─────────────────────────────
md_html("""
<div class='hero'>
  <div class='hero-top'>
    <span class='brand-mark'>◆</span>
    <span class='brand-name'>AUREA</span>
    <span class='brand-pill'>Market intelligence</span>
  </div>
  <div class='hero-title'>Ринкова оцінка вашого авто</div>
  <p class='hero-sub'>Ринкова оцінка авто за повним набором ознак — без ручних коефіцієнтів.</p>
  <div class='chips'>
    <span class='chip'>LightGBM</span>
    <span class='chip'>SHAP-пояснення</span>
    <span class='chip'>Прогноз знецінення</span>
    <span class='chip'>Кредитний калькулятор</span>
    <span class='chip'>Пакетна оцінка</span>
    <span class='chip'>USD · UAH · EUR</span>
  </div>
</div>
""")

# ───────────────────────────── ЗБЕРЕЖЕНІ АВТО / ПАКЕТНА ОЦІНКА ─────────────────────────────
if st.session_state.saved_cars:
    with st.expander("Збережені авто / пакетна оцінка"):
        saved_labels = [f"{c['mark']} {c['model']} ({c['year']})" for c in st.session_state.saved_cars]
        col_sel, col_del = st.columns([3, 1])
        with col_sel:
            chosen = st.selectbox("Оберіть авто", saved_labels, key="saved_selector")
        with col_del:
            st.write("")
            if st.button("Видалити", key="del_saved", type="secondary", use_container_width=True):
                idx = saved_labels.index(chosen)
                st.session_state.saved_cars.pop(idx)
                st.rerun()

        col_load, col_batch = st.columns(2)
        with col_load:
            if st.button("Підставити параметри", key="load_saved", use_container_width=True):
                idx = saved_labels.index(chosen)
                st.session_state["_preload"] = st.session_state.saved_cars[idx]
                st.rerun()
        with col_batch:
            if st.button("Оцінити всі", key="batch_eval", use_container_width=True):
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


# ───────────────────────────── ФОРМА ─────────────────────────────
with card("form"):
    card_head("Характеристики", "Невідомі поля можна лишити як «Не вказано» — модель навчена на пропусках.")

    section("01", "Автомобіль", "Марка, модель, рік і пробіг")
    c_mark, c_model, c_year, c_mile = st.columns([1.3, 1.3, 1, 1])

    with c_mark:
        mark_list = _with_unknown(sorted(valid_marks), extra="Інша")
        mark = st.selectbox(
            "Марка",
            mark_list,
            index=_select_index(mark_list, preload_or("mark", UNKNOWN)),
            key="car_mark",
        )
    with c_model:
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
    with c_year:
        year = st.number_input(
            "Рік випуску",
            min_value=1990, max_value=CURRENT_YEAR, step=1,
            value=int(preload["year"]) if preload and preload.get("year") else 2020,
        )
    with c_mile:
        mileage = st.number_input(
            "Пробіг, тис. км",
            min_value=0, max_value=1000, step=5,
            value=int(preload["mileage"]) if preload and preload.get("mileage") else 100,
        )

    section("02", "Двигун і трансмісія", "Коробка, пальне та об'єм двигуна")
    c_gb, c_fuel, c_eng = st.columns(3)

    with c_gb:
        mapped_gb = gearbox_mapping.get(mark, {}).get(model_name, default_gearboxes)
        available_gearboxes = _with_unknown(mapped_gb or default_gearboxes)
        gearbox = st.selectbox(
            "Коробка передач",
            available_gearboxes,
            index=_select_index(available_gearboxes, preload_or("gearbox", UNKNOWN)),
            key=f"car_gearbox_{mark}_{model_name}",
        )
    with c_fuel:
        mapped_fuel = fuel_mapping.get(mark, {}).get(model_name, default_fuels)
        available_fuels = _with_unknown(mapped_fuel or default_fuels)
        fuel_type = st.selectbox(
            "Тип пального",
            available_fuels,
            index=_select_index(available_fuels, preload_or("fuel", UNKNOWN)),
            key=f"car_fuel_{mark}_{model_name}",
        )
    with c_eng:
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

    section("03", "Додатково", "Необов'язково — але чим більше деталей, тим точніша оцінка")

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
    md_html(f"""
    <div class='status-bar'>
        <div class='sb-num' style='color:{sc0_color}'>{sc0}</div>
        <div>
            <div class='sb-lbl'>Умовний бал стану</div>
            <div class='sb-val' style='color:{sc0_color}'>{sc0_label}</div>
        </div>
        <div class='sb-track'><div class='sb-fill' style='width:{sc0}%;background:{sc0_color}'></div></div>
    </div>
    """)

st.write("")
btn_col, save_col = st.columns([3, 1])
form_snapshot = {
    "mark": mark, "model": model_name, "year": year, "mileage": mileage,
    "fuel": fuel_type, "gearbox": gearbox, "engine": engine_ui if fuel_type != "Електро" else 0.0,
    "body": body_name, "drive": drive_name, "color": color_name,
    "crashed": is_crashed, "custom": is_custom, "first_owner": first_owner,
    "exchange": exchange_possible, "bargain": is_bargain, "urgent": is_urgent,
}
with btn_col:
    calculate_btn = st.button("Оцінити ринкову ціну", use_container_width=True, type="primary")
with save_col:
    if st.button("Зберегти авто", use_container_width=True, type="secondary"):
        st.session_state.saved_cars.append(form_snapshot)
        st.toast(f"{mark} {model_name} збережено")

# ───────────────────────────── ЗАПИТ ДО БЕКЕНДУ ─────────────────────────────
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


# ───────────────────────────── SHAP: спливаюче вікно ─────────────────────────────
def _shap_body() -> None:
    df_shap = pd.DataFrame(
        list(st.session_state.shap_data.items()),
        columns=["Характеристика", "Вплив ($)"],
    ).sort_values("Вплив ($)")
    colors = np.where(df_shap["Вплив ($)"] > 0, "#2ee6a6", "#ff7a59")
    fig, ax = plt.subplots(figsize=(10, max(3, len(df_shap) * 0.55)))
    fig.patch.set_facecolor("#121a30")
    ax.set_facecolor("#121a30")
    ax.barh(df_shap["Характеристика"], df_shap["Вплив ($)"], color=colors, height=0.55)
    ax.axvline(0, color="#7f8db3", linewidth=1.0, linestyle="--")
    ax.tick_params(colors="#eef2ff", length=0)
    ax.set_xlabel("Зміна ціни, USD", color="#a6b4d8")
    ax.margins(x=0.2)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#26324f")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#26324f")
    for y_pos, val in enumerate(df_shap["Вплив ($)"]):
        ax.text(
            val, y_pos, f" {val:+,.0f}$ ".replace(",", " "),
            va="center", ha="left" if val >= 0 else "right", fontsize=9, color="#eef2ff",
        )
    st.pyplot(fig)
    plt.close(fig)


_SHAP_MODE = "expander"
if hasattr(st, "dialog"):
    try:
        shap_popup = st.dialog("Що найбільше вплинуло на ціну", width="large")(_shap_body)
        _SHAP_MODE = "dialog"
    except TypeError:
        pass
if _SHAP_MODE == "expander" and hasattr(st, "popover"):
    _SHAP_MODE = "popover"

# ───────────────────────────── РЕЗУЛЬТАТ ─────────────────────────────
if st.session_state.prediction_done:
    st.write("")
    rates = get_exchange_rates()
    p = st.session_state.payload

    with card("result", result=True):
        head_l, head_r = st.columns([3, 2])
        with head_l:
            md_html("<div class='res-eyebrow'>Результат оцінки</div>")
        with head_r:
            curr = st.radio(
                "Валюта", ["USD", "UAH", "EUR"], key="currency_radio_selector",
                horizontal=True, label_visibility="collapsed",
            )
            if curr != "USD":
                st.caption(f"1 USD = {rates[curr]:.2f} {curr}")

        price_usd = st.session_state.pred_price
        price_conv = price_usd * rates[curr]
        margin = price_conv * 0.05

        col_price, col_range, col_score = st.columns([2.4, 2, 1])
        with col_price:
            md_html(
                f"<div class='res-label'>Прогнозована ринкова ціна</div>"
                f"<div class='res-price'>{fmt_money(price_conv, curr)}</div>"
                f"<div class='res-hint'>прогноз моделі LightGBM</div>"
            )
        with col_range:
            md_html(
                f"<div class='res-label'>Орієнтовний діапазон · ±5%</div>"
                f"<div class='res-range'><span>{fmt_money(price_conv - margin, curr)}</span><i></i>"
                f"<span>{fmt_money(price_conv + margin, curr)}</span></div>"
            )
        sc, sc_color, sc_label = condition_score(
            p.get("Age", 0), p.get("Mileage", 0) or 0,
            p.get("Fuel_Type") or "", p.get("Gearbox") or "",
            p.get("Is_Crashed"),
        )
        with col_score:
            md_html(score_ring_svg(sc, sc_color, sc_label, dark=True))

    if p.get("is_suspicious_mileage") == 1:
        st.warning(
            f"Підозрілий пробіг: {p.get('Age')} р. і лише {p.get('Mileage')} тис. км. "
            "Модель врахувала можливе скручування."
        )

    tab_price, tab_loan, tab_depr, tab_cmp = st.tabs(
        ["Аналітика", "Кредит", "Знецінення", "Порівняння"]
    )

    # ── Аналітика: ціна в оголошенні, пальне, SHAP ──
    with tab_price:
        col_tools1, col_tools2 = st.columns(2)
        with col_tools1:
            with card("price"):
                card_head("Ціна в оголошенні", "Порівняйте ціну продавця з ринковою оцінкою")
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
            with card("fuel"):
                card_head("Витрати на пальне", "Річні витрати залежно від пробігу")
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

        if st.session_state.shap_data:
            with card("shap"):
                card_head("Що впливає на ціну", "Пояснення моделі (SHAP) — відкривається лише за вашим бажанням")
                if _SHAP_MODE == "dialog":
                    if st.button("Показати вплив характеристик", key="open_shap", type="secondary"):
                        shap_popup()
                elif _SHAP_MODE == "popover":
                    with st.popover("Показати вплив характеристик"):
                        _shap_body()
                else:
                    with st.expander("Показати вплив характеристик"):
                        _shap_body()

    # ── Кредит ──
    with tab_loan:
        with card("loan"):
            card_head("Кредитний калькулятор", "Розрахунок платежів для оціненої вартості авто")
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

            _tot = max(total_pay_usd, 1e-9)
            w_down = down_usd / _tot * 100
            w_prin = principal_usd / _tot * 100
            w_over = max(overpay_usd, 0) / _tot * 100
            st.write("")
            md_html(f"""
            <div class='sb-lbl'>Структура загальних виплат</div>
            <div class='split-bar'>
                <div style='width:{w_down:.2f}%;background:#5b8cff'></div>
                <div style='width:{w_prin:.2f}%;background:#2ee6a6'></div>
                <div style='width:{w_over:.2f}%;background:#ff7a59'></div>
            </div>
            <div class='split-legend'>
                <span><span class='dot' style='background:#5b8cff'></span>Внесок <b>{w_down:.0f}%</b></span>
                <span><span class='dot' style='background:#2ee6a6'></span>Тіло кредиту <b>{w_prin:.0f}%</b></span>
                <span><span class='dot' style='background:#ff7a59'></span>Переплата <b>{w_over:.0f}%</b></span>
            </div>
            """)

    # ── Знецінення ──
    with tab_depr:
        with card("depr"):
            card_head(f"Знецінення · {annual_mileage} тис. км/рік", "Прогноз вартості авто на горизонті кількох років")
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
                                line={"color": "#2ee6a6", "strokeWidth": 3},
                                color=alt.Gradient(
                                    gradient="linear",
                                    stops=[
                                        alt.GradientStop(color="rgba(46,230,166,0.38)", offset=0),
                                        alt.GradientStop(color="rgba(46,230,166,0)", offset=1),
                                    ],
                                    x1=1, x2=1, y1=1, y2=0,
                                ),
                            )
                            .encode(
                                x=alt.X("Рік", sort=None, title="", axis=alt.Axis(labelAngle=0)),
                                y=alt.Y("Price", scale=alt.Scale(zero=False), title="Ціна, $"),
                                tooltip=["Рік", "Price"],
                            )
                            .properties(height=300)
                            .configure(
                                background="rgba(0,0,0,0)",
                                view={"strokeWidth": 0},
                                axis={
                                    "labelColor": "#c3cfee", "titleColor": "#c3cfee",
                                    "gridColor": "rgba(255,255,255,0.10)", "domainColor": "rgba(255,255,255,0.2)",
                                    "tickColor": "rgba(255,255,255,0.2)", "labelFontSize": 12,
                                },
                            )
                        )
                        st.altair_chart(chart, use_container_width=True, theme=None)
                        total_loss = body.get("total_loss_usd", depr_data[0]["Price"] - depr_data[-1]["Price"])
                        st.warning(f"Втрата за {depr_years} р.: {fmt_money(total_loss * rates[curr], curr)}")
            except Exception:
                st.caption("Графік знецінення тимчасово недоступний.")

    # ── Порівняння ──
    with tab_cmp:
        with card("cmp"):
            card_head("Порівняння авто", "Додавайте оцінки, щоб бачити їх поруч")
            col_add, _ = st.columns([1, 2])
            with col_add:
                if st.button("Додати до порівняння", use_container_width=True, type="primary"):
                    st.session_state.compare_list.append({
                        "Марка/Модель": f"{p['Mark']} {p['Model']}",
                        "Рік": CURRENT_YEAR - p["Age"],
                        "Оцінка (USD)": int(price_usd),
                        "Оголошення (USD)": actual_price if actual_price > 0 else "—",
                        "Бал стану": sc,
                    })
                    st.toast("Додано до порівняння")
            if st.session_state.compare_list:
                st.dataframe(pd.DataFrame(st.session_state.compare_list), use_container_width=True)
                if st.button("Очистити порівняння", type="secondary"):
                    st.session_state.compare_list = []
                    st.rerun()
            else:
                st.caption("Список порівняння поки порожній.")

# ───────────────────────────── ІСТОРІЯ ─────────────────────────────
if st.session_state.history:
    st.write("")
    with st.expander("Історія цієї сесії"):
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
        if st.button("Очистити історію", key="clear_history", type="secondary"):
            st.session_state.history = []
            st.rerun()

st.markdown("---")
md_html(
    "<p class='foot'>Оцінка орієнтовна. Базується на ринкових оголошеннях і не є офіційною експертизою.</p>"
)