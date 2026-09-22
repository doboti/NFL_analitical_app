"""Streamlit dashboard: Real-Time NFL Cognitive Engine - 1. Fázis (Statisztikai Agy)."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "win_probability_xgb.json"
META_PATH = ROOT / "models" / "win_probability_meta.json"

st.set_page_config(page_title="NFL Win Probability Engine", page_icon="🏈", layout="centered")


@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta


st.title("🏈 NFL Cognitive Engine — Win Probability")
st.caption("1. Fázis: Statisztikai Agy — nflverse play-by-play adatokon tréningezett XGBoost modell")

if not MODEL_PATH.exists():
    st.error("Nincs betanított modell. Futtasd előbb: `python src/train_win_prob.py`")
    st.stop()

model, meta = load_model()

with st.sidebar:
    st.header("Modell infó")
    st.metric("Pontosság (teszt szezon)", f"{meta['metrics']['accuracy']:.1%}")
    st.metric("Log loss", f"{meta['metrics']['log_loss']:.4f}")
    st.metric("Brier score", f"{meta['metrics']['brier_score']:.4f}")
    st.caption(f"Tréning adat: {meta['train_rows']:,} sor | Teszt szezon: {meta['test_season']}")

st.subheader("Játékhelyzet megadása")

col1, col2 = st.columns(2)
with col1:
    qtr = st.selectbox("Negyed (Qtr)", [1, 2, 3, 4, 5], index=2)
    down = st.selectbox("Down", [0, 1, 2, 3, 4], index=1, help="0 = nincs aktuális down (pl. kickoff)")
    ydstogo = st.number_input("Distance (yard a first downhoz)", min_value=0, max_value=99, value=10)
    yardline_100 = st.slider("Yardline (0=touchdown vonal, 100=saját endzone)", 0, 100, 50)

with col2:
    score_differential = st.number_input("Pontkülönbség (labdabirtokos - védekező csapat)", -50, 50, 0)
    minutes_remaining = st.number_input("Hátralévő idő a negyedben (perc)", 0, 15, 7)
    seconds_remaining_extra = st.number_input("+ másodperc", 0, 59, 0)
    posteam_timeouts = st.selectbox("Labdabirtokos timeoutjai", [0, 1, 2, 3], index=3)
    defteam_timeouts = st.selectbox("Védekező csapat timeoutjai", [0, 1, 2, 3], index=3)
    is_home = st.radio("Labdabirtokos csapat", ["Hazai (Home)", "Vendég (Away)"]) == "Hazai (Home)"

quarter_seconds_remaining = minutes_remaining * 60 + seconds_remaining_extra
quarters_left_after_current = max(0, 4 - qtr)
half_seconds_remaining = quarter_seconds_remaining + (60 * 15 if qtr in (1, 3) else 0)
game_seconds_remaining = quarter_seconds_remaining + quarters_left_after_current * 15 * 60

features = pd.DataFrame([{
    "qtr": qtr,
    "down": down,
    "ydstogo": ydstogo,
    "yardline_100": yardline_100,
    "score_differential": score_differential,
    "half_seconds_remaining": half_seconds_remaining,
    "game_seconds_remaining": game_seconds_remaining,
    "posteam_timeouts_remaining": posteam_timeouts,
    "defteam_timeouts_remaining": defteam_timeouts,
    "is_home": int(is_home),
}])[meta["features"]]

win_prob = model.predict_proba(features)[0, 1]

st.subheader("Eredmény")
st.progress(float(win_prob))
st.metric("Labdabirtokos csapat győzelmi esélye", f"{win_prob:.1%}")
st.metric("Ellenfél győzelmi esélye", f"{1 - win_prob:.1%}")

with st.expander("Bemeneti feature vektor"):
    st.dataframe(features)
