"""
Modul: Output & Dashboard UI (felhasználóbarát verzió, menürendszerrel).

Két, EGYMÁSTÓL FÜGGETLEN mód között lehet váltani a bal oldali menüben:

  1. "Élő elemzés" - a Kafka nfl.metadata / nfl.winprob / nfl.detections /
     nfl.highlights topicokból élő pipeline (OCR, YOLO, highlight-vágás)
     kimenete. Ez a videó-alapú, formáció-/play-tanuló rész.
  2. "Szezon predikciók" - a src/train_pregame_model.py által előre
     kiszámolt, jövőbeli meccsekre vonatkozó predikciók (győz/veszít esély,
     dobott/futott yard, sack, ÉS a predikció mögötti indoklás) - ez a
     jelenlegi szezon meccseinek megtippelésére szolgál, KÜLÖN a fenti
     videó-elemzéstől.

Az "Élő elemzés" mód UX-fejlesztései:
  - `@st.fragment(run_every=...)` az `st.rerun()`-alapú, TELJES oldal-
    újratöltés helyett - csak a Kafka-adatot kiíró rész frissül
    másodpercenként, a fejléc/elrendezés nem villog/ugrik vissza tetejére.
  - Csapatszínes, TV-scoreboard-szerű fejléc (data/team_colors.csv alapján).
  - Win Probability TREND-grafikon (nem csak a pillanatnyi szám).
  - A highlight klipek TÉNYLEGESEN LEJÁTSZHATÓK a dashboardon (`st.video`),
    nem csak fájlútvonalként kiírva - ehhez kell a data/ kötet mount.
  - Élő/várakozó/hiba állapotjelző (mikor érkezett utoljára adat).
"""
import json
import time
import uuid
from collections import deque
from pathlib import Path

import pandas as pd
import streamlit as st
from kafka import KafkaConsumer

from kafka_utils import (
    BOOTSTRAP_SERVERS, TOPIC_DETECTIONS, TOPIC_HIGHLIGHTS, TOPIC_METADATA, TOPIC_WINPROB,
    wait_for_kafka,
)

st.set_page_config(page_title="NFL Cognitive Engine", page_icon="🏈", layout="wide")

TEAM_COLORS_CSV = Path("/app/data/team_colors.csv")
PREDICTIONS_PATH = Path("/app/data/predictions_latest.json")
STALE_AFTER_SEC = 15
DEFAULT_TEAM_COLOR = "#4a4a4a"


@st.cache_resource
def get_team_colors() -> dict:
    if not TEAM_COLORS_CSV.exists():
        return {}
    df = pd.read_csv(TEAM_COLORS_CSV)
    return dict(zip(df["team_abbr"], df["team_color"]))


@st.cache_resource
def get_consumer():
    import json

    wait_for_kafka()
    return KafkaConsumer(
        TOPIC_METADATA, TOPIC_WINPROB, TOPIC_DETECTIONS, TOPIC_HIGHLIGHTS,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=f"dashboard-{uuid.uuid4()}",
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms=300,
    )


def _init_state():
    defaults = {
        "metadata": None,
        "winprob": None,
        "detections": None,
        "highlights": [],
        "winprob_history": deque(maxlen=300),
        "last_update": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()
TEAM_COLORS = get_team_colors()

st.sidebar.title("🏈 NFL Cognitive Engine")
mode = st.sidebar.radio(
    "Mód", ["🔴 Élő elemzés", "🔮 Szezon predikciók"],
    help="Élő elemzés: videó-alapú OCR/YOLO/highlight pipeline. "
         "Szezon predikciók: a jelenlegi szezon jövőbeli meccseinek tippjei.",
)


def _team_score_card(team: str, score, color: str) -> str:
    return (
        f"<div style='background:{color};color:white;padding:18px 12px;"
        f"border-radius:10px;text-align:center'>"
        f"<div style='font-size:15px;letter-spacing:1px;opacity:0.85'>{team or '?'}</div>"
        f"<div style='font-size:52px;font-weight:800;line-height:1.1'>{score if score is not None else '-'}</div>"
        f"</div>"
    )


def render_scoreboard(meta):
    if not meta or not meta.get("away_team"):
        st.info("⏳ Várakozás OCR adatra a scoreboard felismeréséhez...")
        return

    away_color = TEAM_COLORS.get(meta["away_team"], DEFAULT_TEAM_COLOR)
    home_color = TEAM_COLORS.get(meta["home_team"], DEFAULT_TEAM_COLOR)

    c1, c2, c3 = st.columns([2, 1.2, 2])
    with c1:
        st.markdown(_team_score_card(meta["away_team"], meta.get("away_score"), away_color),
                    unsafe_allow_html=True)
    with c2:
        quarter = meta.get("quarter") or "-"
        clock = meta.get("clock") or "--:--"
        down = meta.get("down")
        distance = meta.get("distance")
        down_distance = f"{down} & {distance}" if down is not None and distance is not None else "-"
        st.markdown(
            f"<div style='text-align:center;padding-top:6px'>"
            f"<div style='font-size:22px;font-weight:700'>Q{quarter}</div>"
            f"<div style='font-size:26px;font-family:monospace'>{clock}</div>"
            f"<div style='font-size:15px;color:gray;margin-top:6px'>{down_distance}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(_team_score_card(meta["home_team"], meta.get("home_score"), home_color),
                    unsafe_allow_html=True)


def render_winprob(wp, meta):
    st.subheader("📊 Win Probability")
    if not wp:
        st.info("⏳ Várakozás Win Probability adatra...")
        return

    posteam = None
    if meta:
        posteam = meta.get("away_team")  # lásd fusion_service POSTEAM_ASSUMPTION
    color = TEAM_COLORS.get(posteam, DEFAULT_TEAM_COLOR) if posteam else DEFAULT_TEAM_COLOR

    st.markdown(
        f"<div style='font-size:34px;font-weight:800;color:{color}'>"
        f"{wp['posteam_win_prob']:.1%}</div>",
        unsafe_allow_html=True,
    )
    st.progress(wp["posteam_win_prob"])
    st.caption(wp["situation"])

    if len(st.session_state.winprob_history) > 1:
        hist_df = pd.DataFrame(st.session_state.winprob_history).set_index("t")
        st.line_chart(hist_df["win_prob"], height=180)
    else:
        st.caption("A trend-grafikon néhány adatpont után jelenik meg.")


def render_detections(det):
    st.subheader("🏃 Detektált játékosok (YOLO)")
    if not det:
        st.info("⏳ Várakozás detekciós adatra...")
        return

    st.metric("Összes detektált játékos", det["num_players"])
    chart_df = pd.DataFrame({
        "oldal": ["Támadás", "Védekezés"],
        "létszám": [det["offense_count"], det["defense_count"]],
    }).set_index("oldal")
    st.bar_chart(chart_df, height=180, color="#d62728")
    st.caption(f"t = {det['timestamp_sec']:.1f}s")


def render_highlights(highlights: list):
    st.subheader("🎬 Highlight klipek")
    if not highlights:
        st.caption("Még nincs highlight esemény.")
        return

    for h in reversed(highlights[-5:]):
        tag = "✅ Megerősítve (eredmény változott)" if h["confirmed"] else "🔊 Csak audió"
        st.markdown(f"**[{h['trigger_time_sec']:.1f}s]** {tag}")
        clip_path = Path(h["clip_path"])
        if clip_path.exists():
            st.video(str(clip_path))
        else:
            st.caption(f"(a klip fájl innen nem érhető el: {h['clip_path']})")
        st.divider()


def render_live_mode():
    global status_placeholder
    st.title("🏈 NFL Cognitive Engine — Live Dashboard")
    st.caption("Videó-alapú OCR/YOLO/highlight pipeline élő kimenete "
               "(formáció-/play-tanuláshoz, nem a szezon-predikcióhoz)")
    status_placeholder = st.empty()
    live_panel()


@st.fragment(run_every=1)
def live_panel():
    connection_error = None
    try:
        consumer = get_consumer()
        got_message = False
        for msg in consumer:
            got_message = True
            if msg.topic == TOPIC_METADATA:
                st.session_state.metadata = msg.value
            elif msg.topic == TOPIC_WINPROB:
                st.session_state.winprob = msg.value
                st.session_state.winprob_history.append(
                    {"t": msg.value["timestamp_sec"], "win_prob": msg.value["posteam_win_prob"]}
                )
            elif msg.topic == TOPIC_DETECTIONS:
                st.session_state.detections = msg.value
            elif msg.topic == TOPIC_HIGHLIGHTS:
                st.session_state.highlights.append(msg.value)
        if got_message:
            st.session_state.last_update = time.time()
    except Exception as exc:
        connection_error = str(exc)
        # A cache-elt consumer megsérülhetett (átmeneti Kafka-hiba) - eldobjuk,
        # hogy a következő fragment-frissítés újat hozzon létre.
        get_consumer.clear()

    is_live = (
        st.session_state.last_update is not None
        and (time.time() - st.session_state.last_update) < STALE_AFTER_SEC
    )
    if connection_error:
        status_placeholder.warning(f"🔴 Átmeneti Kafka-kapcsolati hiba, automatikus újrapróbálkozás... ({connection_error[:100]})")
    elif is_live:
        status_placeholder.success("🟢 Élő - adat folyamatosan érkezik")
    elif st.session_state.last_update is not None:
        status_placeholder.warning("🟡 Nincs friss adat az elmúlt 15 másodpercben...")
    else:
        status_placeholder.info("🟡 Várakozás az első adatra...")

    render_scoreboard(st.session_state.metadata)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        render_winprob(st.session_state.winprob, st.session_state.metadata)
    with col2:
        render_detections(st.session_state.detections)

    st.divider()
    render_highlights(st.session_state.highlights)


# ---------------------------------------------------------------------------
# "Szezon predikciók" mód
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def load_predictions() -> list:
    if not PREDICTIONS_PATH.exists():
        return []
    with open(PREDICTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _confidence_badge(confidence: str) -> str:
    return {
        "confirmed_healthy": "✅ egészséges kezdő",
        "inferred_due_to_injury": "⚠️ sérülés miatt becsült tartalék",
        "starter_injured_no_known_backup": "❓ kezdő sérült, tartalék ismeretlen",
        "no_history": "❓ nincs historikus adat",
    }.get(confidence, confidence or "?")


def render_prediction_card(pred: dict):
    home, away = pred["home_team"], pred["away_team"]
    home_prob = pred["home_win_prob"]
    home_color = TEAM_COLORS.get(home, DEFAULT_TEAM_COLOR)
    away_color = TEAM_COLORS.get(away, DEFAULT_TEAM_COLOR)

    actual = pred.get("actual")

    with st.container(border=True):
        st.markdown(f"### {away} @ {home}")
        if actual:
            predicted_home_win = home_prob > 0.5
            hit = predicted_home_win == actual["home_win"]
            badge = "✅ Predikció talált" if hit else "❌ Predikció tévedett"
            st.caption(
                f"🕑 **Már lejátszva** - valós végeredmény: {home} {actual['home_score']} - "
                f"{actual['away_score']} {away} · {badge}"
            )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"<div style='background:{away_color};color:white;padding:10px;border-radius:8px;"
                f"text-align:center'><b>{away}</b><br><span style='font-size:28px'>{1 - home_prob:.0%}</span></div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"<div style='background:{home_color};color:white;padding:10px;border-radius:8px;"
                f"text-align:center'><b>{home}</b><br><span style='font-size:28px'>{home_prob:.0%}</span></div>",
                unsafe_allow_html=True,
            )

        st.progress(home_prob)

        hs, as_ = pred["home_starter"], pred["away_starter"]
        st.caption(
            f"Kezdő IR - {home}: **{hs.get('name') or '?'}** ({_confidence_badge(hs.get('confidence'))}) | "
            f"{away}: **{as_.get('name') or '?'}** ({_confidence_badge(as_.get('confidence'))})"
        )

        st.markdown("**Statisztika-predikció**" + (" vs. valós" if actual else ""))
        stats = pred["stats"]
        if actual:
            a_stats = actual["stats"]
            stat_df = pd.DataFrame({
                "Csapat": [home, away],
                "Dobott yard (pred / valós)": [
                    f"{stats['home']['passing_yards']:.0f} / {a_stats['home']['passing_yards']:.0f}",
                    f"{stats['away']['passing_yards']:.0f} / {a_stats['away']['passing_yards']:.0f}",
                ],
                "Futott yard (pred / valós)": [
                    f"{stats['home']['rushing_yards']:.0f} / {a_stats['home']['rushing_yards']:.0f}",
                    f"{stats['away']['rushing_yards']:.0f} / {a_stats['away']['rushing_yards']:.0f}",
                ],
                "Elszenvedett sack (pred / valós)": [
                    f"{stats['home']['sacks_taken']:.1f} / {a_stats['home']['sacks_taken']:.0f}",
                    f"{stats['away']['sacks_taken']:.1f} / {a_stats['away']['sacks_taken']:.0f}",
                ],
            })
        else:
            stat_df = pd.DataFrame({
                "Csapat": [home, away],
                "Dobott yard": [stats["home"]["passing_yards"], stats["away"]["passing_yards"]],
                "Futott yard": [stats["home"]["rushing_yards"], stats["away"]["rushing_yards"]],
                "Elszenvedett sack": [stats["home"]["sacks_taken"], stats["away"]["sacks_taken"]],
            }).round(1)
        st.dataframe(stat_df, hide_index=True, use_container_width=True)

        st.markdown("**Miért ez a predikció:**")
        for reason in pred["reasons"]:
            arrow = "🔺" if reason["contribution"] > 0 else "🔻"
            st.markdown(f"- {arrow} {reason['text']}")


def render_predictions_mode():
    st.title("🔮 Szezon predikciók")
    st.caption("A src/train_pregame_model.py által előre kiszámolt, "
               "jövőbeli meccsekre vonatkozó tippek indoklással.")

    predictions = load_predictions()
    if not predictions:
        st.warning(
            "Nincs elmentett predikció. Futtasd le a modellt előbb:\n\n"
            "```\npython src/train_pregame_model.py\n```"
        )
        return

    season, week = predictions[0]["season"], predictions[0]["week"]
    st.info(f"📅 {season} szezon, {week}. hét ({len(predictions)} meccs)")

    already_played = [p for p in predictions if p.get("actual")]
    if already_played:
        st.subheader("🧪 Modell-hatékonyság (már eldőlt meccseken)")
        st.caption("Ezekre a meccsekre a modell UGYANÚGY, kizárólag a meccs előtti "
                   "adatokból adott predikciót - a valós eredmény csak utólag, "
                   "összehasonlításképp van feltüntetve.")
        for pred in sorted(already_played, key=lambda p: p["home_win_prob"], reverse=True):
            render_prediction_card(pred)
        st.divider()
        st.subheader("🔮 Még hátralévő meccsek")

    for pred in sorted(predictions, key=lambda p: p["home_win_prob"], reverse=True):
        if pred.get("actual"):
            continue
        render_prediction_card(pred)


if mode == "🔴 Élő elemzés":
    render_live_mode()
else:
    render_predictions_mode()
