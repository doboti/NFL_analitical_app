"""
Modul: Output & Dashboard UI.

Élő Streamlit felület, ami a Kafka nfl.metadata / nfl.winprob / nfl.detections
/ nfl.highlights topicokat figyeli és valós időben megjeleníti az állapotot.
"""
import time
import uuid

import streamlit as st
from kafka import KafkaConsumer

from kafka_utils import (
    BOOTSTRAP_SERVERS, TOPIC_DETECTIONS, TOPIC_HIGHLIGHTS, TOPIC_METADATA, TOPIC_WINPROB,
    wait_for_kafka,
)

st.set_page_config(page_title="NFL Cognitive Engine - Live Dashboard", page_icon="🏈", layout="wide")


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


if "metadata" not in st.session_state:
    st.session_state.metadata = None
    st.session_state.winprob = None
    st.session_state.detections = None
    st.session_state.highlights = []

connection_error = None
try:
    consumer = get_consumer()
    for msg in consumer:
        if msg.topic == TOPIC_METADATA:
            st.session_state.metadata = msg.value
        elif msg.topic == TOPIC_WINPROB:
            st.session_state.winprob = msg.value
        elif msg.topic == TOPIC_DETECTIONS:
            st.session_state.detections = msg.value
        elif msg.topic == TOPIC_HIGHLIGHTS:
            st.session_state.highlights.append(msg.value)
except Exception as exc:
    connection_error = str(exc)
    # A cache-elt consumer objektum megsérülhetett (pl. Kafka átmeneti
    # kapcsolat-vesztés) - eldobjuk, hogy a következő rerun újat hozzon létre.
    get_consumer.clear()

st.title("🏈 NFL Cognitive Engine — Live Dashboard")
st.caption("5. Fázis: Kafka-alapú mikroszolgáltatás-architektúra élő kimenete")

if connection_error:
    st.warning(f"Átmeneti Kafka-kapcsolati hiba, automatikus újrapróbálkozás... ({connection_error})")

meta = st.session_state.metadata
wp = st.session_state.winprob
det = st.session_state.detections

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📋 Játékhelyzet (OCR)")
    if meta:
        st.metric("Negyed / Óra", f"Q{meta['quarter']} {meta['clock']}")
        st.metric("Down & Distance", f"{meta['down']} & {meta['distance']}")
        st.metric("Eredmény", f"{meta['away_team']} {meta['away_score']} - "
                               f"{meta['home_score']} {meta['home_team']}")
    else:
        st.info("Várakozás OCR adatra...")

with col2:
    st.subheader("📊 Win Probability")
    if wp:
        st.metric("Labdabirtokos győzelmi esélye", f"{wp['posteam_win_prob']:.1%}")
        st.progress(wp["posteam_win_prob"])
        st.caption(wp["situation"])
    else:
        st.info("Várakozás Win Probability adatra...")

with col3:
    st.subheader("🏃 Detektált játékosok (YOLO)")
    if det:
        st.metric("Támadás / Védekezés", f"{det['offense_count']} / {det['defense_count']}")
        st.caption(f"Összesen {det['num_players']} játékos, t={det['timestamp_sec']:.1f}s")
    else:
        st.info("Várakozás detekciós adatra...")

st.divider()
st.subheader("🎬 Highlight klipek")
if st.session_state.highlights:
    for h in reversed(st.session_state.highlights):
        tag = "✅ Megerősítve" if h["confirmed"] else "🔊 Csak audió"
        st.write(f"**[{h['trigger_time_sec']:.1f}s]** {tag} — ratio={h['ratio']}x — `{h['clip_path']}`")
else:
    st.caption("Még nincs highlight esemény.")

time.sleep(1)
st.rerun()
