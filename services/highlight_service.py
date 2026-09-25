"""
Modul 4: Highlight Generator mikroszolgáltatás.

Egyszerre hallgatja a nfl.raw.audio (RMS-trigger) és a nfl.metadata (eredmény-
történet, a vizuális megerősítéshez) topicokat. Trigger esetén a megosztott
videófájlból (kötet) FFmpeg-gel kivágja a highlightot.

FONTOS "live" STREAM_MODE-ban: a highlight klip (trigger-15s .. trigger+20s)
"utó-részét" (+20s) FIZIKAILAG NEM lehet azonnal kivágni élő adásnál, mert az
a jövőben van - még nem történt meg, amikor a trigger észlelésre kerül. Ezért
"live" módban a vágást NEM azonnal végezzük el, hanem egy várólistára tesszük,
és csak akkor hajtjuk végre, amikor a folyamatosan érkező üzenetek alapján
biztosan eltelt legalább POST_ROLL_SEC a trigger óta (tehát a felvétel már
tartalmazza az utó-részt is). "file" módban (ahol a teljes klip már eleve
adott) ez a késleltetés nem szükséges, ott azonnal vágunk.
"""
import base64
import json
import os
import subprocess
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from kafka import KafkaConsumer

from kafka_utils import (
    BOOTSTRAP_SERVERS, TOPIC_HIGHLIGHTS, TOPIC_METADATA, TOPIC_RAW_AUDIO,
    consume_forever, make_producer, wait_for_kafka,
)

SERVICE_NAME = "highlight_service"
STREAM_MODE = os.environ.get("STREAM_MODE", "file").lower()
VIDEO_PATH = Path(os.environ.get(
    "VIDEO_PATH",
    "/app/data/video/live_recording.mp4" if STREAM_MODE == "live" else "/app/data/video/test_clip_av.mp4",
))
HIGHLIGHTS_DIR = Path("/app/data/highlights")

BASELINE_WINDOW = 20  # 0.5s-es chunkokban -> 10 mp
TRIGGER_RATIO = 1.3
MIN_ABS_RMS = 0.05
COOLDOWN_SEC = 20.0
PRE_ROLL_SEC = 15.0
POST_ROLL_SEC = 20.0
POST_ROLL_SAFETY_BUFFER_SEC = 3.0  # kis ráhagyás, hogy a felvétel biztosan tartalmazza az utolsó másodperceket is


def cut_clip(trigger_time: float, confirmed: bool) -> Optional[str]:
    HIGHLIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    start = max(0.0, trigger_time - PRE_ROLL_SEC)
    duration = PRE_ROLL_SEC + POST_ROLL_SEC
    tag = "confirmed" if confirmed else "audio_only"
    out_path = HIGHLIGHTS_DIR / f"highlight_{trigger_time:.0f}s_{tag}.mp4"
    result = subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-i", str(VIDEO_PATH), "-t", str(duration),
         "-c", "copy", str(out_path)],
        capture_output=True,
    )
    if result.returncode != 0 or not out_path.exists():
        print(f"[{SERVICE_NAME}] FFmpeg vágás sikertelen ({trigger_time:.1f}s): "
              f"{result.stderr.decode(errors='replace')[-300:]}")
        return None
    return str(out_path)


def main():
    print(f"[{SERVICE_NAME}] Indul ({STREAM_MODE} mód), feliratkozás "
          f"nfl.raw.audio + nfl.metadata topicokra... (videó: {VIDEO_PATH})")
    wait_for_kafka()
    consumer = KafkaConsumer(
        TOPIC_RAW_AUDIO, TOPIC_METADATA,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id="highlight-service",
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )
    producer = make_producer()

    rms_history = deque(maxlen=BASELINE_WINDOW)
    score_history = deque(maxlen=40)  # (timestamp, away_score, home_score)
    state = {"last_trigger_time": -1e9, "latest_t_sec": 0.0}
    pending_cuts = []  # [{"trigger_time_sec", "confirmed", "ready_at_t_sec"}]

    def try_cut(trigger_time_sec: float, confirmed: bool):
        clip_path = cut_clip(trigger_time_sec, confirmed)
        if clip_path is None:
            return
        event = {
            "trigger_time_sec": round(trigger_time_sec, 1),
            "confirmed": confirmed,
            "clip_path": clip_path,
        }
        producer.send(TOPIC_HIGHLIGHTS, event)
        status = "MEGERŐSÍTVE" if confirmed else "csak audió"
        print(f"[{SERVICE_NAME}] KLIP KÉSZ [{trigger_time_sec:.1f}s] -> {status} -> {clip_path}")

    def flush_ready_pending_cuts():
        still_pending = []
        for pc in pending_cuts:
            if state["latest_t_sec"] >= pc["ready_at_t_sec"]:
                try_cut(pc["trigger_time_sec"], pc["confirmed"])
            else:
                still_pending.append(pc)
        pending_cuts[:] = still_pending

    def handle(msg):
        data = msg.value
        state["latest_t_sec"] = max(state["latest_t_sec"], data.get("timestamp_sec", 0.0))

        if msg.topic == TOPIC_METADATA:
            if data.get("away_score") is not None:
                score_history.append((data["timestamp_sec"], data["away_score"], data["home_score"]))
            if STREAM_MODE == "live":
                flush_ready_pending_cuts()
            return

        # nfl.raw.audio
        pcm_bytes = base64.b64decode(data["pcm_base64"])
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        if len(samples) == 0:
            return
        rms = float(np.sqrt(np.mean(samples ** 2)))
        t_sec = data["timestamp_sec"]

        if len(rms_history) == rms_history.maxlen:
            baseline = float(np.mean(rms_history))
            ratio = rms / baseline if baseline > 0 else 0
            is_new_enough = (t_sec - state["last_trigger_time"]) > COOLDOWN_SEC
            if rms > MIN_ABS_RMS and ratio > TRIGGER_RATIO and is_new_enough:
                state["last_trigger_time"] = t_sec

                before = [s for s in score_history if s[0] <= t_sec - 5]
                after = [s for s in score_history if s[0] >= t_sec + 5]
                confirmed = bool(before and after and (before[-1][1], before[-1][2]) != (after[0][1], after[0][2]))

                print(f"[{SERVICE_NAME}] TRIGGER ÉSZLELVE [{t_sec:.1f}s] ratio={ratio:.2f}x")

                if STREAM_MODE == "live":
                    # Az utó-rész még nincs felvéve - várólistára tesszük, amíg elkészül.
                    pending_cuts.append({
                        "trigger_time_sec": t_sec,
                        "confirmed": confirmed,
                        "ready_at_t_sec": t_sec + POST_ROLL_SEC + POST_ROLL_SAFETY_BUFFER_SEC,
                    })
                else:
                    try_cut(t_sec, confirmed)

        rms_history.append(rms)
        if STREAM_MODE == "live":
            flush_ready_pending_cuts()

    consume_forever(consumer, handle, service_name=SERVICE_NAME)


if __name__ == "__main__":
    main()
