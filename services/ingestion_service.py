"""
Modul 1: Ingestion & Streaming Layer.

Beolvassa a helyi videó+hang fájlt (data/video/test_clip_av.mp4), és úgy
"streameli" a Kafka topicokba, mintha élő adás lenne: 0.5 másodpercenként
egy videókockát a nfl.raw.frames-be, a hozzá tartozó hangszeletet pedig a
nfl.raw.audio-ba.

PLAYBACK_SPEED env változó szabályozza a sebességet (pl. 4.0 = 4x gyorsabb
mint valós idő - hasznos teszteléshez; 1.0 = valós idejű szimuláció).
LOOP=true esetén a klip véget érve újraindul a streamelés a kliphez tartozó
Kafka topicokon (valós élő adásnál ez nem kellene, de egy véges tesztklipnél
így marad "élő" a demo, a docker-compose restart policy-ja helyett/mellett).
"""
import base64
import os
import subprocess
import time
from pathlib import Path

import cv2
import librosa
import numpy as np

from kafka_utils import TOPIC_RAW_AUDIO, TOPIC_RAW_FRAMES, make_producer

SERVICE_NAME = "ingestion"
VIDEO_PATH = Path(os.environ.get("VIDEO_PATH", "/app/data/video/test_clip_av.mp4"))
STEP_SEC = 0.5
PLAYBACK_SPEED = float(os.environ.get("PLAYBACK_SPEED", "4.0"))
LOOP = os.environ.get("LOOP", "true").lower() == "true"


def load_audio(video_path: Path):
    wav_path = video_path.with_suffix(".ingest.wav")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
         "-ar", "22050", "-ac", "1", str(wav_path), "-loglevel", "error"],
        capture_output=True,
    )
    if result.returncode != 0 or not wav_path.exists():
        raise RuntimeError(
            f"Hangkinyerés sikertelen ({video_path}): {result.stderr.decode(errors='replace')}"
        )
    y, sr = librosa.load(str(wav_path), sr=None)
    return y, sr


def stream_once(producer, audio_y, audio_sr, fps, frame_step) -> int:
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"A videó nem nyitható meg: {VIDEO_PATH}")

    frame_idx = 0
    chunk_idx = 0
    wall_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            t_sec = frame_idx / fps

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                producer.send(TOPIC_RAW_FRAMES, {
                    "frame_id": chunk_idx,
                    "timestamp_sec": round(t_sec, 2),
                    "jpg_base64": base64.b64encode(buf.tobytes()).decode("ascii"),
                })

            a_start = int(t_sec * audio_sr)
            a_end = int((t_sec + STEP_SEC) * audio_sr)
            chunk = audio_y[a_start:a_end]
            if len(chunk) > 0:
                pcm_bytes = (chunk * 32767).astype(np.int16).tobytes()
                producer.send(TOPIC_RAW_AUDIO, {
                    "chunk_id": chunk_idx,
                    "timestamp_sec": round(t_sec, 2),
                    "sample_rate": audio_sr,
                    "pcm_base64": base64.b64encode(pcm_bytes).decode("ascii"),
                })

            chunk_idx += 1

            target_wall = wall_start + t_sec / PLAYBACK_SPEED
            sleep_for = target_wall - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)

        frame_idx += 1

    producer.flush()
    cap.release()
    return chunk_idx


def main():
    print(f"[{SERVICE_NAME}] Videó: {VIDEO_PATH} | playback speed: {PLAYBACK_SPEED}x | loop: {LOOP}")
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"A videófájl nem található: {VIDEO_PATH}")

    producer = make_producer()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    frame_step = max(1, round(fps * STEP_SEC))

    audio_y, audio_sr = load_audio(VIDEO_PATH)

    while True:
        try:
            sent = stream_once(producer, audio_y, audio_sr, fps, frame_step)
            print(f"[{SERVICE_NAME}] Kör kész. {sent} frame/audio chunk elküldve.")
        except Exception as exc:
            print(f"[{SERVICE_NAME}] Hiba streamelés közben, 5s múlva próbálja újra: {exc}")
            time.sleep(5)

        if not LOOP:
            break
        print(f"[{SERVICE_NAME}] LOOP=true, klip újraindítása...")


if __name__ == "__main__":
    main()
