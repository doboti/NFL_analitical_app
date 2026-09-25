"""Modul 2: OCR mikroszolgáltatás - nfl.raw.frames -> nfl.metadata."""
import base64
import os
import sys
from dataclasses import asdict
from pathlib import Path

import cv2

cv2.setNumThreads(int(os.environ.get("OPENCV_NUM_THREADS", "2")))

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ocr_scoreboard import ScoreboardReader

from kafka_utils import TOPIC_METADATA, TOPIC_RAW_FRAMES, consume_forever, make_consumer, make_producer

SERVICE_NAME = "ocr_service"
# A scoreboard (óra, eredmény, down&distance) nem változik sub-szekundum
# gyakorisággal, ezért nem kell minden bejövő frame-et (0.5s-enként) OCR-ezni -
# ez volt a legnagyobb CPU-fogyasztó a rendszerben (EasyOCR ~170%+ CPU
# másodpercenkénti 2 híváskor). Minden N. frame-et dolgozunk fel.
OCR_STRIDE = int(os.environ.get("OCR_STRIDE", "2"))


def main():
    print(f"[{SERVICE_NAME}] Indul, EasyOCR betöltése...")
    reader = ScoreboardReader()
    consumer = make_consumer(TOPIC_RAW_FRAMES, group_id="ocr-service")
    producer = make_producer()
    print(f"[{SERVICE_NAME}] Kész, várakozás a nfl.raw.frames topicra...")

    counter = {"i": 0}

    def handle(msg):
        counter["i"] += 1
        if counter["i"] % OCR_STRIDE != 0:
            return

        data = msg.value
        jpg_bytes = base64.b64decode(data["jpg_base64"])
        frame = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[{SERVICE_NAME}] Érvénytelen JPEG a frame_id={data.get('frame_id')} üzenetben, kihagyva.")
            return

        state = reader.read_frame(frame, data["timestamp_sec"])
        payload = asdict(state)
        payload["frame_id"] = data["frame_id"]
        producer.send(TOPIC_METADATA, payload)
        print(f"[{SERVICE_NAME}] [{state.timestamp_sec:6.1f}s] Q{state.quarter} {state.clock} "
              f"{state.down}&{state.distance} {state.away_team} {state.away_score}-"
              f"{state.home_score} {state.home_team}")

    consume_forever(consumer, handle, service_name=SERVICE_NAME)


if __name__ == "__main__":
    main()
