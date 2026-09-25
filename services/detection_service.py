"""Modul 3: YOLO detekciós mikroszolgáltatás - nfl.raw.frames -> nfl.detections.

CPU-n a YOLO+színklasszifikáció frame-enként ~100-300ms, ezért minden N.
frame-et dolgozunk fel (DETECTION_STRIDE), hogy a Kafka topic ne torlódjon fel.
"""
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

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from detect_players import detect_and_classify
from homography import FieldHomography, HOMOGRAPHY_PATH

from kafka_utils import TOPIC_DETECTIONS, TOPIC_RAW_FRAMES, consume_forever, make_consumer, make_producer

SERVICE_NAME = "detection_service"
DETECTION_STRIDE = int(os.environ.get("DETECTION_STRIDE", "3"))
AWAY_TEAM = os.environ.get("AWAY_TEAM", "CLE")
HOME_TEAM = os.environ.get("HOME_TEAM", "NE")
POSTEAM = os.environ.get("POSTEAM", "CLE")


def main():
    print(f"[{SERVICE_NAME}] YOLO modell betöltése...")
    model = YOLO("yolov8n.pt")
    homography = FieldHomography.load(HOMOGRAPHY_PATH)
    if homography is None:
        print(f"[{SERVICE_NAME}] Figyelem: nincs kalibrált homográfia, field_x/field_y nélkül fut.")

    consumer = make_consumer(TOPIC_RAW_FRAMES, group_id="detection-service")
    producer = make_producer()
    print(f"[{SERVICE_NAME}] Kész, várakozás a nfl.raw.frames topicra...")

    counter = {"i": 0}

    def handle(msg):
        counter["i"] += 1
        if counter["i"] % DETECTION_STRIDE != 0:
            return

        data = msg.value
        jpg_bytes = base64.b64decode(data["jpg_base64"])
        frame = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[{SERVICE_NAME}] Érvénytelen JPEG a frame_id={data.get('frame_id')} üzenetben, kihagyva.")
            return

        detections = detect_and_classify(frame, AWAY_TEAM, HOME_TEAM, POSTEAM, model, homography)
        offense_n = sum(1 for d in detections if d.side == "offense")
        defense_n = sum(1 for d in detections if d.side == "defense")

        producer.send(TOPIC_DETECTIONS, {
            "frame_id": data["frame_id"],
            "timestamp_sec": data["timestamp_sec"],
            "num_players": len(detections),
            "offense_count": offense_n,
            "defense_count": defense_n,
            "players": [asdict(d) for d in detections],
        })
        print(f"[{SERVICE_NAME}] [{data['timestamp_sec']:6.1f}s] {len(detections)} játékos "
              f"(off={offense_n}, def={defense_n})")

    consume_forever(consumer, handle, service_name=SERVICE_NAME)


if __name__ == "__main__":
    main()
