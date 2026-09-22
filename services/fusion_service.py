"""Modul 5: Valós idejű fúzió - nfl.metadata -> nfl.winprob (XGBoost Win Probability)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fusion_demo import load_model, ocr_state_to_features

from kafka_utils import TOPIC_METADATA, TOPIC_WINPROB, consume_forever, make_consumer, make_producer

SERVICE_NAME = "fusion_service"


def main():
    print(f"[{SERVICE_NAME}] Modell betöltése...")
    model, meta = load_model()
    consumer = make_consumer(TOPIC_METADATA, group_id="fusion-service")
    producer = make_producer()
    print(f"[{SERVICE_NAME}] Kész, várakozás a nfl.metadata topicra...")

    def handle(msg):
        state = msg.value
        features = ocr_state_to_features(state, meta["features"])
        if features is None:
            return

        posteam_wp = float(model.predict_proba(features)[0, 1])
        result = {
            "timestamp_sec": state["timestamp_sec"],
            "situation": f"Q{state['quarter']} {state['clock']} | {state['down']}&{state['distance']} | "
                         f"{state['away_team']} {state['away_score']} - {state['home_team']} {state['home_score']}",
            "posteam_win_prob": round(posteam_wp, 4),
            "defteam_win_prob": round(1 - posteam_wp, 4),
        }
        producer.send(TOPIC_WINPROB, result)
        print(f"[{SERVICE_NAME}] [{result['timestamp_sec']:6.1f}s] {result['situation']} "
              f"-> win prob: {posteam_wp:.1%}")

    consume_forever(consumer, handle, service_name=SERVICE_NAME)


if __name__ == "__main__":
    main()
