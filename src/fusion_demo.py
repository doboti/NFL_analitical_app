"""
Bonus: OCR -> Win Probability fúziós demó.

Az src/ocr_scoreboard.py kimenetét (data/ocr_output.jsonl) összeköti a
src/train_win_prob.py által betanított XGBoost modellel, hogy valós
videóból kiolvasott játékhelyzetre élőben számoljon győzelmi esélyt.

FONTOS KORLÁTOZÁSOK (a scoreboard OCR nem ad meg mindent, amit a modell vár):
  - yardline_100: nincs a scoreboardon -> YARDLINE_DEFAULT placeholder (50 = középpálya)
  - posteam_timeouts_remaining / defteam_timeouts_remaining: nincs a scoreboardon
    ebben az adásban -> TIMEOUTS_DEFAULT placeholder (3 = teljes timeout keret)
  - labdabirtokos csapat: a scoreboard nem jelöl explicit birtoklás-indikátort ebben
    a ROI-ban, ezért heurisztikaként az AWAY csapatot (a down&distance doboz melletti,
    balra eső csapat) tekintjük labdabirtokosnak (POSTEAM_ASSUMPTION).

Ezért ez a demó azt mutatja meg, hogy MŰKÖDIK a láncolat (OCR -> feature vektor ->
modell -> valószínűség), nem pedig végleges, pontos élő win probability motort.
A pontos yardline-hoz és a labdabirtoklás egyértelmű detektálásához a 3. Fázis
(YOLO + pálya-homográfia) szükséges.
"""
import json
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
OCR_INPUT = ROOT / "data" / "ocr_output.jsonl"
MODEL_PATH = ROOT / "models" / "win_probability_xgb.json"
META_PATH = ROOT / "models" / "win_probability_meta.json"
FUSION_OUTPUT = ROOT / "data" / "fusion_output.jsonl"

YARDLINE_DEFAULT = 50
TIMEOUTS_DEFAULT = 3
POSTEAM_ASSUMPTION = "away"  # a down&distance doboz melletti (bal oldali) csapat


def load_model():
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta


def ocr_state_to_features(state: dict, features_order: list[str]):
    required = ["quarter", "down", "distance", "away_team", "away_score", "home_team", "home_score", "clock"]
    if any(state.get(k) is None for k in required):
        return None

    qtr_map = {"1ST": 1, "2ND": 2, "3RD": 3, "4TH": 4, "OT": 5}
    qtr = qtr_map.get(state["quarter"])
    if qtr is None:
        return None

    minutes, seconds = (int(p) for p in state["clock"].split(":"))
    quarter_seconds_remaining = minutes * 60 + seconds
    quarters_left_after_current = max(0, 4 - qtr)
    half_seconds_remaining = quarter_seconds_remaining + (60 * 15 if qtr in (1, 3) else 0)
    game_seconds_remaining = quarter_seconds_remaining + quarters_left_after_current * 15 * 60

    is_home = POSTEAM_ASSUMPTION == "home"
    posteam_score = state["home_score"] if is_home else state["away_score"]
    defteam_score = state["away_score"] if is_home else state["home_score"]

    row = {
        "qtr": qtr,
        "down": state["down"],
        "ydstogo": state["distance"],
        "yardline_100": YARDLINE_DEFAULT,
        "score_differential": posteam_score - defteam_score,
        "half_seconds_remaining": half_seconds_remaining,
        "game_seconds_remaining": game_seconds_remaining,
        "posteam_timeouts_remaining": TIMEOUTS_DEFAULT,
        "defteam_timeouts_remaining": TIMEOUTS_DEFAULT,
        "is_home": int(is_home),
    }
    return pd.DataFrame([row])[features_order]


def main():
    if not OCR_INPUT.exists():
        print(f"Nincs OCR kimenet: {OCR_INPUT}. Futtasd előbb: python src/ocr_scoreboard.py")
        return

    model, meta = load_model()
    features_order = meta["features"]

    skipped = 0
    written = 0
    with open(OCR_INPUT, encoding="utf-8") as f_in, open(FUSION_OUTPUT, "w", encoding="utf-8") as f_out:
        for line in f_in:
            state = json.loads(line)
            features = ocr_state_to_features(state, features_order)
            if features is None:
                skipped += 1
                continue

            posteam_wp = float(model.predict_proba(features)[0, 1])
            posteam = state["away_team"] if POSTEAM_ASSUMPTION == "away" else state["home_team"]
            defteam = state["home_team"] if POSTEAM_ASSUMPTION == "away" else state["away_team"]

            result = {
                "timestamp_sec": state["timestamp_sec"],
                "situation": f"Q{state['quarter']} {state['clock']} | {state['down']}&{state['distance']} | "
                             f"{state['away_team']} {state['away_score']} - {state['home_team']} {state['home_score']}",
                "posteam_assumption": posteam,
                "posteam_win_prob": round(posteam_wp, 4),
                "defteam_win_prob": round(1 - posteam_wp, 4),
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1
            print(
                f"[{result['timestamp_sec']:6.1f}s] {result['situation']:45s} "
                f"-> {posteam} győzelmi esély: {posteam_wp:.1%} (feltételezett labdabirtokos: {posteam})"
            )

    print(f"\nFeldolgozva: {written} sor | Kihagyva (hiányos OCR adat): {skipped} sor")
    print(f"Kimenet: {FUSION_OUTPUT}")


if __name__ == "__main__":
    main()
