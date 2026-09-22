"""Unit tesztek az OCR-állapot -> modell-feature-vektor átalakításra."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fusion_demo import ocr_state_to_features  # noqa: E402

FEATURES_ORDER = [
    "qtr", "down", "ydstogo", "yardline_100", "score_differential",
    "half_seconds_remaining", "game_seconds_remaining",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining", "is_home",
]

COMPLETE_STATE = {
    "quarter": "2ND",
    "clock": "10:15",
    "down": 3,
    "distance": 7,
    "away_team": "CLE",
    "away_score": 10,
    "home_team": "NE",
    "home_score": 7,
}


def test_complete_state_produces_feature_row():
    features = ocr_state_to_features(COMPLETE_STATE, FEATURES_ORDER)
    assert features is not None
    assert list(features.columns) == FEATURES_ORDER
    row = features.iloc[0]
    assert row["qtr"] == 2
    assert row["down"] == 3
    assert row["ydstogo"] == 7
    # POSTEAM_ASSUMPTION == "away" -> a labdabirtokos az away csapat (CLE)
    assert row["score_differential"] == 10 - 7
    assert row["is_home"] == 0


def test_missing_required_field_returns_none():
    for missing_key in COMPLETE_STATE:
        incomplete = dict(COMPLETE_STATE)
        incomplete[missing_key] = None
        assert ocr_state_to_features(incomplete, FEATURES_ORDER) is None


def test_unparseable_quarter_returns_none():
    state = dict(COMPLETE_STATE)
    state["quarter"] = "GARBAGE"
    assert ocr_state_to_features(state, FEATURES_ORDER) is None


def test_clock_parsed_into_seconds_remaining():
    state = dict(COMPLETE_STATE)
    state["quarter"] = "1ST"
    state["clock"] = "5:00"
    features = ocr_state_to_features(state, FEATURES_ORDER)
    row = features.iloc[0]
    # 1. negyedben: hátralévő idő a félidőből = negyedből hátralévő + még egy teljes negyed (15*60)
    assert row["half_seconds_remaining"] == 5 * 60 + 15 * 60
    # a teljes meccsből hátralévő idő: negyedből hátralévő + 3 hátralévő negyed * 15*60
    assert row["game_seconds_remaining"] == 5 * 60 + 3 * 15 * 60
