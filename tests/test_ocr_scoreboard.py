"""Unit tesztek az OCR token-parszoló logikára (nem igényel EasyOCR-t/torch-ot)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ocr_scoreboard import parse_ocr_tokens  # noqa: E402


def test_cbs_style_single_token_scoreboard():
    """CBS-stílus: down&distance külön tokenben a negyed+óra elől, csapatnevek középen."""
    tokens = [
        (280.0, "2ND & 7", 0.9),
        (470.0, "CLE", 0.99),
        (560.0, "0", 1.0),
        (650.0, "NE", 1.0),
        (740.0, "0", 0.9),
        (830.0, "1ST 12:00 08", 0.8),
    ]
    state = parse_ocr_tokens(tokens)
    assert state.down == 2
    assert state.distance == 7
    assert state.quarter == "1ST"
    assert state.clock == "12:00"
    assert state.away_team == "CLE"
    assert state.away_score == 0
    assert state.home_team == "NE"
    assert state.home_score == 0


def test_quarter_and_clock_glued_without_space():
    """NBC-stílus: az EasyOCR néha szóköz nélkül fűzi a negyedet és az órát ('2ND14:34')."""
    tokens = [
        (300.0, "1ST & 10", 0.85),
        (450.0, "PHI", 0.95),
        (520.0, "0", 1.0),
        (600.0, "NYG", 0.9),
        (670.0, "0", 1.0),
        (800.0, "2ND14:34", 0.7),
    ]
    state = parse_ocr_tokens(tokens)
    assert state.quarter == "2ND"
    assert state.clock == "14:34"
    assert state.down == 1
    assert state.distance == 10


def test_clock_with_period_instead_of_colon():
    """Az EasyOCR néha pontot ír kettőspont helyett ('14.36')."""
    tokens = [(500.0, "2ND 14.36", 0.7)]
    state = parse_ocr_tokens(tokens)
    assert state.clock == "14:36"
    assert state.quarter == "2ND"


def test_ordinal_ocr_noise_still_matches_via_fuzzy():
    """'IST' (I helyett 1) és 'ZND' (Z helyett 2) tipikus OCR-hibák - fuzzy matchelve."""
    tokens = [(500.0, "IST 9:00", 0.6)]
    state = parse_ocr_tokens(tokens)
    assert state.quarter == "1ST"

    tokens2 = [(200.0, "ZNd & 7", 0.6)]
    state2 = parse_ocr_tokens(tokens2)
    assert state2.down == 2
    assert state2.distance == 7


def test_missing_fields_stay_none():
    tokens = [(100.0, "NFL", 0.9), (200.0, "CBS", 0.9)]
    state = parse_ocr_tokens(tokens)
    assert state.down is None
    assert state.quarter is None
    assert state.away_team is None
    assert state.away_score is None


def test_empty_tokens():
    state = parse_ocr_tokens([])
    assert state.raw_text == ""
    assert state.down is None
