"""Unit tesztek az OCR token-parszoló logikára (nem igényel EasyOCR-t/torch-ot)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ocr_scoreboard import (  # noqa: E402
    REFERENCE_RESOLUTION,
    SCOREBOARD_ROI,
    SCOREBOARD_ROI_BOTTOM,
    SCOREBOARD_ROI_CANDIDATES,
    SCOREBOARD_ROI_TOP,
    ScoreboardState,
    _state_score,
    crop_and_upscale,
    parse_ocr_tokens,
)


def test_crop_and_upscale_scales_roi_to_actual_frame_size():
    """Regresszió-teszt: amikor a frame nem a kalibrációs (1280x720) felbontású
    (pl. egy alacsonyabb felbontású élő stream miatt), a ROI-nak arányosan kell
    skálázódnia, különben a kivágás üres lesz / kicsúszik a képből (lásd az
    élő módban ténylegesen előfordult OpenCV 'ssize.empty()' hibát)."""
    ref_w, ref_h = REFERENCE_RESOLUTION
    full_res_frame = np.zeros((ref_h, ref_w, 3), dtype=np.uint8)
    crop_full = crop_and_upscale(full_res_frame)
    assert crop_full.size > 0

    half_res_frame = np.zeros((ref_h // 2, ref_w // 2, 3), dtype=np.uint8)
    crop_half = crop_and_upscale(half_res_frame)
    assert crop_half.size > 0
    # a kivágásnak kb. feleakkorának kell lennie (upscale előtt), mint a teljes felbontásún
    x1, y1, x2, y2 = SCOREBOARD_ROI
    assert crop_half.shape[1] < crop_full.shape[1]
    assert crop_half.shape[0] < crop_full.shape[0]


def test_crop_and_upscale_empty_when_frame_too_short_for_roi():
    # Az arányosan lefelé skálázott ROI ennél a magasságnál (5px) 0 magasságú
    # kivágást ad - ezt kezelnie kell hiba nélkül, üres tömböt visszaadva.
    tiny_frame = np.zeros((5, 1280, 3), dtype=np.uint8)
    crop = crop_and_upscale(tiny_frame)
    assert crop.size == 0


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


def test_single_recognized_team_assigned_to_correct_side_by_score_position():
    """Regressziós teszt: élesben előfordult, hogy csak EGY csapatkódot
    (a jobb oldali, home csapatot) ismerte fel az OCR, és a régi logika ezt
    tévesen mindig 'away'-ként rögzítette, a home_team None maradt - a
    dashboard scoreboard fejléce ekkor rossz csapatot/szín-t mutatott.
    Most a felismert csapatnak a HOZZÁ KÖZELEBBI eredmény-pozíció alapján
    kell a helyes oldalra (home) kerülnie."""
    tokens = [
        (300.0, "1&10", 0.8),
        (450.0, "35", 1.0),   # bal oldali (away) eredmény - NYG csapatkódja nem ismerődött fel
        (900.0, "DAL", 0.95),  # jobb oldali (home) csapatkód, sikeresen felismerve
        (950.0, "44", 1.0),   # jobb oldali (home) eredmény, közel a DAL-hoz
    ]
    state = parse_ocr_tokens(tokens)
    assert state.home_team == "DAL"
    assert state.away_team is None
    assert state.away_score == 35
    assert state.home_score == 44


def test_single_recognized_team_assigned_to_away_when_closer_to_left_score():
    tokens = [
        (100.0, "NYG", 0.95),  # bal oldali (away) csapatkód, felismerve
        (150.0, "35", 1.0),    # bal oldali (away) eredmény, közel az NYG-hez
        (600.0, "44", 1.0),    # jobb oldali (home) eredmény - DAL kódja nem ismerődött fel
    ]
    state = parse_ocr_tokens(tokens)
    assert state.away_team == "NYG"
    assert state.home_team is None
    assert state.away_score == 35
    assert state.home_score == 44


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


def test_scoreboard_roi_candidates_cover_top_and_bottom_of_frame():
    """Regressziós teszt: élesben előfordult egy CBS-stílusú felvétel, ahol a
    csapatok/pontok/idő sáv a kép TETEJÉN volt, nem alul (mint az addig
    kalibrált NBC-stílusú klipnél) - az OCR-nek emiatt fel kell tudnia
    dolgozni mindkét elrendezést, ROI-jelöltek listájával."""
    ref_w, ref_h = REFERENCE_RESOLUTION
    assert SCOREBOARD_ROI_BOTTOM in SCOREBOARD_ROI_CANDIDATES
    assert SCOREBOARD_ROI_TOP in SCOREBOARD_ROI_CANDIDATES

    _, top_y1, _, top_y2 = SCOREBOARD_ROI_TOP
    _, bottom_y1, _, bottom_y2 = SCOREBOARD_ROI_BOTTOM
    assert top_y2 < ref_h / 2  # a felső jelölt a kép felső felében van
    assert bottom_y1 > ref_h / 2  # az alsó jelölt a kép alsó felében van
    assert SCOREBOARD_ROI == SCOREBOARD_ROI_BOTTOM  # visszafelé kompatibilis alapérték


def test_state_score_counts_recognized_fields():
    empty_state = ScoreboardState(timestamp_sec=0.0)
    assert _state_score(empty_state) == 0

    full_state = ScoreboardState(
        timestamp_sec=0.0, quarter="1ST", clock="10:20", down=2, distance=7,
        away_team="NE", away_score=0, home_team="DEN", home_score=3,
    )
    assert _state_score(full_state) == 8

    partial_state = ScoreboardState(timestamp_sec=0.0, away_team="NE", home_team="DEN")
    assert _state_score(partial_state) == 2
    assert _state_score(partial_state) > _state_score(empty_state)
