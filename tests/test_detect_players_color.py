"""Unit tesztek a csapatszín (hue-alapú) klasszifikációra - nem igényel YOLO-t/torch-ot."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from detect_players import (  # noqa: E402
    classify_team, dominant_team_hue, hex_to_rgb,
)

CLE_SWATCHES = {"CLE": [(255, 60, 0), (49, 29, 0)]}
CLE_NE_SWATCHES = {"CLE": [(255, 60, 0), (49, 29, 0)], "NE": [(0, 34, 68), (198, 12, 48)]}


def test_hex_to_rgb():
    assert hex_to_rgb("#FF3C00") == (255, 60, 0)
    assert hex_to_rgb("002244") == (0, 34, 68)


def test_classify_team_matches_close_hue():
    # Egy kissé elsötétített narancs (adás-kompresszió hatása) még mindig CLE-nek illeszkedjen.
    assert classify_team(hue=10.0, swatches=CLE_NE_SWATCHES) == "CLE"


def test_classify_team_distinguishes_two_teams():
    cle_hue = 7.0   # a CLE narancs (#FF3C00) OpenCV hue-ja
    ne_hue = 105.0  # a NE navy (#002244) OpenCV hue-ja
    assert classify_team(cle_hue, CLE_NE_SWATCHES) == "CLE"
    assert classify_team(ne_hue, CLE_NE_SWATCHES) == "NE"


def test_classify_team_returns_none_when_far_from_all_swatches():
    far_hue = 90.0  # zöld-ish, egyik csapatszíntől is távol
    assert classify_team(far_hue, CLE_SWATCHES) is None


def test_classify_team_none_hue_returns_none():
    assert classify_team(None, CLE_NE_SWATCHES) is None


def test_dominant_team_hue_on_solid_orange_patch():
    # BGR sorrendben egy tiszta narancs (#FF3C00 -> RGB(255,60,0) -> BGR(0,60,255))
    patch = np.full((20, 20, 3), (0, 60, 255), dtype=np.uint8)
    hue = dominant_team_hue(patch, [0, 0, 20, 20])
    assert hue is not None
    assert hue == pytest.approx(7.0, abs=3.0)


def test_dominant_team_hue_on_grass_only_returns_none():
    # Tiszta fűzöld (BGR-ben kb. (60, 140, 60)) - ez a GRASS_HUE_RANGE-be esik, kizárva.
    patch = np.full((20, 20, 3), (60, 140, 60), dtype=np.uint8)
    hue = dominant_team_hue(patch, [0, 0, 20, 20])
    assert hue is None


def test_dominant_team_hue_empty_bbox_returns_none():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    hue = dominant_team_hue(frame, [10, 10, 10, 10])
    assert hue is None
