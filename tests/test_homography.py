"""Unit tesztek a pálya-homográfia pixel<->yard transzformációjára."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from homography import FieldHomography  # noqa: E402


def test_from_points_round_trip_on_a_simple_square():
    # Egy egyszerű, torzítatlan négyzet-leképezés: pixel (0-100,0-100) -> yard (0-10,0-10).
    pixel_points = [(0, 0), (100, 0), (100, 100), (0, 100)]
    field_points = [(0, 0), (10, 0), (10, 10), (0, 10)]
    h = FieldHomography.from_points(pixel_points, field_points)

    x, y = h.pixel_to_field(50, 50)
    assert x == pytest.approx(5.0, abs=0.01)
    assert y == pytest.approx(5.0, abs=0.01)

    x2, y2 = h.pixel_to_field(0, 0)
    assert x2 == pytest.approx(0.0, abs=0.01)
    assert y2 == pytest.approx(0.0, abs=0.01)


def test_from_points_requires_at_least_four_points():
    with pytest.raises(ValueError):
        FieldHomography.from_points([(0, 0), (1, 1)], [(0, 0), (1, 1)])


def test_from_points_requires_matching_lengths():
    with pytest.raises(ValueError):
        FieldHomography.from_points([(0, 0), (1, 0), (1, 1), (0, 1)], [(0, 0), (1, 0), (1, 1)])


def test_pixel_to_field_without_matrix_raises():
    h = FieldHomography()
    with pytest.raises(RuntimeError):
        h.pixel_to_field(1, 1)


def test_save_and_load_round_trip(tmp_path):
    pixel_points = [(0, 0), (100, 0), (100, 100), (0, 100)]
    field_points = [(0, 0), (10, 0), (10, 10), (0, 10)]
    h = FieldHomography.from_points(pixel_points, field_points)

    save_path = tmp_path / "homography.json"
    h.save(save_path)
    assert save_path.exists()

    loaded = FieldHomography.load(save_path)
    x, y = loaded.pixel_to_field(50, 50)
    assert x == pytest.approx(5.0, abs=0.01)
    assert y == pytest.approx(5.0, abs=0.01)


def test_load_missing_file_returns_none(tmp_path):
    assert FieldHomography.load(tmp_path / "does_not_exist.json") is None
