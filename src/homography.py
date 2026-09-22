"""
Pálya-homográfia: képpont (pixel) -> pálya-koordináta (yard) transzformáció.

Koordináta-rendszer: x = 0..120 (a saját endzone gólvonalától a másik endzone
gólvonaláig, 0-10 és 110-120 az endzone-ok), y = 0..53.33 (oldalvonaltól
oldalvonalig, yardban). Ez megegyezik a Big Data Bowl konvencióval.

A transzformációt cv2.findHomography számítja ki >= 4 (pixel, yard) pont-
párból. A kalibrációs pontokat a src/calibrate_homography.py interaktív
Streamlit eszközzel lehet felvenni és models/homography.json-ba menteni.

FONTOS KORLÁTOZÁS: egy adott kalibráció csak addig érvényes, amíg a kamera
nem mozdul/zoomol. Élő adásban a kamera folyamatosan pásztáz -> éles
rendszerben kockánkénti (vagy legalább snap-enkénti) újrakalibrálás vagy
automatikus pályavonal-detekció kellene. Ez az MVP egyetlen rögzített
kameraállásra (a data/reference/calibration_frame.png-re) kalibrál.
"""
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HOMOGRAPHY_PATH = ROOT / "models" / "homography.json"


class FieldHomography:
    def __init__(self, matrix: Optional[np.ndarray] = None):
        self.matrix = matrix

    @classmethod
    def from_points(cls, pixel_points: list[tuple[float, float]], field_points: list[tuple[float, float]]):
        if len(pixel_points) < 4 or len(pixel_points) != len(field_points):
            raise ValueError("Legalább 4, egyező számú pixel- és pálya-pont kell.")
        src = np.array(pixel_points, dtype=np.float32)
        dst = np.array(field_points, dtype=np.float32)
        matrix, _ = cv2.findHomography(src, dst, method=0)
        if matrix is None:
            raise ValueError("A homográfia számítása sikertelen (kollineáris pontok?).")
        return cls(matrix)

    def pixel_to_field(self, x: float, y: float) -> tuple[float, float]:
        if self.matrix is None:
            raise RuntimeError("Nincs betöltött/kiszámított homográfia mátrix.")
        pt = np.array([[[x, y]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.matrix)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def save(self, path: Path = HOMOGRAPHY_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"matrix": self.matrix.tolist()}, f, indent=2)

    @classmethod
    def load(cls, path: Path = HOMOGRAPHY_PATH) -> Optional["FieldHomography"]:
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return cls(np.array(data["matrix"], dtype=np.float64))


def is_calibrated(path: Path = HOMOGRAPHY_PATH) -> bool:
    return path.exists()
