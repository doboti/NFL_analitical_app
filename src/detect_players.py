"""
Modul 3: Objektumdetekció és csapat-/oldal-besorolás (pre-snap formáció alapja).

YOLOv8n (COCO pretrained, 'person' osztály) detektálja a játékosokat, majd:
  1. Egy egyszerű y-koordináta alapú "field mask" kiszűri az oldalvonalon
     kívüli személyeket (nézők, edzők) - lásd FIELD_Y_MIN.
  2. A mezszín (jersey) mintavételezéséből, az nflverse hivatalos
     csapatszín-adatbázisával (data/team_colors.csv) összevetve minden
     játékost hozzárendel a két csapat egyikéhez.
  3. A labdabirtokos csapat (OCR-ből vagy paraméterből) alapján
     támadó/védekező címkét ad minden játékosnak.
  4. Ha van kalibrált homográfia (models/homography.json), minden
     játékos lábponttját pálya-koordinátává alakítja.

KORLÁTOZÁS: a YOLO 'person' detekció nem különbözteti meg a bírót a
játékosoktól kizárólag alak alapján - a bíró jellemzően a csíkos,
fekete-fehér mezéről kiszűrhető lenne (nem illeszkedik egyik csapatszínre
sem), ezért alacsony színegyezési biztonságnál a játékos "unknown"
címkét kap ahelyett, hogy erőltetnénk a csapatba sorolást.
"""
import json
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from homography import FieldHomography, HOMOGRAPHY_PATH

ROOT = Path(__file__).resolve().parent.parent
TEAM_COLORS_CSV = ROOT / "data" / "team_colors.csv"

FIELD_Y_MIN = 195  # a calibration_frame.png-hez (1280x720) igazítva: e fölött oldalvonali személyzet
HUE_MATCH_MAX_DIST = 20  # OpenCV hue fok (0-179 skála) - efölött "unknown" a csapatcímke
MIN_SATURATION = 60  # ez alatt (fű, árnyék, fehér mez) a pixel nem számít csapatszín-jelöltnek
GRASS_HUE_RANGE = (30, 95)  # kizárva a domináns szín kereséséből (pálya zöldje)
MIN_TEAM_PIXELS = 5  # ennél kevesebb egyező pixelnél "unknown" a besorolás


@dataclass
class PlayerDetection:
    bbox: list
    confidence: float
    team: Optional[str] = None
    side: Optional[str] = None  # "offense" | "defense" | None
    field_x: Optional[float] = None
    field_y: Optional[float] = None


def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


@lru_cache(maxsize=1)
def _load_team_csv() -> pd.DataFrame:
    return pd.read_csv(TEAM_COLORS_CSV)


def load_team_swatches(team_abbrs: list[str]) -> dict:
    df = _load_team_csv()
    swatches = {}
    for abbr in team_abbrs:
        row = df[df.team_abbr == abbr]
        if row.empty:
            continue
        colors = [hex_to_rgb(row.iloc[0][c]) for c in ["team_color", "team_color2"] if pd.notna(row.iloc[0][c])]
        swatches[abbr] = colors
    return swatches


def dominant_team_hue(frame: np.ndarray, bbox: list) -> Optional[float]:
    """A teljes bounding box pixeleiből kiszűri a pálya zöldjét és a szürke/
    fehér (árnyék, fehér mez) pixeleket, majd a maradék (jellemzően sisak +
    mez-szegély) pixelek medián hue-ját adja vissza. Ez sokkal robusztusabb a
    testtartásra (guggoló vs. álló játékos), mint egy fix pozíciójú "sisak-
    régió" kivágás, mert nem tesz feltételezést arról, HOL van a csapatszín
    a dobozon belül - csak azt keresi, hogy VAN-e elég belőle."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    patch = frame[max(y1, 0):max(y2, 1), max(x1, 0):max(x2, 1)]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    h, s = hsv[:, 0], hsv[:, 1]
    grass_lo, grass_hi = GRASS_HUE_RANGE
    mask = (s > MIN_SATURATION) & ~((h > grass_lo) & (h < grass_hi))
    if mask.sum() < MIN_TEAM_PIXELS:
        return None
    return float(np.median(h[mask]))


def _rgb_to_hue(rgb: tuple) -> float:
    arr = np.uint8([[[int(rgb[0]), int(rgb[1]), int(rgb[2])]]])
    return float(cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[0, 0, 0])


def _hue_dist(h1: float, h2: float) -> float:
    diff = abs(h1 - h2)
    return min(diff, 180 - diff)


def classify_team(hue: Optional[float], swatches: dict) -> Optional[str]:
    if hue is None:
        return None
    best_team, best_dist = None, float("inf")
    for team, colors in swatches.items():
        for color in colors:
            dist = _hue_dist(hue, _rgb_to_hue(color))
            if dist < best_dist:
                best_dist = dist
                best_team = team
    if best_dist > HUE_MATCH_MAX_DIST:
        return None
    return best_team


def detect_and_classify(
    frame: np.ndarray,
    away_team: str,
    home_team: str,
    posteam: str,
    model: YOLO,
    homography: Optional[FieldHomography] = None,
) -> list[PlayerDetection]:
    swatches = load_team_swatches([away_team, home_team])
    results = model.predict(source=frame, classes=[0], conf=0.25, verbose=False)[0]

    detections = []
    for box in results.boxes:
        bbox = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        _, y1, _, y2 = bbox
        if (y1 + y2) / 2 < FIELD_Y_MIN:
            continue  # oldalvonali személyzet, nem játékos a pályán

        hue = dominant_team_hue(frame, bbox)
        team = classify_team(hue, swatches)
        side = None
        if team is not None:
            side = "offense" if team == posteam else "defense"

        field_x = field_y = None
        if homography is not None:
            foot_x = (bbox[0] + bbox[2]) / 2
            foot_y = bbox[3]
            field_x, field_y = homography.pixel_to_field(foot_x, foot_y)

        detections.append(PlayerDetection(bbox=bbox, confidence=conf, team=team, side=side,
                                           field_x=field_x, field_y=field_y))
    return detections


def draw_annotated(frame: np.ndarray, detections: list[PlayerDetection]) -> np.ndarray:
    colors = {"offense": (0, 0, 255), "defense": (255, 100, 0), None: (128, 128, 128)}
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d.bbox]
        color = colors.get(d.side, (128, 128, 128))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.team or '?'} {d.side or ''}"
        if d.field_x is not None:
            label += f" {d.field_x:.0f}yd"
        cv2.putText(out, label, (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


def main():
    frame_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "reference" / "calibration_frame.png"
    away_team = sys.argv[2] if len(sys.argv) > 2 else "CLE"
    home_team = sys.argv[3] if len(sys.argv) > 3 else "NE"
    posteam = sys.argv[4] if len(sys.argv) > 4 else "CLE"

    frame = cv2.imread(str(frame_path))
    model = YOLO("yolov8n.pt")
    homography = FieldHomography.load(HOMOGRAPHY_PATH)
    if homography is None:
        print(f"[figyelem] Nincs kalibrált homográfia ({HOMOGRAPHY_PATH} hiányzik) - "
              f"field_x/field_y nélkül futunk. Kalibráláshoz: streamlit run src/calibrate_homography.py")

    detections = detect_and_classify(frame, away_team, home_team, posteam, model, homography)

    offense_n = sum(1 for d in detections if d.side == "offense")
    defense_n = sum(1 for d in detections if d.side == "defense")
    unknown_n = sum(1 for d in detections if d.side is None)
    print(f"Detektált játékosok: {len(detections)} (támadás: {offense_n}, védekezés: {defense_n}, "
          f"besorolatlan: {unknown_n})")
    for d in detections:
        pos = f"field=({d.field_x:.1f}, {d.field_y:.1f})" if d.field_x is not None else ""
        print(f"  {d.team or '?':4s} {d.side or '?':8s} conf={d.confidence:.2f} {pos}")

    annotated = draw_annotated(frame, detections)
    out_path = ROOT / "data" / "detections_annotated.png"
    cv2.imwrite(str(out_path), annotated)
    print(f"\nAnnotált kép mentve: {out_path}")

    json_out = ROOT / "data" / "detections.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump([asdict(d) for d in detections], f, indent=2)
    print(f"JSON mentve: {json_out}")


if __name__ == "__main__":
    main()
