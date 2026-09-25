"""
Modul 2: OCR és Valós Idejű Játék-Metaadat Extrakció.

Beolvassa a videó scoreboard sávját (ROI), OCR-ezi EasyOCR-rel, majd
strukturált JSON-t állít elő: negyed, óra, down & distance, eredmény.

Megjegyzés: a ROI koordináták a data/video/test_clip.mp4 (CBS grafika)
kalibrációjához igazodnak. Más adásstílushoz (FOX/NBC/ESPN) a ROI-t
újra kell kalibrálni - lásd calibrate_roi().
"""
import difflib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ROI-k a 1280x720-as forráshoz kalibrálva (bal, fent, jobb, lent).
# A scoreboard sáv adásonként MÁS-MÁS HELYEN lehet a képen (pl. NBC/ESPN
# jellemzően ALUL, CBS jellemzően FELÜL rajzolja ki) - ezért nem egyetlen
# fix ROI-t nézünk, hanem több JELÖLT ROI-t (lásd SCOREBOARD_ROI_CANDIDATES
# lent), és minden frame-nél azt fogadjuk el, amelyiken a legtöbb mezőt
# sikerül felismerni. Ha egy teljesen új adásstílus ROI-ja egyikkel sem
# egyezik, ide kell felvenni egy újat.
REFERENCE_RESOLUTION = (1280, 720)  # a lenti ROI-k ehhez a felbontáshoz vannak kalibrálva
SCOREBOARD_ROI_BOTTOM = (180, 635, 1080, 675)  # alsó sáv (pl. NBC-stílus)
SCOREBOARD_ROI_TOP = (280, 20, 1000, 100)  # felső sáv (pl. CBS-stílus)
SCOREBOARD_ROI_CANDIDATES = [SCOREBOARD_ROI_BOTTOM, SCOREBOARD_ROI_TOP]
SCOREBOARD_ROI = SCOREBOARD_ROI_BOTTOM  # visszafelé kompatibilis alapérték (pl. tesztekhez)
UPSCALE_FACTOR = 3
SAMPLE_INTERVAL_SEC = 1.0

VALID_ORDINALS = ["1ST", "2ND", "3RD", "4TH", "OT"]
NFL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
    "TEN", "WAS",
]


@dataclass
class ScoreboardState:
    timestamp_sec: float
    quarter: Optional[str] = None
    clock: Optional[str] = None
    down: Optional[int] = None
    distance: Optional[int] = None
    away_team: Optional[str] = None
    away_score: Optional[int] = None
    home_team: Optional[str] = None
    home_score: Optional[int] = None
    raw_text: str = ""


def _closest_match(token: str, choices: list[str], cutoff: float = 0.5) -> Optional[str]:
    matches = difflib.get_close_matches(token.upper(), choices, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def _ordinal_to_number(ordinal: str) -> Optional[int]:
    return {"1ST": 1, "2ND": 2, "3RD": 3, "4TH": 4}.get(ordinal)


def crop_and_upscale(frame: np.ndarray, roi=SCOREBOARD_ROI, scale=UPSCALE_FACTOR) -> np.ndarray:
    """A roi a REFERENCE_RESOLUTION-höz van kalibrálva - ha a frame ettől eltérő
    felbontású (pl. az élő stream alacsonyabb felbontású formátumot ad, CPU-
    kímélés miatt), a ROI-t arányosan átskálázzuk a frame tényleges méretéhez,
    hogy ne csússzon ki a képből / ne váljon üressé a kivágás."""
    frame_h, frame_w = frame.shape[:2]
    ref_w, ref_h = REFERENCE_RESOLUTION
    sx, sy = frame_w / ref_w, frame_h / ref_h

    x1, y1, x2, y2 = roi
    x1, x2 = int(x1 * sx), int(x2 * sx)
    y1, y2 = int(y1 * sy), int(y2 * sy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    return cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)


def parse_ocr_tokens(tokens: list[tuple]) -> ScoreboardState:
    """tokens: list of (x_left, text, confidence), balról jobbra rendezve."""
    tokens = sorted(tokens, key=lambda t: t[0])
    raw_text = " ".join(t[1] for t in tokens)
    state = ScoreboardState(timestamp_sec=0.0, raw_text=raw_text)

    team_hits = []
    score_hits = []

    for x, text, conf in tokens:
        clean = re.sub(r"[^A-Z0-9&:.]", "", text.upper())
        if not clean:
            continue

        # Down & distance: pl. "2ND&7" vagy "3RD&10" egy tokenben.
        m_down = re.search(r"([A-Z0-9]{1,3})(ST|ND|RD|TH).{0,3}&\s*(\d{1,2})", clean)
        if m_down:
            ordinal_candidate = _closest_match(
                m_down.group(1) + m_down.group(2), VALID_ORDINALS[:4], cutoff=0.3
            )
            if ordinal_candidate:
                state.down = _ordinal_to_number(ordinal_candidate)
            state.distance = int(m_down.group(3))
            continue

        # Csak distance (ha a down ordinal külön tokenbe esett).
        if "&" in clean:
            m_dist = re.search(r"&\s*(\d{1,2})", clean)
            if m_dist:
                state.distance = int(m_dist.group(1))
            continue

        # Óra (mm:ss) - az EasyOCR néha pontot ír kettőspont helyett ("14.36").
        clock_m = re.search(r"(\d{1,2})[:.](\d{2})", clean)
        if clock_m and state.clock is None:
            state.clock = f"{clock_m.group(1)}:{clock_m.group(2)}"

        # Negyedjelző: a token ELEJÉN lévő ordinal mintázat (pl. "1ST", "IST" -> "1ST"),
        # nem a teljes (esetleg órával egybeírt, pl. "2ND14:34") tokenre illesztve -
        # EasyOCR adásfüggően szóköz nélkül fűzi össze a negyedet és az órát.
        ordinal_only = re.sub(r"[^A-Z0-9]", "", re.split(r"[:.]", clean)[0])
        m_q = re.match(r"^([A-Z0-9]{1,2})(ST|ND|RD|TH)", ordinal_only)
        if m_q:
            candidate = _closest_match(m_q.group(1) + m_q.group(2), VALID_ORDINALS, cutoff=0.4)
            if candidate:
                state.quarter = candidate
            continue

        # Csapatkód (2-4 betű, szigorú egyezés a lefoglalt tokenek kizárása után).
        letters_only = re.sub(r"[^A-Z]", "", clean)
        if 2 <= len(letters_only) <= 4:
            candidate = _closest_match(letters_only, NFL_TEAMS, cutoff=0.75)
            if candidate:
                team_hits.append((x, candidate))
                continue

        # Eredmény (önálló 1-2 jegyű szám).
        if re.fullmatch(r"\d{1,2}", clean):
            score_hits.append((x, int(clean)))

    team_hits.sort(key=lambda t: t[0])
    score_hits.sort(key=lambda t: t[0])

    if len(score_hits) >= 1:
        state.away_score = score_hits[0][1]
    if len(score_hits) >= 2:
        state.home_score = score_hits[1][1]

    if len(team_hits) >= 2:
        state.away_team = team_hits[0][1]
        state.home_team = team_hits[1][1]
    elif len(team_hits) == 1 and len(score_hits) == 2:
        # Csak EGY csapatkódot ismertünk fel (a másik OCR-hibából kimaradt) -
        # NEM szabad automatikusan "away"-nek venni, mert lehet, hogy épp a
        # jobb oldali (home) csapatot sikerült felismerni. Ehelyett a hozzá
        # X-koordinátában LEGKÖZELEBBI eredmény-számhoz igazítjuk: ha a bal
        # oldali (away) eredményhez van közelebb, akkor away, különben home.
        team_x = team_hits[0][0]
        dist_to_away_score = abs(team_x - score_hits[0][0])
        dist_to_home_score = abs(team_x - score_hits[1][0])
        if dist_to_away_score <= dist_to_home_score:
            state.away_team = team_hits[0][1]
        else:
            state.home_team = team_hits[0][1]
    elif len(team_hits) == 1:
        # Nincs 2 eredmény-jelzés a pozíció-alapú döntéshez - jobb híján az
        # away alapértelmezésre esünk vissza (a korábbi, kevésbé pontos
        # viselkedés), mert ekkor sincs jobb infónk a döntéshez.
        state.away_team = team_hits[0][1]

    return state


def _state_score(state: ScoreboardState) -> int:
    """Hány mezőt sikerült felismerni - ez alapján választjuk ki, melyik ROI-
    jelölt (felső/alsó sáv) illeszkedik az aktuális adásra."""
    fields = (state.quarter, state.clock, state.down, state.distance,
              state.away_team, state.away_score, state.home_team, state.home_score)
    return sum(1 for f in fields if f is not None)


class ScoreboardReader:
    def __init__(self, roi=None, scale=UPSCALE_FACTOR, roi_candidates=None):
        import easyocr  # lusta import: a tiszta parszoló logika (pl. tesztekhez)
        # ne igényelje az easyocr+torch telepítését, csak ha ténylegesen OCR-ezünk.

        # Ha a hívó KIFEJEZETTEN megad egy roi-t, csak azt próbáljuk (pl. korábbi
        # kalibrálás után, teljesítmény miatt) - egyébként minden jelöltet
        # (felső + alsó sáv) végignézünk, és a legjobb találatot tartjuk meg.
        self.roi_candidates = [roi] if roi is not None else (roi_candidates or SCOREBOARD_ROI_CANDIDATES)
        self.scale = scale
        self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def read_frame(self, frame: np.ndarray, timestamp_sec: float) -> ScoreboardState:
        best_state = None
        best_score = -1
        for roi in self.roi_candidates:
            crop = crop_and_upscale(frame, roi, self.scale)
            if crop.size == 0:
                continue
            results = self.reader.readtext(crop)
            tokens = [(bbox[0][0], text, conf) for bbox, text, conf in results]
            state = parse_ocr_tokens(tokens)
            score = _state_score(state)
            if score > best_score:
                best_state, best_score = state, score

        if best_state is None:
            best_state = ScoreboardState(raw_text="")
        best_state.timestamp_sec = round(timestamp_sec, 2)
        return best_state


def process_video(video_path: Path, out_path: Path, sample_interval=SAMPLE_INTERVAL_SEC):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, round(fps * sample_interval))
    reader = ScoreboardReader()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    with open(out_path, "w", encoding="utf-8") as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                timestamp_sec = frame_idx / fps
                state = reader.read_frame(frame, timestamp_sec)
                line = json.dumps(asdict(state), ensure_ascii=False)
                f.write(line + "\n")
                f.flush()
                print(
                    f"[{state.timestamp_sec:6.1f}s] Q{state.quarter} {state.clock} | "
                    f"{state.down}&{state.distance} | "
                    f"{state.away_team} {state.away_score} - {state.home_team} {state.home_score} "
                    f"| raw='{state.raw_text}'"
                )
            frame_idx += 1
    cap.release()
    print(f"\nKész. Kimenet: {out_path}")


if __name__ == "__main__":
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "video" / "test_clip.mp4"
    output = ROOT / "data" / "ocr_output.jsonl"
    process_video(video, output)
