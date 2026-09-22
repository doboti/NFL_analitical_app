"""
Modul 4: Multimodális Highlight Generator (Audió + Vizuális fúzió).

1. Audio Trigger: az RMS hangenergia-görbét egy gördülő (causal) alapvonalhoz
   viszonyítjuk. Ha a pillanatnyi RMS a közelmúltbeli átlaghoz (baseline)
   képest hirtelen jelentősen megugrik -> potenciális highlight esemény.
   (Abszolút küszöb helyett RELATÍV, mert a háttérzaj szintje adásonként/
   stadiononként eltérő - egy fix dB-küszöb nem generalizálna.)
2. Vizuális megerősítés: az OCR scoreboard-olvasóval megnézzük, változott-e
   az eredmény az esemény előtti/utáni pillanatban. Ha igen -> "confirmed"
   (nagy valószínűséggel valódi scoring play). Ha nem -> "audio_only" (pl.
   nagy játék de nem pontszerzés, vagy hamis riasztás), de a klipet ettől
   függetlenül elkészítjük - a megerősítés csak egy megbízhatósági jelző.
3. Automatikus vágás: FFmpeg-gel esemény-15s .. esemény+20s (35 mp) klip
   a highlights/ mappába.

KORLÁTOZÁS: a doksi eredeti terve bírói kézmozdulat-felismerést is javasol
vizuális megerősítésként (pl. touchdown jelzés) - ez egy külön pózosztályozó
modellt igényelne, amihez nincs címkézett tréningadat. Az OCR-alapú
eredményváltozás-ellenőrzés egy pragmatikus, azonnal működő helyettesítő.
"""
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FFMPEG = ROOT / "tools" / "ffmpeg.exe"
HIGHLIGHTS_DIR = ROOT / "data" / "highlights"

HOP_SEC = 0.5
BASELINE_WINDOW_SEC = 10.0
TRIGGER_RATIO = 1.3       # a pillanatnyi RMS ennyiszerese a baseline-nak -> trigger
MIN_ABS_RMS = 0.05        # ez alatt (csend/szünet) nem számít, még ha relatíve ugrik is
COOLDOWN_SEC = 20.0       # egy trigger után ennyi ideig nem indul új (ne duplikáljunk egy eseményt)
PRE_ROLL_SEC = 15.0
POST_ROLL_SEC = 20.0


@dataclass
class HighlightEvent:
    trigger_time_sec: float
    rms: float
    baseline: float
    ratio: float
    confirmed: bool
    clip_path: Optional[str] = None


def compute_rms_envelope(video_path: Path):
    wav_path = video_path.with_suffix(".wav")
    subprocess.run(
        [str(FFMPEG), "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
         "-ar", "22050", "-ac", "1", str(wav_path)],
        capture_output=True, check=True,
    )
    y, sr = librosa.load(str(wav_path), sr=None)
    hop_length = int(sr * HOP_SEC)
    rms = librosa.feature.rms(y=y, frame_length=hop_length * 2, hop_length=hop_length)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
    return times, rms


def detect_triggers(times, rms) -> list[HighlightEvent]:
    baseline_window = max(1, int(BASELINE_WINDOW_SEC / HOP_SEC))
    events = []
    last_trigger_time = -1e9

    for i in range(baseline_window, len(rms)):
        baseline = float(np.mean(rms[i - baseline_window:i]))
        current = float(rms[i])
        if baseline <= 0:
            continue
        ratio = current / baseline
        t = float(times[i])

        if current > MIN_ABS_RMS and ratio > TRIGGER_RATIO and (t - last_trigger_time) > COOLDOWN_SEC:
            events.append(HighlightEvent(trigger_time_sec=round(t, 1), rms=round(current, 4),
                                          baseline=round(baseline, 4), ratio=round(ratio, 2),
                                          confirmed=False))
            last_trigger_time = t
    return events


def confirm_with_ocr(video_path: Path, event: HighlightEvent) -> bool:
    """Megnézi, változott-e az eredmény az esemény előtti/utáni pillanatban."""
    sys.path.insert(0, str(ROOT / "src"))
    from ocr_scoreboard import ScoreboardReader
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    reader = ScoreboardReader()

    def read_at(t_sec):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t_sec * fps)))
        ret, frame = cap.read()
        if not ret:
            return None
        return reader.read_frame(frame, t_sec)

    before = read_at(max(0, event.trigger_time_sec - 8))
    after = read_at(event.trigger_time_sec + 8)
    cap.release()

    if before is None or after is None:
        return False
    if before.away_score is None or after.away_score is None:
        return False
    return (before.away_score, before.home_score) != (after.away_score, after.home_score)


def cut_clip(video_path: Path, event: HighlightEvent) -> Path:
    HIGHLIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    start = max(0.0, event.trigger_time_sec - PRE_ROLL_SEC)
    duration = PRE_ROLL_SEC + POST_ROLL_SEC
    tag = "confirmed" if event.confirmed else "audio_only"
    out_path = HIGHLIGHTS_DIR / f"highlight_{event.trigger_time_sec:.0f}s_{tag}.mp4"

    subprocess.run(
        [str(FFMPEG), "-y", "-ss", str(start), "-i", str(video_path), "-t", str(duration),
         "-c", "copy", str(out_path)],
        capture_output=True, check=True,
    )
    return out_path


def process_video(video_path: Path, run_ocr_confirmation: bool = True) -> list[HighlightEvent]:
    print(f"Hangenergia (RMS) elemzése: {video_path}")
    times, rms = compute_rms_envelope(video_path)

    events = detect_triggers(times, rms)
    print(f"Detektált audió-triggerek: {len(events)}")

    for event in events:
        if run_ocr_confirmation:
            event.confirmed = confirm_with_ocr(video_path, event)
        clip_path = cut_clip(video_path, event)
        event.clip_path = str(clip_path)
        status = "MEGERŐSÍTVE (eredmény változott)" if event.confirmed else "csak audió (eredmény nem változott)"
        print(f"  [{event.trigger_time_sec:5.1f}s] ratio={event.ratio}x baseline -> {status}")
        print(f"           klip: {clip_path}")

    return events


if __name__ == "__main__":
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "video" / "test_clip_av.mp4"
    process_video(video)
