"""
Modul 1: Ingestion & Streaming Layer.

Két üzemmódot támogat (STREAM_MODE env változó):

  "file" (alapértelmezett): egy helyi videó+hang fájlt (VIDEO_PATH) "streamel"
      a Kafka topicokba, mintha élő adás lenne. LOOP=true esetén a klip
      végén újraindul, hogy a demo ne álljon le.

  "live": egy VALÓDI élő YouTube HLS-adatfolyamot (LIVE_URL) dolgoz fel
      folyamatosan, megszakítás/hiba esetén automatikus újracsatlakozással.
      A frame-eket és hangot AZONNAL, késleltetés nélkül publikálja a Kafka
      topicokra (az OCR/Win Probability így valós időben megy). Emellett a
      teljes folyamot egy helyi fájlba is rögzíti (LIVE_RECORDING_PATH),
      hogy a highlight_service ebből tudjon klipet vágni - lásd a
      highlight_service.py tetején lévő megjegyzést arról, hogy miért nem
      lehet a highlight "utó-részét" azonnal kivágni élő adásnál.

PLAYBACK_SPEED csak "file" módban számít (pl. 4.0 = 4x gyorsabb mint valós
idő - hasznos teszteléshez; "live" módban a forrás saját tempója számít).
"""
import base64
import os
import subprocess
import time
from pathlib import Path

import cv2

cv2.setNumThreads(int(os.environ.get("OPENCV_NUM_THREADS", "2")))

import librosa
import numpy as np

from kafka_utils import TOPIC_RAW_AUDIO, TOPIC_RAW_FRAMES, make_producer

SERVICE_NAME = "ingestion"
STREAM_MODE = os.environ.get("STREAM_MODE", "file").lower()
VIDEO_PATH = Path(os.environ.get("VIDEO_PATH", "/app/data/video/test_clip_av.mp4"))
STEP_SEC = 0.5
PLAYBACK_SPEED = float(os.environ.get("PLAYBACK_SPEED", "4.0"))
LOOP = os.environ.get("LOOP", "true").lower() == "true"

LIVE_URL = os.environ.get("LIVE_URL", "https://youtu.be/0iZSW6aAhNA")
LIVE_RECORDING_PATH = Path(os.environ.get("LIVE_RECORDING_PATH", "/app/data/video/live_recording.mp4"))
# 480p elég az OCR-hez (a scoreboard sáv olvasásához) és a detekcióhoz is -
# 720p helyett ezt választva a folyamatos frame-dekódolás CPU-igénye jelentősen
# csökken (kevesebb pixel/frame), miközben a highlight-klipek is így készülnek
# (ugyanezt a forrást rögzítjük) - még mindig jól nézhető minőség.
LIVE_MAX_VIDEO_HEIGHT = int(os.environ.get("LIVE_MAX_VIDEO_HEIGHT", "480"))
AUDIO_SR = 22050


# --------------------------------------------------------------------------
# "file" mód: helyi fájl visszajátszása (a korábbi, egyszerűbb megvalósítás)
# --------------------------------------------------------------------------

def load_audio(video_path: Path):
    wav_path = video_path.with_suffix(".ingest.wav")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
         "-ar", str(AUDIO_SR), "-ac", "1", str(wav_path), "-loglevel", "error"],
        capture_output=True,
    )
    if result.returncode != 0 or not wav_path.exists():
        raise RuntimeError(
            f"Hangkinyerés sikertelen ({video_path}): {result.stderr.decode(errors='replace')}"
        )
    y, sr = librosa.load(str(wav_path), sr=None)
    return y, sr


def stream_file_once(producer, audio_y, audio_sr, fps, frame_step) -> int:
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"A videó nem nyitható meg: {VIDEO_PATH}")

    frame_idx = 0
    chunk_idx = 0
    wall_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            t_sec = frame_idx / fps
            _publish_frame(producer, chunk_idx, t_sec, frame)

            a_start = int(t_sec * audio_sr)
            a_end = int((t_sec + STEP_SEC) * audio_sr)
            chunk = audio_y[a_start:a_end]
            if len(chunk) > 0:
                _publish_audio_from_float(producer, chunk_idx, t_sec, chunk, audio_sr)

            chunk_idx += 1

            target_wall = wall_start + t_sec / PLAYBACK_SPEED
            sleep_for = target_wall - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)

        frame_idx += 1

    producer.flush()
    cap.release()
    return chunk_idx


def run_file_mode(producer):
    print(f"[{SERVICE_NAME}] Videó: {VIDEO_PATH} | playback speed: {PLAYBACK_SPEED}x | loop: {LOOP}")
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"A videófájl nem található: {VIDEO_PATH}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    frame_step = max(1, round(fps * STEP_SEC))

    audio_y, audio_sr = load_audio(VIDEO_PATH)

    while True:
        try:
            sent = stream_file_once(producer, audio_y, audio_sr, fps, frame_step)
            print(f"[{SERVICE_NAME}] Kör kész. {sent} frame/audio chunk elküldve.")
        except Exception as exc:
            print(f"[{SERVICE_NAME}] Hiba streamelés közben, 5s múlva próbálja újra: {exc}")
            time.sleep(5)

        if not LOOP:
            break
        print(f"[{SERVICE_NAME}] LOOP=true, klip újraindítása...")


# --------------------------------------------------------------------------
# "live" mód: valódi élő HLS-adatfolyam folyamatos feldolgozása
# --------------------------------------------------------------------------

def resolve_live_urls(youtube_url: str) -> tuple[str, str]:
    """Feloldja a jelenleg élő adásfolyam közvetlen videó- és hang-URL-jét.

    Nem hardcode-olt itag-számokra támaszkodik (azok adásonként/csatornánként
    változhatnak), hanem a legjobb <=720p, csak-videó és a legjobb csak-hang
    formátumot választja ki a yt-dlp által visszaadott formátum-listából."""
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(youtube_url, download=False)

    # Élő HLS manifestnél a yt-dlp gyakran nem tudja megállapítani a pontos
    # kodeket (acodec=None, nem "none" string) - ezért a vcodec jelenléte/
    # hiánya alapján különböztetjük meg a videó- és hang-only formátumokat,
    # nem az acodec/abr mezők (nem) meglétére támaszkodva.
    video_formats = [
        f for f in info["formats"]
        if f.get("vcodec") not in (None, "none")
        and (f.get("height") or 0) <= LIVE_MAX_VIDEO_HEIGHT
    ]
    audio_formats = [
        f for f in info["formats"]
        if f.get("vcodec") in (None, "none") and (f.get("height") or 0) == 0
    ]
    if not video_formats or not audio_formats:
        raise RuntimeError(
            f"Nem található megfelelő csak-videó/csak-hang formátum az élő adásban "
            f"({len(info['formats'])} formátum összesen)."
        )

    best_video = max(video_formats, key=lambda f: f.get("height") or 0)
    # Az abr (audio bitrate) élő manifestnél gyakran ismeretlen (None) mindkét
    # hangformátumra - ilyenkor a nagyobb format_id-jút választjuk (a yt-dlp
    # jellemzően a jobb minőségű hangformátumot listázza magasabb id-vel).
    def _format_id_num(f):
        try:
            return int(f.get("format_id") or 0)
        except ValueError:
            return 0

    best_audio = max(audio_formats, key=lambda f: (f.get("abr") or 0, _format_id_num(f)))
    return best_video["url"], best_audio["url"]


def _publish_frame(producer, frame_id: int, t_sec: float, frame) -> None:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return
    producer.send(TOPIC_RAW_FRAMES, {
        "frame_id": frame_id,
        "timestamp_sec": round(t_sec, 2),
        "jpg_base64": base64.b64encode(buf.tobytes()).decode("ascii"),
    })


def _publish_audio_from_float(producer, chunk_id: int, t_sec: float, chunk: np.ndarray, sr: int) -> None:
    pcm_bytes = (chunk * 32767).astype(np.int16).tobytes()
    producer.send(TOPIC_RAW_AUDIO, {
        "chunk_id": chunk_id,
        "timestamp_sec": round(t_sec, 2),
        "sample_rate": sr,
        "pcm_base64": base64.b64encode(pcm_bytes).decode("ascii"),
    })


def _publish_audio_from_pcm_bytes(producer, chunk_id: int, t_sec: float, pcm_bytes: bytes, sr: int) -> None:
    producer.send(TOPIC_RAW_AUDIO, {
        "chunk_id": chunk_id,
        "timestamp_sec": round(t_sec, 2),
        "sample_rate": sr,
        "pcm_base64": base64.b64encode(pcm_bytes).decode("ascii"),
    })


def start_background_recorder(video_url: str, audio_url: str) -> subprocess.Popen:
    """Folyamatosan rögzíti az élő adást egy helyi fájlba, amiből a
    highlight_service később klipet tud vágni. A frag_keyframe/empty_moov
    flag-ek nélkül egy MP4 fájl nem olvasható addig, amíg az író folyamat
    be nem zárja - ezekkel a flag-ekkel viszont ÍRÁS KÖZBEN is olvasható/
    vágható marad (streamelhető MP4)."""
    LIVE_RECORDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_path = LIVE_RECORDING_PATH.with_suffix(".recorder.log")
    log_file = open(log_path, "wb")  # a subprocess örökli, mi nem olvassuk - fájlba, nem PIPE-ba (nincs deadlock-kockázat)
    return subprocess.Popen(
        # -bsf:a aac_adtstoasc: a HLS-forrásból jövő AAC hang ADTS formátumú,
        # ezt kell átalakítani MP4-kompatibilis formátumra a "-c copy" mellett -
        # enélkül a muxer néhány frame után hibával leáll (ADTS bitstream error).
        ["ffmpeg", "-y", "-i", video_url, "-i", audio_url,
         "-c", "copy", "-bsf:a", "aac_adtstoasc",
         "-movflags", "frag_keyframe+empty_moov+default_base_moof",
         str(LIVE_RECORDING_PATH), "-loglevel", "warning"],
        stdout=subprocess.DEVNULL, stderr=log_file,
    )


def stream_live_once(producer) -> None:
    print(f"[{SERVICE_NAME}] Élő adásfolyam feloldása: {LIVE_URL}")
    video_url, audio_url = resolve_live_urls(LIVE_URL)
    print(f"[{SERVICE_NAME}] Feloldva. Rögzítés indítása: {LIVE_RECORDING_PATH}")

    recorder = start_background_recorder(video_url, audio_url)

    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        recorder.terminate()
        raise RuntimeError("Az élő videofolyam nem nyitható meg.")

    audio_proc = subprocess.Popen(
        ["ffmpeg", "-i", audio_url, "-f", "s16le", "-ar", str(AUDIO_SR), "-ac", "1",
         "-loglevel", "error", "pipe:1"],
        stdout=subprocess.PIPE,
    )
    chunk_bytes = int(STEP_SEC * AUDIO_SR) * 2  # 16 bites minták -> 2 byte/minta

    wall_start = time.time()
    frame_idx = 0
    chunk_idx = 0
    last_publish = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Az élő videofolyam megszakadt (nincs több frame).")

            now = time.time() - wall_start
            if now - last_publish >= STEP_SEC:
                last_publish = now
                _publish_frame(producer, chunk_idx, now, frame)

                pcm_bytes = audio_proc.stdout.read(chunk_bytes)
                if pcm_bytes:
                    _publish_audio_from_pcm_bytes(producer, chunk_idx, now, pcm_bytes, AUDIO_SR)

                if chunk_idx % 20 == 0:
                    print(f"[{SERVICE_NAME}] [élő, {now:6.1f}s] {chunk_idx} chunk elküldve")
                chunk_idx += 1

            frame_idx += 1
    finally:
        cap.release()
        audio_proc.terminate()
        recorder.terminate()


def run_live_mode(producer):
    print(f"[{SERVICE_NAME}] ÉLŐ MÓD - forrás: {LIVE_URL}")
    print(f"[{SERVICE_NAME}] Figyelem: ez egy 24/7 forgó demó-csatorna (archív meccsek), "
          f"NEM egy jelenleg zajló, hivatalos NFL-közvetítés.")
    while True:
        try:
            stream_live_once(producer)
        except Exception as exc:
            print(f"[{SERVICE_NAME}] Élő stream hiba, 10s múlva újracsatlakozás: {exc}")
            time.sleep(10)


def main():
    producer = make_producer()
    if STREAM_MODE == "live":
        run_live_mode(producer)
    else:
        run_file_mode(producer)


if __name__ == "__main__":
    main()
