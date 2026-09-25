# NFL Cognitive Engine

Valós idejű NFL elemző rendszer: nflverse statisztikai adatokból tanult Win
Probability modell, videó-alapú OCR scoreboard-olvasás, YOLO-alapú
játékosdetekció és csapat-/oldal-besorolás, automatikus highlight-vágás, és
mindezt összekötő Kafka + Docker mikroszolgáltatás-architektúra élő
Streamlit dashboarddal.

A projekt az eredeti [fejlesztési terv](fejlesztési%20terv.txt) alapján, 5
fázisban épült fel. Minden fázis önmagában is futtatható script formájában
(`src/`), az 5. Fázis pedig ezeket köti össze éles, Kafka-alapú
mikroszolgáltatásokká (`services/`).

## Architektúra

```
[ingestion] --raw.frames--> [ocr_service] --metadata--> [fusion_service] --winprob--> [dashboard]
     |                                                                                      ^
     +----raw.frames--------> [detection_service] --detections------------------------------|
     |                                                                                      |
     +----raw.audio---------> [highlight_service] (+ metadata) --highlights----> highlights/*.mp4
```

| Fázis | Modul | Leírás |
|---|---|---|
| 1 | `src/train_win_prob.py`, `src/app.py` | nflverse play-by-play adatokon tanított XGBoost Win Probability modell + Streamlit UI |
| 2 | `src/ocr_scoreboard.py` | EasyOCR-alapú scoreboard-olvasás (negyed, óra, down&distance, eredmény) |
| 3 | `src/detect_players.py`, `src/homography.py` | YOLOv8 játékosdetekció, csapat-/oldal- (támadás/védekezés) besorolás hue-alapú színillesztéssel, pálya-homográfia yardline-becsléshez |
| 4 | `src/highlight_generator.py` | Audió RMS-trigger + OCR-alapú eredményváltozás-megerősítés + automatikus FFmpeg-vágás |
| 5 | `services/*.py`, `docker-compose.yml` | Az összes fenti modul Kafka-alapú mikroszolgáltatásként, Dockerben |

## Előfeltételek

- Python 3.11+ (helyi script-futtatáshoz)
- Docker + Docker Compose (a teljes pipeline-hoz)
- FFmpeg (helyi script-futtatáshoz; a Docker image-be be van építve)

## Gyors indulás - helyi scriptek

```bash
python -m venv venv
./venv/Scripts/pip install -r requirements.txt

# 1. Fázis: adatletöltés + modell tréning
python src/download_data.py 2019 2020 2021 2022 2023 2024
python src/train_win_prob.py
streamlit run src/app.py

# 2. Fázis: OCR egy videófájlon
python src/ocr_scoreboard.py data/video/<klip>.mp4

# 3. Fázis: játékosdetekció egy képkockán
python src/detect_players.py data/reference/calibration_frame.png <AWAY> <HOME> <POSTEAM>
# Homográfia kalibrálásához:
streamlit run src/calibrate_homography.py

# 4. Fázis: highlight-generátor
python src/highlight_generator.py data/video/<klip>.mp4
```

## Gyors indulás - teljes Kafka pipeline (Docker)

```bash
mkdir -p data/video
cp <a saját tesztklipedet> data/video/test_clip_av.mp4
docker compose up -d
```

A dashboard: **http://localhost:8501**

Minden szolgáltatás `restart: unless-stopped` policy-val fut, az `ingestion`
pedig `LOOP=true` esetén (alapértelmezett) a klip végén automatikusan
újraindítja a streamelést, így a demo nem áll le.

Minden Python szolgáltatás **egyetlen közös `nflapp-app` image-et** használ
(lásd a `docker-compose.yml` `x-app` horgonyát) - ez elkerüli, hogy 6 külön,
egyenként ~4GB-os image épüljön (torch/easyocr/ultralytics duplikálva).

### Élő stream mód

```bash
STREAM_MODE=live docker compose up -d
```

Ilyenkor az `ingestion` egy valódi élő YouTube HLS-adatfolyamot (`LIVE_URL`,
alapból egy 24/7 forgó archív-meccs csatorna) dolgoz fel folyamatosan,
azonnal publikálva a Kafka topicokra - az OCR/Win Probability így valós
időben megy. A highlightokhoz szükséges "utó-részt" (POST_ROLL) fizikailag
nem lehet azonnal kivágni élőben (még nem történt meg) - a `highlight_service`
ezt egy késleltetett várólistával kezeli.

**Erőforrás-korlátok**: a torch/OpenCV alapból minden elérhető magot
lefoglalna egyetlen híváskor is, ami 8 magos gépen ~500%+ CPU-t okozott. A
`docker-compose.yml` ezért szál-limiteket (`OMP_NUM_THREADS` stb.),
kemény per-konténer CPU/memória limiteket, 480p élő felbontást és ritkított
OCR-hívást (`OCR_STRIDE`) állít be - nyugalmi állapotban ~10-15% CPU, rövid
OCR-tüskékkel.

## Tesztelés

```bash
pip install -r requirements-test.txt
pytest tests/ -v
```

A `tests/` a tiszta üzleti logikát fedi le (OCR-token parszolás, homográfia
matematika, feature-vektor építés, Kafka hibatűrés, csapatszín-illesztés) -
**nem** igényel EasyOCR/torch/ultralytics telepítést, ezért gyors (~4s).

A `.github/workflows/ci.yml` minden push/PR-nél lefuttatja:
1. **unit-tests** - a fenti pytest suite
2. **docker-integration** - megépíti a teljes Docker image-et, elindítja a
   valódi Kafka pipeline-t egy kis (`tests/fixtures/sample_clip.mp4`, ~10 mp)
   videóval, és ellenőrzi, hogy a dashboard válaszol, az OCR/fusion/detection
   szolgáltatások valós adatot termelnek, és egyetlen konténer sem állt le
   váratlanul.

## Ismert korlátozások

Ezek tudatos MVP-egyszerűsítések, nem elfelejtett hibák:

- **Az OCR ROI és a homográfia kalibráció adásfüggő.** Egy adott kameraállásra/
  grafikai stílusra (pl. CBS felső sáv vs. NBC alsó sáv) van kalibrálva -
  másik meccs/csatorna esetén újra kell kalibrálni (`src/calibrate_homography.py`).
- **A homográfia egyetlen állóképre kalibrált**, nem követi a kamera
  panorámázását/zoomolását - a yardline-becslés ezért csak közelítő (±3-5 yard).
- **A csapat-/oldal-besorolás két, előre megadott csapatra van paraméterezve**
  (`AWAY_TEAM`/`HOME_TEAM`/`POSTEAM` env változók), nem ismeri fel dinamikusan,
  melyik két csapat játszik.
- **A highlight vizuális megerősítése** eredményváltozás-detekción alapul
  (nem bírói kézmozdulat-felismerésen, ahogy az eredeti terv javasolta) -
  ehhez külön címkézett tréningadat és pózosztályozó modell kellene.
- **Nincs formáció-osztályozás** (Cover 1/Cover 3/Blitz) és **AI Scouting
  Assistant (RAG)** - ezek az eredeti terv további, jelentős önálló
  munkát igénylő kiterjesztései, nem részei a jelenlegi megvalósításnak.

## Projekt struktúra

```
src/                    Önállóan futtatható script-ek (1-4. Fázis)
services/               Kafka mikroszolgáltatások (5. Fázis)
tests/                  Unit tesztek + CI videó-fixture
models/                 Betanított Win Probability modell + metaadatok
data/team_colors.csv    nflverse hivatalos csapatszín-adatbázis
.github/workflows/      CI pipeline
Dockerfile              Közös image minden Python szolgáltatáshoz
docker-compose.yml      Kafka + mikroszolgáltatások + dashboard
```
