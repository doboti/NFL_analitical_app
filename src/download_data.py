"""Letölti az nflverse play-by-play, sérülés és menetrend (games) adatait a data/ mappába."""
import sys
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet"
INJURIES_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{year}.parquet"
GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet"

DEFAULT_YEARS = range(2016, 2025)


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"{dest.name}: már megvan, kihagyva ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    print(f"{dest.name}: letöltés {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    tmp = dest.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    print(f"{dest.name}: kész ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def download_year(year: int) -> Path:
    """Egy szezon play-by-play adata."""
    return _download(PBP_URL.format(year=year), DATA_DIR / f"play_by_play_{year}.parquet")


def download_injuries_year(year: int) -> Path:
    """Egy szezon heti sérülés-jelentése (src/pregame_feature_engineering.py-hoz)."""
    return _download(INJURIES_URL.format(year=year), DATA_DIR / f"injuries_{year}.parquet")


def download_games() -> Path:
    """A teljes menetrend-tábla (minden szezon egy fájlban) - pihenőnap és
    időjárás mezőket tartalmaz (src/pregame_feature_engineering.py-hoz)."""
    return _download(GAMES_URL, DATA_DIR / "games.parquet")


def main(years=None, include_injuries: bool = True, include_games: bool = True) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    years = years or DEFAULT_YEARS

    for year in years:
        try:
            download_year(year)
        except requests.HTTPError as exc:
            print(f"{year}: PBP letöltése sikertelen ({exc})", file=sys.stderr)

        if include_injuries:
            try:
                download_injuries_year(year)
            except requests.HTTPError as exc:
                print(f"{year}: injuries letöltése sikertelen ({exc})", file=sys.stderr)

    if include_games:
        try:
            download_games()
        except requests.HTTPError as exc:
            print(f"games.parquet letöltése sikertelen ({exc})", file=sys.stderr)


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else None
    main(args)
