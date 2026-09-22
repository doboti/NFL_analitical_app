"""Letölti az nflverse play-by-play parquet fájljait a data/ mappába."""
import sys
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RELEASE_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet"

DEFAULT_YEARS = range(2016, 2025)


def download_year(year: int) -> Path:
    dest = DATA_DIR / f"play_by_play_{year}.parquet"
    if dest.exists():
        print(f"{year}: már megvan, kihagyva ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    url = RELEASE_URL.format(year=year)
    print(f"{year}: letöltés {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    tmp = dest.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    print(f"{year}: kész ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def main(years=None) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    years = years or DEFAULT_YEARS
    for year in years:
        try:
            download_year(year)
        except requests.HTTPError as exc:
            print(f"{year}: nem sikerült letölteni ({exc})", file=sys.stderr)


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else None
    main(args)
