"""
Irányító-szintű (QB), időjárás-kondicionált teljesítmény-profil.

Ez a modul KIEGÉSZÍTI a src/pregame_feature_engineering.py csapat-szintű
feature-jeit egy játékos-szintű réteggel. A csapat-szintű `qb_composite`
feature elmossa a különbséget "a kezdő irányító formában van" és "a kezdő
megsérült, a tartalék lépett be" között - ha egy csapat lecseréli az
irányítóját, a csapat-szintű gördülő átlag a KETTŐ keverékét mutatná,
miközben minket kifejezetten a KONKRÉT, pályára lépő irányító várható
teljesítménye érdekel, adott körülmények (időjárás) között.

PÉLDA (ami a modult motiválta): ha Joe Burrow esőben/hidegben gyengébben
teljesít, ez csak akkor látszik, ha az ő EPA/CPOE-ját KÜLÖN követjük
időjárási kategóriánként, nem a Bengals csapat összesített heti átlagában.

ADATSZIVÁRGÁS: ugyanaz az elv, mint a csapat-szintű modulban - minden
profil-számítás kizárólag az adott hét ELŐTTI meccsekből épül fel
(lásd compute_qb_weather_profile).

KORLÁTOZÁS (fontos, hogy tudj róla): az nflverse `games` táblában NINCS
külön csapadék (eső/hó) mező, csak hőmérséklet (`temp`) és szélsebesség
(`wind`) - lásd src/pregame_feature_engineering.compute_weather_features().
A "rossz idő" kategóriát ezekből közelítjük (hideg és/vagy szeles) - a
tiszta "esik-e az eső" jelet ebből NEM lehet megkülönböztetni egy száraz,
hideg naptól. Ha pontosabb csapadék-adat kell, egy külső, dátum+helyszín
alapú historikus időjárás-API (pl. Visual Crossing, NOAA) bevonása
szükséges - ez jelenleg nincs megoldva.

MÁSIK KORLÁTOZÁS: a `infer_likely_starter` egy DOKUMENTÁLTAN bizonytalan
heurisztika, nem tényadat - egy még le nem játszott meccs kezdő
irányítóját valós időben csak friss csapathírekből lehet biztosan tudni.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from pregame_feature_engineering import InsufficientHistoryError

WEATHER_BUCKETS: list[str] = ["dome", "cold", "windy", "mild", "unknown"]


def build_qb_game_stats(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Irányító-szintű (NEM csapat-szintű) meccsenkénti EPA/CPOE, a
    `passer_player_id` alapján csoportosítva - ez a modul alapköve, mert
    ez különbözteti meg a kezdő és egy adott meccsen esetleg becserélt
    tartalék irányító teljesítményét.
    """
    pass_df = pbp_df[(pbp_df["pass_attempt"] == 1) & pbp_df["passer_player_id"].notna()]
    stats = (
        pass_df.groupby(["game_id", "season", "week", "posteam", "passer_player_id", "passer_player_name"])
        .agg(qb_epa=("epa", "mean"), qb_cpoe=("cpoe", "mean"), qb_dropbacks=("epa", "size"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    return stats


def assign_weather_bucket(
    weather_df: pd.DataFrame, cold_threshold_f: float = 45.0, wind_threshold_mph: float = 15.0,
) -> pd.DataFrame:
    """
    A src/pregame_feature_engineering.compute_weather_features() kimenetét
    (game_id, is_dome, temp_f, wind_mph) durva kategóriákba sorolja. A
    windy-t a cold ELÉ soroljuk (előbb nézzük), mert egy szeles-hideg meccs
    a passzjátékra elsősorban a szél miatt nehéz, azt tekintjük domináns
    tényezőnek - ez egy dokumentált, egyszerűsítő tervezési döntés.
    """
    df = weather_df.copy()

    def _bucket(row) -> str:
        if row["is_dome"]:
            return "dome"
        if pd.isna(row["temp_f"]) and pd.isna(row["wind_mph"]):
            return "unknown"
        if pd.notna(row["wind_mph"]) and row["wind_mph"] >= wind_threshold_mph:
            return "windy"
        if pd.notna(row["temp_f"]) and row["temp_f"] <= cold_threshold_f:
            return "cold"
        return "mild"

    df["weather_bucket"] = df.apply(_bucket, axis=1)
    return df[["game_id", "weather_bucket"]]


def compute_qb_weather_profile(
    qb_game_weather: pd.DataFrame,
    as_of_season: int,
    as_of_week: int,
    shrinkage_k: float = 8.0,
) -> pd.DataFrame:
    """
    Minden irányítóra kiszámolja az adott IDŐJÁRÁSI kategóriában várható
    EPA/CPOE-t, EMPIRIKUS BAYES ZSUGORÍTÁSSAL az irányító ÖSSZES (bármilyen
    időjárású) átlaga felé.

    Miért kell a zsugorítás: a legtöbb irányítónak kevés meccse van
    szélsőséges körülmények között (pl. 2-3 hó/nagy szél meccs egy karrier
    alatt) - egy nyers, kis mintás átlag megbízhatatlan és túlreagálná a
    véletlen ingadozást. A képlet:

        shrunk = (n_bucket * bucket_mean + shrinkage_k * overall_mean)
                 / (n_bucket + shrinkage_k)

    minél kevesebb meccse volt az irányítónak abban a kategóriában, annál
    inkább az összesített (minden körülmény közötti) átlaga felé húz a
    becslés - `shrinkage_k` a "hány meccsnyi bizalmat" adunk az összesített
    átlagnak (nagyobb érték = erősebb zsugorítás/óvatosabb becslés).

    Args:
        qb_game_weather: build_qb_game_stats() kimenete, `weather_bucket`
            oszloppal kiegészítve (lásd assign_weather_bucket).

    Raises:
        InsufficientHistoryError: nincs historikus adat az adott hét előtt.
    """
    history = qb_game_weather[
        (qb_game_weather["season"] < as_of_season)
        | ((qb_game_weather["season"] == as_of_season) & (qb_game_weather["week"] < as_of_week))
    ]
    if history.empty:
        raise InsufficientHistoryError(
            f"Nincs historikus irányító-adat a(z) {as_of_season} szezon {as_of_week}. hete előtt."
        )

    overall = (
        history.groupby("passer_player_id")
        .agg(overall_epa=("qb_epa", "mean"), overall_cpoe=("qb_cpoe", "mean"))
        .reset_index()
    )
    bucketed = (
        history.groupby(["passer_player_id", "weather_bucket"])
        .agg(bucket_epa=("qb_epa", "mean"), bucket_cpoe=("qb_cpoe", "mean"), n_bucket=("qb_epa", "size"))
        .reset_index()
    )
    merged = bucketed.merge(overall, on="passer_player_id")
    merged["qb_weather_epa"] = (
        (merged["n_bucket"] * merged["bucket_epa"] + shrinkage_k * merged["overall_epa"])
        / (merged["n_bucket"] + shrinkage_k)
    )
    merged["qb_weather_cpoe"] = (
        (merged["n_bucket"] * merged["bucket_cpoe"] + shrinkage_k * merged["overall_cpoe"])
        / (merged["n_bucket"] + shrinkage_k)
    )
    merged["season"] = as_of_season
    merged["week"] = as_of_week
    return merged[
        ["passer_player_id", "weather_bucket", "qb_weather_epa", "qb_weather_cpoe", "n_bucket", "season", "week"]
    ]


def build_qb_weather_panel(qb_game_weather: pd.DataFrame, shrinkage_k: float = 8.0) -> pd.DataFrame:
    """Végigmegy minden (season, week) kombináción, mindegyikhez az akkor
    még csak MEGELŐZŐ adatokból elérhető irányító-időjárás profilt számolva."""
    combos = qb_game_weather[["season", "week"]].drop_duplicates().sort_values(["season", "week"])

    panels = []
    for season, week in combos.itertuples(index=False):
        try:
            panels.append(compute_qb_weather_profile(qb_game_weather, season, week, shrinkage_k))
        except InsufficientHistoryError:
            continue

    if not panels:
        return pd.DataFrame(
            columns=["passer_player_id", "weather_bucket", "qb_weather_epa", "qb_weather_cpoe", "n_bucket",
                     "season", "week"]
        )
    return pd.concat(panels, ignore_index=True)


def infer_likely_starter(
    team: str,
    as_of_season: int,
    as_of_week: int,
    qb_game_stats: pd.DataFrame,
    injuries_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Legjobb becslés arra, KI fog irányítóként pályára lépni egy még le nem
    játszott meccsen.

    EZ INHERENSEN BIZONYTALAN - egy jövőbeli meccs kezdő irányítóját valós
    időben csak friss csapathírekből (beat reporterek, hivatalos bejelentés)
    lehet biztosan tudni. Ez a függvény egy dokumentált, adat-alapú
    HEURISZTIKÁT ad, nem tényt:

      1. A csapat legutóbbi (as_of előtti) meccsének domináns (legtöbb
         dropback-kal rendelkező) irányítója az alapértelmezett jelölt.
      2. Ha az injuries_df szerint EZ az irányító "Out" az adott héten
         (gsis_id alapján, ami megegyezik a PBP passer_player_id-jével),
         megkeressük a csapat legutóbbi MÁSIK irányítóját tartalék-
         jelöltnek - durva közelítés arra, "ki a #2 a mélységi listán",
         pontosabbhoz az nflverse depth_charts adatforrása kellene
         (ezt a projekt jelenleg nem tölti le).

    A visszaadott dict `confidence` mezője jelzi, melyik ág futott le - ha
    nem "confirmed_healthy", a dashboardnak ezt EXPLICIT jeleznie kell a
    felhasználónak, nem szabad tényként feltüntetni.

    ISMERT, VALÓS ADATON TALÁLT KORLÁTOZÁS: ez a függvény KIZÁRÓLAG a
    sérülés-jelentést nézi - egy idénközbeni TRADE, elengedés vagy
    teljesítmény miatti lecserélés esetén "confirmed_healthy"-t adna vissza
    a már NEM a csapatnál lévő játékosra, hiszen ő nincs a sérülés-
    jelentésben (nem sérült, csak elment). Konkrét, ezzel a projekttel
    ellenőrzött példa: Josh Dobbs (ARI, 2023) idénközben Minnesotába
    került, sérülés-bejegyzés nélkül - a függvény ekkor tévesen őt adná
    vissza. Ennek kezeléséhez az nflverse roster/transaction adata
    (jelenleg nincs letöltve ebbe a projektbe) is kellene.
    """
    history = qb_game_stats[
        (qb_game_stats["team"] == team)
        & (
            (qb_game_stats["season"] < as_of_season)
            | ((qb_game_stats["season"] == as_of_season) & (qb_game_stats["week"] < as_of_week))
        )
    ].sort_values(["season", "week"])

    if history.empty:
        return {"passer_player_id": None, "passer_player_name": None, "confidence": "no_history"}

    last_game_id = history.iloc[-1]["game_id"]
    last_game = history[history["game_id"] == last_game_id]
    last_starter = last_game.sort_values("qb_dropbacks", ascending=False).iloc[0]

    is_injured = False
    if injuries_df is not None and "gsis_id" in injuries_df.columns:
        report = injuries_df[
            (injuries_df["season"] == as_of_season)
            & (injuries_df["week"] == as_of_week)
            & (injuries_df["team"] == team)
            & (injuries_df["gsis_id"] == last_starter["passer_player_id"])
        ]
        is_injured = bool((report["report_status"] == "Out").any())

    if not is_injured:
        return {
            "passer_player_id": last_starter["passer_player_id"],
            "passer_player_name": last_starter["passer_player_name"],
            "confidence": "confirmed_healthy",
        }

    other = history[history["passer_player_id"] != last_starter["passer_player_id"]]
    if other.empty:
        return {
            "passer_player_id": None, "passer_player_name": None,
            "confidence": "starter_injured_no_known_backup",
        }
    backup = other.sort_values(["season", "week"]).iloc[-1]
    return {
        "passer_player_id": backup["passer_player_id"],
        "passer_player_name": backup["passer_player_name"],
        "confidence": "inferred_due_to_injury",
    }
