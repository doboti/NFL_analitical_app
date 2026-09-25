"""
Pre-game feature engineering pipeline az nflverse play-by-play adatokból,
XGBoost-alapú meccs-kimenetel (moneyline/spread) modellhez.

Ez a modul TUDATOSAN nem használ in-game szituációs változókat (pl. aktuális
állás, hátralévő idő - lásd src/train_win_prob.py-t azokhoz). Ehelyett stabil,
ellenfél-korrigált, gördülő pre-game metrikákat épít fel, amik egy adott hét
ELŐTT elérhető információból számolhatók.

ADATSZIVÁRGÁS (LOOKAHEAD BIAS) - a legfontosabb tervezési szempont:
Minden egyes függvény, ami egy adott (season, week, team) sorhoz feature-t
számol, KIZÁRÓLAG az adott hét ELŐTTI adatokat használhatja fel. Ezt két
helyen kell explicit kikényszeríteni:
  1. A gördülő EMA-nál (`add_rolling_ema_features`): a shift(1) hívás nélkül
     az N. hét sora tartalmazná a saját (N. heti) eredményét is.
  2. Az ellenfél-korrigált Ridge regressziónál (`compute_opponent_adjusted_epa`):
     a modellt minden egyes (season, week) predikciós pontra ÚJRA kell
     illeszteni, kizárólag az azt megelőző meccsek playjein - egy egyszeri,
     teljes szezonra illesztett modell "belelátna" a jövőbeli meccsekbe is.

ORCHESTRÁCIÓ / AIRFLOW: minden lépés egy önálló, tiszta (bemenet DataFrame ->
kimenet DataFrame) függvény mellékhatás nélkül - ezek egyenként Airflow
PythonOperator taskokká alakíthatók, a `build_feature_matrix()` pedig a teljes
DAG-ot reprezentáló orchestrátor (ami helyben egy sima Python hívás-lánc, de
1:1 megfeleltethető egy Airflow task-gráfnak).

PYSPARK ÁTÜLTETHETŐSÉG: a `groupby().agg()` és `groupby().transform()`
hívások közvetlenül megfeleltethetők PySpark `groupBy().agg()` és ablak-
függvényeknek (`Window.partitionBy("team").orderBy("week")` +
`F.avg(...).over(window)`). Az egyetlen kivétel a Ridge regresszió
(`compute_opponent_adjusted_epa`) - ez nem ablakfüggvény, skálázáshoz
`pyspark.ml.regression.LinearRegression` (elasticNetParam=0, regParam=alpha)
használható heti particionálással, vagy a heti újra-illesztés cadence-ét
(pl. csak havonta) ritkítani kell nagy adatmennyiségnél.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

class InsufficientHistoryError(ValueError):
    """Nincs elég (vagy semmi) historikus play-adat egy adott (season, week)
    predikciós ponthoz - ettől explicit meg kell különböztetni minden más
    hibát (pl. NaN érték a bemeneti adatban), különben egy silent `except
    ValueError` blokk elrejtene egy valódi adatminőségi hibát is."""


EMA_FEATURE_COLUMNS: list[str] = [
    "off_epa_pass", "off_epa_rush", "def_epa_pass", "def_epa_rush", "edsr", "qb_composite",
]


@dataclass
class FeaturePipelineConfig:
    """A pipeline futtatási paraméterei egy helyen - Airflow DAG paraméterként
    (Airflow Variable / DAG run config) közvetlenül átadható."""
    ema_span: int = 5
    reset_ema_each_season: bool = True
    ridge_alpha: float = 1.0
    ridge_lookback_weeks: Optional[int] = 32
    garbage_time_wp_low: float = 0.05
    garbage_time_wp_high: float = 0.95
    qb_epa_weight: float = 0.7
    qb_cpoe_weight: float = 0.3


# ---------------------------------------------------------------------------
# 1. Adatbetöltés, tisztítás és szűrés
# ---------------------------------------------------------------------------

def load_pbp(parquet_glob: str) -> pd.DataFrame:
    """Betölti a nyers nflverse play-by-play parquet fájlokat (DuckDB-n át,
    hogy nagy, több szezonos adathalmazoknál se kelljen mindent memóriába
    olvasni egyben a szűrés előtt)."""
    con = duckdb.connect()
    return con.execute(f"SELECT * FROM read_parquet('{parquet_glob}', union_by_name=True)").df()


def filter_garbage_time(
    df: pd.DataFrame, wp_low: float = 0.05, wp_high: float = 0.95,
) -> pd.DataFrame:
    """
    Kiszűri a "garbage time" (eldőlt meccs) playeket a labdabirtokos csapat
    győzelmi esélye (nflverse `wp` oszlop) alapján. Az ilyen helyzetekben a
    csapatok viselkedése (pass-heavy trailing, run-heavy leading, tartalék
    játékosok) torzítja a valódi erőviszonyokat reprezentáló statisztikákat.

    Args:
        df: nyers play-by-play sorok, `wp` oszloppal.
        wp_low: e alatt garbage time-nak számít (a labdabirtokos szinte
            biztosan veszít).
        wp_high: e fölött garbage time-nak számít (a labdabirtokos szinte
            biztosan nyer).
    """
    return df[df["wp"].between(wp_low, wp_high, inclusive="neither")].copy()


def filter_non_plays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eltávolítja a térdeltetéseket (qb_kneel), a spike-okat (qb_spike) és a
    tisztán büntetés miatt megismételt/lefújt akciókat (play_type=="no_play"),
    mert ezeknél nem történt valódi futtatás/passz, torzítanák az EPA-alapú
    hatékonysági mutatókat.
    """
    mask = (
        (df.get("qb_kneel", 0) != 1)
        & (df.get("qb_spike", 0) != 1)
        & (df["play_type"] != "no_play")
    )
    return df[mask].copy()


def clean_pbp(
    df: pd.DataFrame, wp_low: float = 0.05, wp_high: float = 0.95,
) -> pd.DataFrame:
    """A teljes tisztítási lánc: garbage time + nem-valódi playek kiszűrése."""
    return filter_non_plays(filter_garbage_time(df, wp_low, wp_high))


# ---------------------------------------------------------------------------
# 2. Meccs-szintű aggregációk
# ---------------------------------------------------------------------------

_GROUP_KEYS_OFF = ["game_id", "posteam", "week", "season"]
_GROUP_KEYS_DEF = ["game_id", "defteam", "week", "season"]


def offensive_epa_by_game(df: pd.DataFrame) -> pd.DataFrame:
    """Csapatonkénti, meccsenkénti támadó EPA/play, pass és rush lebontásban.

    Megjegyzés (PySpark): ez `df.groupBy("game_id","posteam","week","season")
    .agg(avg("epa"))`-ként fut le, play_type szerint két külön aggregációval.
    """
    pass_epa = (
        df[df["pass_attempt"] == 1]
        .groupby(_GROUP_KEYS_OFF)["epa"].mean()
        .rename("off_epa_pass")
    )
    rush_epa = (
        df[df["rush_attempt"] == 1]
        .groupby(_GROUP_KEYS_OFF)["epa"].mean()
        .rename("off_epa_rush")
    )
    out = pd.concat([pass_epa, rush_epa], axis=1).reset_index()
    return out.rename(columns={"posteam": "team"})


def defensive_epa_by_game(df: pd.DataFrame) -> pd.DataFrame:
    """Csapatonkénti, meccsenkénti védekező EPA/play (amit a csapat ELLEN
    engedett), pass és rush lebontásban. Alacsonyabb érték = jobb védekezés."""
    pass_epa = (
        df[df["pass_attempt"] == 1]
        .groupby(_GROUP_KEYS_DEF)["epa"].mean()
        .rename("def_epa_pass")
    )
    rush_epa = (
        df[df["rush_attempt"] == 1]
        .groupby(_GROUP_KEYS_DEF)["epa"].mean()
        .rename("def_epa_rush")
    )
    out = pd.concat([pass_epa, rush_epa], axis=1).reset_index()
    return out.rename(columns={"defteam": "team"})


def early_down_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    """1. és 2. down 'success' arány (EDSR) csapatonként/meccsenként.

    A `success` az nflverse saját, down/distance-hez igazított sikerességi
    jelzője (1/0 per play) - nem kell újraszámolni, csak átlagolni.
    """
    early = df[df["down"].isin([1, 2])]
    edsr = (
        early.groupby(_GROUP_KEYS_OFF)["success"].mean()
        .rename("edsr").reset_index()
    )
    return edsr.rename(columns={"posteam": "team"})


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if not std or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def qb_efficiency_composite(
    df: pd.DataFrame, epa_weight: float = 0.7, cpoe_weight: float = 0.3,
) -> pd.DataFrame:
    """
    Kompozit QB hatékonysági mutató passz-playeken: standardizált (z-score)
    EPA/play és CPOE súlyozott átlaga.

    A CPOE (percentage point, kb. -20..+20) és az EPA (jellemzően -1..+1
    play-enként) más skálán mozog - emiatt mindkettőt az adott meccs-mintán
    belül standardizáljuk (z-score) az összeadás előtt, nem fix osztóval,
    hogy a súlyozás adat-vezérelt és értelmezhető maradjon.
    """
    pass_df = df[(df["pass_attempt"] == 1) & df["cpoe"].notna()]

    grouped = (
        pass_df.groupby(_GROUP_KEYS_OFF)
        .agg(qb_epa=("epa", "mean"), qb_cpoe=("cpoe", "mean"))
        .reset_index()
    )
    grouped["qb_composite"] = (
        epa_weight * _zscore(grouped["qb_epa"]) + cpoe_weight * _zscore(grouped["qb_cpoe"])
    )
    return grouped.rename(columns={"posteam": "team"})[["game_id", "team", "week", "season", "qb_composite"]]


def build_team_game_table(df: pd.DataFrame) -> pd.DataFrame:
    """Egyesíti az összes meccs-szintű csapat-statisztikát egyetlen
    (game_id, team, week, season) kulcsú táblába - ez a további gördülő
    feature-számítás alapja."""
    merge_keys = ["game_id", "team", "week", "season"]
    merged = offensive_epa_by_game(df).merge(defensive_epa_by_game(df), on=merge_keys, how="outer")
    merged = merged.merge(early_down_success_rate(df), on=merge_keys, how="outer")
    merged = merged.merge(qb_efficiency_composite(df), on=merge_keys, how="outer")
    return merged.sort_values(["team", "season", "week"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Gördülő (EMA) feature-ök - LEAKAGE-BIZTOS
# ---------------------------------------------------------------------------

def add_rolling_ema_features(
    team_game_df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    span: int = 5,
    reset_each_season: bool = True,
) -> pd.DataFrame:
    """
    Exponenciális mozgóátlagot (EMA) számol minden feature-re, csapatonként,
    az utolsó `span` meccs alapján.

    ADATSZIVÁRGÁS ELKERÜLÉSE: minden oszlopra előbb `.shift(1)`-et hívunk
    csapatonként, MIELŐTT az EMA-t számolnánk. Enélkül az N. hét sora a saját
    (N. heti, predikció idején még ismeretlen) eredményét is tartalmazná az
    átlagában. A shift(1) biztosítja, hogy az N. hét sora kizárólag az N-1.
    hétig bezárólag lezajlott meccsekből számolt EMA-t lássa.

    Args:
        reset_each_season: True esetén az EMA minden szezon elején nullázódik
            (nem szivárog át az előző szezon roster-je/formája) - ez az
            alapértelmezett, konzervatívabb választás.
    """
    feature_cols = feature_cols or EMA_FEATURE_COLUMNS
    df = team_game_df.sort_values(["team", "season", "week"]).copy()
    group_keys = ["team", "season"] if reset_each_season else ["team"]

    for col in feature_cols:
        shifted = df.groupby(group_keys)[col].shift(1)
        df[f"{col}_ema{span}"] = (
            shifted.groupby([df[k] for k in group_keys])
            .transform(lambda s: s.ewm(span=span, min_periods=1).mean())
        )

    return df


# ---------------------------------------------------------------------------
# 4. Ellenfél-korrigált EPA (Ridge regresszió) - LEAKAGE-BIZTOS
# ---------------------------------------------------------------------------

def compute_opponent_adjusted_epa(
    play_df: pd.DataFrame,
    as_of_season: int,
    as_of_week: int,
    alpha: float = 1.0,
    lookback_weeks: Optional[int] = 32,
) -> pd.DataFrame:
    """
    Ellenfél-korrigált EPA becslése Ridge regresszióval, csapat-dummy
    változókkal, KIZÁRÓLAG az `as_of_season`/`as_of_week` ELŐTTI playeken.

    Módszertan: minden play egy megfigyelés, y = epa. A magyarázó változók
    az Offense_Team és Defense_Team one-hot dummy-i (referencia-kategória
    nélkül - a Ridge L2-regularizációja kezeli az emiatti multikollinearitást,
    ellentétben a hagyományos OLS-szel, ahol ez szinguláris mátrixot adna).
    A visszakapott offense-dummy együttható az adott csapat elszigetelt,
    ellenfél-erősségtől megtisztított támadó hozzájárulása; a defense-dummy
    együttható ugyanez védekezésben (mivel y az ELLENFÉL EPA-ja az adott
    defteam ellen, itt az ALACSONYABB/negatívabb érték jelent jobb védekezést).

    FONTOS: ezt a függvényt minden egyes (season, week) predikciós pontra
    ÚJRA KELL hívni (lásd `build_opponent_adjusted_panel`) - egy egyszeri,
    teljes szezonra illesztett modell a jövőbeli meccsek adatát is látná.

    Args:
        lookback_weeks: ha meg van adva, a regresszió csak az utolsó ennyi
            (season, week) hétre korlátozódik a historikus adatban (nem a
            teljes, akár 2019-ig visszanyúló múltra). Ez két okból fontos:
            (1) SEBESSÉG - enélkül minden újabb szezonnal egyre lassabb lesz
            minden egyes heti újra-illesztés (méréssel igazolva: 8 szezonnal,
            korlátozás nélkül, a teljes panel-építés 134s-ot vett igénybe,
            32 hetes korláttal ~konstans idő szezonok számától függetlenül);
            (2) MÓDSZERTAN - egy csapat 5-7 évvel korábbi EPA-ja alig
            releváns a jelenlegi rosterére (kereskedések, edzőváltás, draft) -
            a korlátlan múlt felhígítja a friss formát reprezentáló jelet.
            None esetén a teljes elérhető múltat használja (a korábbi,
            korlátozás nélküli viselkedés).

    Raises:
        InsufficientHistoryError: ha nincs elérhető historikus adat (pl. az
            első hét előtt) - ezt a hívó félnek explicit kezelnie kell, nem
            lehet csendben nulla/placeholder értéket visszaadni. Minden MÁS
            kivétel (pl. NaN a bemeneti adatban) szándékosan továbbterjed,
            nem nyelhető el csendben.
    """
    history = play_df[
        (play_df["season"] < as_of_season)
        | ((play_df["season"] == as_of_season) & (play_df["week"] < as_of_week))
    ]

    if lookback_weeks is not None and not history.empty:
        recent_weeks = (
            history[["season", "week"]].drop_duplicates()
            .sort_values(["season", "week"])
            .tail(lookback_weeks)
        )
        history = history.merge(recent_weeks, on=["season", "week"], how="inner")
    # Az epa NaN lehet néhány speciális playnél (pl. bizonyos kickoff/timeout
    # jellegű sorok) - ezeket ki kell zárni a regresszióból, MIELŐTT illesztünk,
    # különben a sklearn Ridge.fit() egy nehezen értelmezhető hibával áll le.
    history = history[history["epa"].notna()]
    if history.empty:
        raise InsufficientHistoryError(
            f"Nincs (epa-val rendelkező) historikus play-adat a(z) {as_of_season} szezon "
            f"{as_of_week}. hete előtt - ez a függvény nem adhat vissza értéket "
            f"adatszivárgás nélkül."
        )

    off_dummies = pd.get_dummies(history["posteam"], prefix="off", dtype=float)
    def_dummies = pd.get_dummies(history["defteam"], prefix="def", dtype=float)
    X = pd.concat([off_dummies, def_dummies], axis=1)
    y = history["epa"].to_numpy()

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y)
    coefs = pd.Series(model.coef_, index=X.columns)

    off_adj = coefs[coefs.index.str.startswith("off_")]
    off_adj.index = off_adj.index.str.removeprefix("off_")
    def_adj = coefs[coefs.index.str.startswith("def_")]
    def_adj.index = def_adj.index.str.removeprefix("def_")

    result = pd.merge(
        off_adj.rename("adj_off_epa").rename_axis("team").reset_index(),
        def_adj.rename("adj_def_epa").rename_axis("team").reset_index(),
        on="team", how="outer",
    )
    result["season"] = as_of_season
    result["week"] = as_of_week
    return result


def build_opponent_adjusted_panel(
    play_df: pd.DataFrame, alpha: float = 1.0, lookback_weeks: Optional[int] = 32,
) -> pd.DataFrame:
    """Végigmegy a play_df-ben szereplő összes (season, week) kombináción, és
    mindegyikhez kiszámolja az akkor még csak a MEGELŐZŐ adatokból elérhető
    ellenfél-korrigált EPA-t. Az első hét(ek), ahol még nincs historikus
    adat, automatikusan kimaradnak (lásd compute_opponent_adjusted_epa).

    A `lookback_weeks` jelentése: lásd compute_opponent_adjusted_epa - ez a
    paraméter tartja korlátok között a futásidőt sok szezonos adatnál."""
    combos = play_df[["season", "week"]].drop_duplicates().sort_values(["season", "week"])

    panels = []
    for season, week in combos.itertuples(index=False):
        try:
            panels.append(
                compute_opponent_adjusted_epa(
                    play_df, season, week, alpha=alpha, lookback_weeks=lookback_weeks,
                )
            )
        except InsufficientHistoryError:
            continue

    if not panels:
        return pd.DataFrame(columns=["team", "adj_off_epa", "adj_def_epa", "season", "week"])
    return pd.concat(panels, ignore_index=True)


# ---------------------------------------------------------------------------
# 5. Kontextuális feature-ök
# ---------------------------------------------------------------------------

def compute_rest_advantage(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pihenőnap-előny (Rest Advantage) meccsenként: hazai csapat pihenőnapjai
    mínusz vendég csapat pihenőnapjai.

    Bemenet: nflverse menetrend-tábla (`nflreadr::load_schedules()` szintű),
    aminek legalább `game_id`, `home_team`, `away_team`, `home_rest`,
    `away_rest` oszlopai vannak - ezeket az nflverse már kiszámolja az előző
    meccs dátuma alapján, itt csak egyetlen jelzésszámmá vonjuk össze.
    """
    required = {"game_id", "home_team", "away_team", "home_rest", "away_rest"}
    missing = required - set(schedule_df.columns)
    if missing:
        raise ValueError(f"Hiányzó oszlopok a schedule táblából: {sorted(missing)}")

    out = schedule_df[list(required)].copy()
    out["rest_advantage_home"] = out["home_rest"] - out["away_rest"]
    return out


def compute_weather_features(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Időjárási kontextus meccsenként (nflverse `games` tábla `roof`/`temp`/
    `wind` oszlopaiból).

    Fedett/klimatizált stadionoknál (roof in dome/closed) a nyers temp/wind
    mérés hiányzik az nflverse adatban (irreleváns, klimatizált) - ilyenkor
    semleges értékekre töltjük ki (70°F, 0 mph szél), az `is_dome` flag
    jelzi a modellnek, hogy ez mesterséges kitöltés, nem valós mérés (enélkül
    a modell tévesen "szélcsendes, kellemes idő" mintaként tanulná meg a
    dome-okat, ami torzítaná a szabadtéri hideg/szeles meccsek hatását).

    MEGJEGYZÉS: a szabadtéri (`is_dome=False`) meccsek kb. 19%-ánál a nyers
    nflverse adatban ÍGY IS hiányzik a temp/wind mérés (historikus adathiány,
    nem a mi hibánk) - ezeket SZÁNDÉKOSAN nem töltjük ki mesterségesen, NaN
    marad. Az XGBoost natívan, jól kezeli a hiányzó értékeket (a fa-alapú
    split logika automatikusan tanulja meg, melyik ágra irányítsa őket).
    """
    required = {"game_id", "roof", "temp", "wind"}
    missing = required - set(schedule_df.columns)
    if missing:
        raise ValueError(f"Hiányzó oszlopok a schedule táblából: {sorted(missing)}")

    out = schedule_df[list(required)].copy()
    out["is_dome"] = out["roof"].isin(["dome", "closed"])
    out["temp_f"] = out["temp"].where(~out["is_dome"], 70.0)
    out["wind_mph"] = out["wind"].where(~out["is_dome"], 0.0)
    return out[["game_id", "is_dome", "temp_f", "wind_mph"]]


INJURY_STATUS_WEIGHTS: dict[str, float] = {"Out": 1.0, "Doubtful": 0.75, "Questionable": 0.25}


def load_injuries(parquet_glob: str) -> pd.DataFrame:
    """Betölti az nflverse heti sérülés-jelentéseket (injuries_<year>.parquet
    fájlok, `nflreadr::load_injuries()` szintű tartalommal)."""
    con = duckdb.connect()
    return con.execute(f"SELECT * FROM read_parquet('{parquet_glob}', union_by_name=True)").df()


def compute_injury_burden(
    injuries_df: pd.DataFrame, status_weights: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Csapatonkénti/hetenkénti sérülés-teher: a jelentett játékosok súlyozott
    összege (Out=1.0, Doubtful=0.75, Questionable=0.25 alapból), kiegészítve
    egy külön `qb_out` jelzővel (a pozíció kiemelt fontossága miatt egy
    egyszerű súlyozott összeg nem adná vissza megfelelően a hatását).

    NEM IGÉNYEL shift(1)-et / külön adatszivárgás-védelmet: a heti injury
    report MÁR ELEVE az adott heti meccs ELŐTT (jellemzően szerda-péntek)
    kerül közzétételre - ez natívan pre-game információ, ellentétben az
    EPA-alapú statisztikákkal, amik a LEJÁTSZOTT meccs UTÁNI eredmények és
    ezért shift(1)-et igényelnek. Itt a (season, week) egyszerűen az adott
    hétre vonatkozó, a hét mérkőzése előtt ismert állapotot jelenti.
    """
    weights = status_weights or INJURY_STATUS_WEIGHTS
    df = injuries_df.copy()
    df["injury_weight"] = df["report_status"].map(weights).fillna(0.0)

    burden = (
        df.groupby(["season", "week", "team"])
        .agg(
            injury_burden=("injury_weight", "sum"),
            players_out=("report_status", lambda s: (s == "Out").sum()),
        )
        .reset_index()
    )

    qb_out = (
        df[(df["position"] == "QB") & (df["report_status"] == "Out")]
        .groupby(["season", "week", "team"])
        .size()
        .rename("qb_out")
        .reset_index()
    )
    burden = burden.merge(qb_out, on=["season", "week", "team"], how="left")
    burden["qb_out"] = burden["qb_out"].fillna(0).astype(int)
    return burden


# ---------------------------------------------------------------------------
# Orchestrátor
# ---------------------------------------------------------------------------

def build_feature_matrix(
    pbp_parquet_glob: str,
    schedule_df: Optional[pd.DataFrame] = None,
    injuries_df: Optional[pd.DataFrame] = None,
    config: Optional[FeaturePipelineConfig] = None,
) -> pd.DataFrame:
    """
    Végponttól-végpontig lefuttatja a pre-game feature pipeline-t.

    Kimenet: egy (season, week, team, game_id) granularitású DataFrame,
    ahol minden feature-oszlop KIZÁRÓLAG az adott hét előtt (vagy az adott
    hét mérkőzése előtt közzétett, pl. injury report) elérhető adatból
    származik - tréning-kész egy XGBoost meccs-kimenetel modellhez (ezt a
    függvényt hívva mindkét csapatra, majd a két sort egy meccs-szintű sorrá
    összefésülve kapható meg a végleges, home/away különbségi feature-öket
    is tartalmazó tréning tábla - ez a lépés szándékosan nincs itt, mert a
    "Do not write the XGBoost training loop yet" megkötés miatt a nyers,
    csapat-szintű feature táblát adjuk vissza).

    Args:
        schedule_df: opcionális nflverse `games` tábla - ha megadva, a
            pihenőnap-előny (`rest_advantage_home`) ÉS az időjárási
            feature-ök (`is_dome`, `temp_f`, `wind_mph`) is bekerülnek,
            game_id alapján összefésülve (mindkét csapat sora ugyanazt az
            érétket kapja, hiszen ezek meccs-szintű, nem csapat-szintű
            attribútumok).
        injuries_df: opcionális nflverse heti injury report tábla - ha
            megadva, a sérülés-teher (`injury_burden`, `players_out`,
            `qb_out`) is bekerül, (season, week, team) alapján összefésülve.

    Airflow-ban ez a függvény egy DAG-ot reprezentál: minden belső lépés
    (clean_pbp -> build_team_game_table -> add_rolling_ema_features ->
    build_opponent_adjusted_panel -> merge) külön PythonOperator taskként
    futtatható, parquet/DB checkpoint-okkal a lépések között.
    """
    config = config or FeaturePipelineConfig()

    raw = load_pbp(pbp_parquet_glob)
    clean = clean_pbp(raw, config.garbage_time_wp_low, config.garbage_time_wp_high)

    team_game = build_team_game_table(clean)
    team_game = add_rolling_ema_features(
        team_game, span=config.ema_span, reset_each_season=config.reset_ema_each_season,
    )

    opponent_adjusted = build_opponent_adjusted_panel(
        clean, alpha=config.ridge_alpha, lookback_weeks=config.ridge_lookback_weeks,
    )
    features = team_game.merge(opponent_adjusted, on=["team", "season", "week"], how="left")

    if schedule_df is not None:
        rest = compute_rest_advantage(schedule_df)
        features = features.merge(rest, on="game_id", how="left")
        weather = compute_weather_features(schedule_df)
        features = features.merge(weather, on="game_id", how="left")

    if injuries_df is not None:
        injury_burden = compute_injury_burden(injuries_df)
        features = features.merge(injury_burden, on=["season", "week", "team"], how="left")
        # Ha egy csapatnak nincs sora az injury report-ban (pl. hiányzó adat),
        # 0 sérülés-teher az ésszerű alapértelmezés, nem NaN (ami XGBoost-ban
        # is mást jelentene: "hiányzó infó" vs. "nincs bejelentett sérülés").
        for col in ["injury_burden", "players_out", "qb_out"]:
            if col in features.columns:
                features[col] = features[col].fillna(0)

    return features


if __name__ == "__main__":
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    glob = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "play_by_play_*.parquet")

    games_path = ROOT / "data" / "games.parquet"
    schedule_df = pd.read_parquet(games_path) if games_path.exists() else None
    if schedule_df is None:
        print(f"[figyelem] {games_path} nem található - rest/időjárás feature-ök nélkül fut.")

    injuries_glob = str(ROOT / "data" / "injuries_*.parquet")
    injuries_df = load_injuries(injuries_glob) if list(ROOT.glob("data/injuries_*.parquet")) else None
    if injuries_df is None:
        print("[figyelem] Nincs data/injuries_*.parquet - sérülés feature-ök nélkül fut.")

    matrix = build_feature_matrix(glob, schedule_df=schedule_df, injuries_df=injuries_df)
    out_path = ROOT / "data" / "pregame_feature_matrix.parquet"
    matrix.to_parquet(out_path, index=False)
    print(f"Feature mátrix elmentve: {out_path} ({len(matrix):,} sor, {len(matrix.columns)} oszlop)")
