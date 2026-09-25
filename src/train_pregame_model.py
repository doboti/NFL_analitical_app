"""
A hiányzó összekötő láncszem: betanít egy XGBoost modellt a
src/pregame_feature_engineering.py + src/qb_profile_engineering.py által
épített historikus feature-mátrixon, majd ezt használva PREDIKCIÓT ad a
még le nem játszott (jövőbeli) meccsekre.

Eddig csak feature-eket építettünk (ahogy az eredeti kérés kérte: "Do not
write the XGBoost training loop yet"). Ez a fájl az a lépés, ami tényleges
predikcióvá alakítja őket: csapat-szintű sorokból MECCS-szintű (hazai vs.
vendég) táblát épít, és arra tanít egy modellt.

FONTOS, amit tudnod kell a jövőbeli meccsek predikciójáról:
  - A `roof` (dome/outdoors) és a pihenőnapok ELŐRE ismertek (menetrend
    alapján), ezért ezek a jövőbeli meccseknél is helyesen számolódnak.
  - A `temp`/`wind` NEM ismert előre (csak a meccs közelében/után kerül
    be az nflverse adatba) - a jövőbeli szabadtéri meccsek időjárás-
    kategóriája ezért "unknown" lesz, amíg nincs bekötve egy előrejelzés-
    forrás (pl. külső időjárás API).
  - A kezdő irányító azonosítása (`infer_likely_starter`) egy dokumentáltan
    bizonytalan heurisztika - lásd src/qb_profile_engineering.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pregame_feature_engineering import (  # noqa: E402
    FeaturePipelineConfig, InsufficientHistoryError, build_feature_matrix,
    compute_opponent_adjusted_epa, compute_weather_features, load_injuries, load_pbp,
)
from qb_profile_engineering import (  # noqa: E402
    assign_weather_bucket, build_qb_game_stats, build_qb_weather_panel, compute_qb_weather_profile,
    infer_likely_starter,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "pregame_outcome_xgb.json"
PREDICTIONS_PATH = ROOT / "data" / "predictions_latest.json"

TEAM_FEATURE_COLS = [
    "off_epa_pass_ema5", "off_epa_rush_ema5", "def_epa_pass_ema5", "def_epa_rush_ema5",
    "edsr_ema5", "adj_off_epa", "adj_def_epa", "is_dome", "injury_burden", "qb_out",
    "qb_weather_epa", "qb_weather_cpoe",
]


def determine_current_week(schedule_df: pd.DataFrame) -> tuple[int, int]:
    """A legkorábbi (season, week) pár, amelyben van még le nem játszott meccs
    (result == NaN). Ez a "jelenlegi" hét, amelyre predikciót kell adni - nem
    feltétlenül a legutóbb ELKEZDŐDÖTT hét, mert egy hét (pl. a csütörtök
    esti meccs miatt) már elkezdődhetett, miközben a hét többi meccse még
    hátravan és azokra még van értelme predikciót adni."""
    unplayed = schedule_df[schedule_df["result"].isna()]
    if unplayed.empty:
        raise ValueError("Nincs több le nem játszott meccs a menetrendben.")
    first = unplayed.sort_values(["season", "week"]).iloc[0]
    return int(first["season"]), int(first["week"])


def build_matchup_table(
    team_features: pd.DataFrame, schedule_df: pd.DataFrame, actuals: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Csapat-szintű sorokból MECCS-szintű (hazai vs. vendég) táblát épít -
    ez volt az eredeti tervben szándékosan kihagyott lépés
    ("ezt a függvényt hívva mindkét csapatra, majd a két sort egy meccs-
    szintű sorrá összefésülve...", lásd build_feature_matrix docstringje).

    Minden feature-oszlopból két verzió lesz: `home_<col>` és `away_<col>`,
    plusz egy `diff_<col>` (home - away) - a fa-alapú XGBoost mindkettőt
    hasznosítani tudja, a diff gyakran közvetlenebb jelzés.

    Args:
        actuals: opcionális build_team_game_actuals() kimenet - ha megadva,
            a `home_actual_*`/`away_actual_*` oszlopok (dobott/futott yard,
            sack) is bekerülnek, ezek a stat-predikciós regresszorok
            célváltozói.
    """
    games = schedule_df[["game_id", "season", "week", "home_team", "away_team", "result"]].copy()

    matchup = games.merge(
        team_features.add_prefix("home_"), left_on=["game_id", "home_team"],
        right_on=["home_game_id", "home_team"], how="left",
    )
    matchup = matchup.merge(
        team_features.add_prefix("away_"), left_on=["game_id", "away_team"],
        right_on=["away_game_id", "away_team"], how="left",
    )

    for col in TEAM_FEATURE_COLS:
        h, a = f"home_{col}", f"away_{col}"
        if h in matchup.columns and a in matchup.columns:
            matchup[f"diff_{col}"] = matchup[h] - matchup[a]

    matchup["home_win"] = np.where(matchup["result"] > 0, 1, np.where(matchup["result"] < 0, 0, np.nan))

    if actuals is not None:
        matchup = matchup.merge(
            actuals.add_prefix("home_"), left_on=["game_id", "home_team"],
            right_on=["home_game_id", "home_team"], how="left",
        )
        matchup = matchup.merge(
            actuals.add_prefix("away_"), left_on=["game_id", "away_team"],
            right_on=["away_game_id", "away_team"], how="left",
        )

    return matchup


def get_feature_columns(matchup_df: pd.DataFrame) -> list[str]:
    cols = []
    for col in TEAM_FEATURE_COLS:
        for prefix in ("home_", "away_", "diff_"):
            name = f"{prefix}{col}"
            if name in matchup_df.columns:
                cols.append(name)
    if "home_rest_advantage_home" in matchup_df.columns:
        cols.append("home_rest_advantage_home")
    return cols


def train_pregame_model(matchup_df: pd.DataFrame, test_season: Optional[int] = None):
    """XGBoost bináris osztályozó: nyer-e a hazai csapat. Csak a lejátszott
    (ismert kimenetelű, home_win nem NaN) meccseken tanul/tesztel."""
    labeled = matchup_df.dropna(subset=["home_win"]).copy()
    feature_cols = get_feature_columns(labeled)
    # A bool oszlopok (pl. is_dome) NaN-nal keveredve "object" dtype-ra válthatnak
    # a merge-ek során - az XGBoost csak numerikus/bool/category dtype-ot fogad el.
    for col in feature_cols:
        if labeled[col].dtype == object:
            labeled[col] = labeled[col].astype(float)

    if test_season is not None:
        train_df = labeled[labeled["season"] < test_season] if "season" in labeled.columns else labeled
        test_df = labeled[labeled["season"] == test_season] if "season" in labeled.columns else labeled.iloc[:0]
    else:
        train_df, test_df = labeled, labeled.iloc[:0]

    model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss", n_jobs=-1)
    model.fit(train_df[feature_cols], train_df["home_win"])

    metrics = {}
    if len(test_df) > 0:
        from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
        proba = model.predict_proba(test_df[feature_cols])[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = {
            "accuracy": accuracy_score(test_df["home_win"], pred),
            "log_loss": log_loss(test_df["home_win"], proba),
            "brier_score": brier_score_loss(test_df["home_win"], proba),
            "n_test": len(test_df),
        }
    return model, feature_cols, metrics


# ---------------------------------------------------------------------------
# Statisztika-predikciók (dobott/futott yard, sack) - NEM csak győz/veszít
# ---------------------------------------------------------------------------

STAT_TARGETS = ["actual_passing_yards", "actual_rushing_yards", "actual_sacks_taken"]


def build_team_game_actuals(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Egy csapat egy meccsen TÉNYLEGESEN elért statisztikái (a lejátszott
    meccs valós eredménye, nem előrejelzés) - ezek a stat-predikciós
    regressziós modellek célváltozói tréning közben."""
    passing = (
        pbp_df[pbp_df["pass_attempt"] == 1].groupby(["game_id", "posteam"])["passing_yards"]
        .sum().rename("actual_passing_yards")
    )
    rushing = (
        pbp_df[pbp_df["rush_attempt"] == 1].groupby(["game_id", "posteam"])["rushing_yards"]
        .sum().rename("actual_rushing_yards")
    )
    sacks = (
        pbp_df[pbp_df["sack"] == 1].groupby(["game_id", "posteam"]).size()
        .rename("actual_sacks_taken")
    )
    out = pd.concat([passing, rushing, sacks], axis=1).reset_index().rename(columns={"posteam": "team"})
    out[STAT_TARGETS] = out[STAT_TARGETS].fillna(0)
    return out


def train_stat_models(matchup_df: pd.DataFrame, feature_cols: list[str], test_season: Optional[int] = None):
    """Külön XGBoost REGRESSZOR minden (hazai/vendég × dobott yard/futott
    yard/sack) kombinációra - ugyanazokat a meccs-szintű feature-öket
    használva, mint a győz/veszít osztályozó."""
    from xgboost import XGBRegressor

    models = {}
    metrics = {}
    for side in ("home", "away"):
        for stat in STAT_TARGETS:
            target_col = f"{side}_{stat}"
            if target_col not in matchup_df.columns:
                continue
            labeled = matchup_df.dropna(subset=[target_col]).copy()
            for col in feature_cols:
                if labeled[col].dtype == object:
                    labeled[col] = labeled[col].astype(float)

            if test_season is not None and "season" in labeled.columns:
                train_df = labeled[labeled["season"] < test_season]
                test_df = labeled[labeled["season"] == test_season]
            else:
                train_df, test_df = labeled, labeled.iloc[:0]

            model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, n_jobs=-1)
            model.fit(train_df[feature_cols], train_df[target_col])
            models[f"{side}_{stat}"] = model

            if len(test_df) > 0:
                from sklearn.metrics import mean_absolute_error
                pred = model.predict(test_df[feature_cols])
                metrics[f"{side}_{stat}"] = {"mae": mean_absolute_error(test_df[target_col], pred)}
    return models, metrics


# ---------------------------------------------------------------------------
# Magyarázhatóság - MIÉRT ezt jósolja a modell EBBEN a konkrét meccsben
# ---------------------------------------------------------------------------

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "qb_weather_epa": "az irányító időjárás-kondicionált EPA-hatékonysága",
    "qb_weather_cpoe": "az irányító célzási pontossága (CPOE) ilyen időjárásban",
    "qb_out": "a kezdő irányító sérülés miatt hiányzik",
    "adj_off_epa": "az ellenfél-korrigált támadó hatékonyság",
    "adj_def_epa": "az ellenfél-korrigált védekező hatékonyság",
    "off_epa_pass_ema5": "a passzjáték elmúlt formája (EPA/play)",
    "off_epa_rush_ema5": "a futójáték elmúlt formája (EPA/play)",
    "def_epa_pass_ema5": "a passz-védekezés elmúlt formája",
    "def_epa_rush_ema5": "a fut-védekezés elmúlt formája",
    "edsr_ema5": "a korai down (1-2. down) siker-arány",
    "is_dome": "fedett stadion (nincs időjárási hatás)",
    "injury_burden": "a csapat összesített sérülés-terhe",
    "rest_advantage_home": "pihenőnap-előny",
}


def build_feature_row(home_row: pd.Series, away_row: pd.Series, feature_cols: list[str]) -> pd.DataFrame:
    """Egyetlen meccs feature-sorát építi fel egy hazai és egy vendég
    csapat-szintű sorból (home_/away_/diff_ prefixekkel) - ugyanaz a
    logika, mint a build_matchup_table-ben, csak egyetlen (jellemzően
    jövőbeli, még le nem játszott) meccsre alkalmazva."""
    feat = {}
    for col in TEAM_FEATURE_COLS:
        h_val = home_row.get(col, np.nan)
        a_val = away_row.get(col, np.nan)
        h_val = float(h_val) if pd.notna(h_val) else np.nan
        a_val = float(a_val) if pd.notna(a_val) else np.nan
        feat[f"home_{col}"] = h_val
        feat[f"away_{col}"] = a_val
        feat[f"diff_{col}"] = h_val - a_val if pd.notna(h_val) and pd.notna(a_val) else np.nan
    feat["home_rest_advantage_home"] = home_row.get("rest_advantage_home", np.nan)

    feat_df = pd.DataFrame([feat])
    missing = [c for c in feature_cols if c not in feat_df.columns]
    for c in missing:
        feat_df[c] = np.nan
    return feat_df[feature_cols]


def explain_prediction(model, feat_df: pd.DataFrame, feature_cols: list[str], top_n: int = 5):
    """
    XGBoost NATÍV SHAP-kontribúciója (`pred_contribs=True`) - ez nem
    általános feature-fontosság (ami az EGÉSZ modellre vonatkozna), hanem
    az EBBEN a konkrét meccsben mennyit nyomott a latba az adott feature.
    Nem igényel külön `shap` csomagot, az XGBoost saját booster-je adja.
    """
    import xgboost as xgb

    booster = model.get_booster()
    dmat = xgb.DMatrix(feat_df[feature_cols])
    contribs = booster.predict(dmat, pred_contribs=True)[0]
    pairs = list(zip(feature_cols, contribs[:-1]))  # az utolsó elem a bias/intercept
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return pairs[:top_n]


def humanize_reason(feature_name: str, contribution: float, home_team: str, away_team: str) -> str:
    """Egy (feature_name, kontribúció) párt olvasható magyar mondattá alakít."""
    if feature_name.startswith("home_"):
        base, subject = feature_name[len("home_"):], home_team
    elif feature_name.startswith("away_"):
        base, subject = feature_name[len("away_"):], away_team
    elif feature_name.startswith("diff_"):
        base, subject = feature_name[len("diff_"):], None
    else:
        base, subject = feature_name, None

    desc = FEATURE_DESCRIPTIONS.get(base, base)
    favored = home_team if contribution > 0 else away_team
    strength = "erősen" if abs(contribution) > 0.1 else "enyhén"

    if subject:
        return f"{subject}: {desc} - {strength} {favored} felé billenti az esélyt"
    return f"{desc} különbsége - {strength} {favored} felé billenti az esélyt"


def build_prediction_rows_for_week(
    pbp_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    injuries_df: pd.DataFrame,
    team_features_history: pd.DataFrame,
    target_season: int,
    target_week: int,
    config: Optional[FeaturePipelineConfig] = None,
) -> pd.DataFrame:
    """
    Egy még le nem játszott hétre építi fel a csapat-szintű feature-sorokat,
    KIZÁRÓLAG a target_week ELŐTTI adatokból (ugyanaz a leakage-védelem, mint
    a historikus panel-építésnél, csak itt egy konkrét, jövőbeli hétre).
    """
    config = config or FeaturePipelineConfig()
    teams_playing = pd.unique(
        schedule_df.loc[
            (schedule_df["season"] == target_season) & (schedule_df["week"] == target_week),
            ["home_team", "away_team"],
        ].values.ravel()
    )

    past = team_features_history[
        (team_features_history["season"] < target_season)
        | ((team_features_history["season"] == target_season) & (team_features_history["week"] < target_week))
    ]

    rows = []
    for team in teams_playing:
        team_hist = past[past["team"] == team].sort_values(["season", "week"])
        if team_hist.empty:
            continue
        row = {"team": team, "season": target_season, "week": target_week}
        for col in ["off_epa_pass", "off_epa_rush", "def_epa_pass", "def_epa_rush", "edsr"]:
            row[f"{col}_ema5"] = team_hist[col].ewm(span=config.ema_span, min_periods=1).mean().iloc[-1]
        rows.append(row)
    team_rows = pd.DataFrame(rows)

    try:
        opp_adj = compute_opponent_adjusted_epa(
            pbp_df, target_season, target_week,
            alpha=config.ridge_alpha, lookback_weeks=config.ridge_lookback_weeks,
        )
        team_rows = team_rows.merge(opp_adj[["team", "adj_off_epa", "adj_def_epa"]], on="team", how="left")
    except InsufficientHistoryError:
        team_rows["adj_off_epa"] = np.nan
        team_rows["adj_def_epa"] = np.nan

    game_ids = schedule_df.loc[
        (schedule_df["season"] == target_season) & (schedule_df["week"] == target_week), "game_id",
    ]
    team_to_game = {}
    for _, g in schedule_df[schedule_df["game_id"].isin(game_ids)].iterrows():
        team_to_game[g["home_team"]] = g["game_id"]
        team_to_game[g["away_team"]] = g["game_id"]
    team_rows["game_id"] = team_rows["team"].map(team_to_game)

    weather = compute_weather_features(schedule_df)
    team_rows = team_rows.merge(weather[["game_id", "is_dome"]], on="game_id", how="left")

    rest = schedule_df[["game_id", "home_rest", "away_rest"]].copy()
    rest["rest_advantage_home"] = rest["home_rest"] - rest["away_rest"]
    team_rows = team_rows.merge(rest[["game_id", "rest_advantage_home"]], on="game_id", how="left")

    qb_stats = build_qb_game_stats(pbp_df)
    starters = []
    for team in teams_playing:
        starter = infer_likely_starter(team, target_season, target_week, qb_stats, injuries_df)
        starters.append({"team": team, **starter})
    starters_df = pd.DataFrame(starters)
    team_rows = team_rows.merge(starters_df, on="team", how="left")

    buckets = assign_weather_bucket(compute_weather_features(schedule_df))
    qb_weather = qb_stats.merge(buckets, on="game_id", how="left")
    try:
        qb_profile = compute_qb_weather_profile(qb_weather, target_season, target_week)
    except InsufficientHistoryError:
        qb_profile = pd.DataFrame(columns=["passer_player_id", "weather_bucket", "qb_weather_epa", "qb_weather_cpoe"])

    team_rows = team_rows.merge(buckets, on="game_id", how="left")
    team_rows = team_rows.merge(
        qb_profile[["passer_player_id", "weather_bucket", "qb_weather_epa", "qb_weather_cpoe"]],
        on=["passer_player_id", "weather_bucket"], how="left",
    )

    if injuries_df is not None:
        report = injuries_df[
            (injuries_df["season"] == target_season) & (injuries_df["week"] == target_week)
        ]
        weights = {"Out": 1.0, "Doubtful": 0.75, "Questionable": 0.25}
        report = report.copy()
        report["injury_weight"] = report["report_status"].map(weights).fillna(0.0)
        burden = report.groupby("team").agg(
            injury_burden=("injury_weight", "sum"),
            qb_out=("position", lambda s: 0),
        ).reset_index()
        qb_out = (
            report[(report["position"] == "QB") & (report["report_status"] == "Out")]
            .groupby("team").size().rename("qb_out_n").reset_index()
        )
        burden = burden.drop(columns=["qb_out"]).merge(qb_out, on="team", how="left")
        burden["qb_out"] = burden["qb_out_n"].fillna(0).astype(int)
        team_rows = team_rows.merge(burden[["team", "injury_burden", "qb_out"]], on="team", how="left")
        team_rows["injury_burden"] = team_rows["injury_burden"].fillna(0)
        team_rows["qb_out"] = team_rows["qb_out"].fillna(0)

    return team_rows


if __name__ == "__main__":
    print("Adatok betöltése...")
    pbp = load_pbp(str(ROOT / "data" / "play_by_play_*.parquet"))
    schedule = pd.read_parquet(ROOT / "data" / "games.parquet")
    injuries = load_injuries(str(ROOT / "data" / "injuries_*.parquet"))

    print("Historikus feature-mátrix építése...")
    team_features = build_feature_matrix(
        str(ROOT / "data" / "play_by_play_*.parquet"), schedule_df=schedule, injuries_df=injuries,
    )

    print("Irányító-időjárás profil hozzáadása...")
    qb_stats = build_qb_game_stats(pbp)
    weather_buckets = assign_weather_bucket(compute_weather_features(schedule))
    qb_weather = qb_stats.merge(weather_buckets, on="game_id", how="left")
    qb_panel = build_qb_weather_panel(qb_weather)

    starters_hist = []
    for _, r in team_features[["team", "season", "week"]].drop_duplicates().iterrows():
        s = infer_likely_starter(r["team"], r["season"], r["week"], qb_stats, injuries)
        starters_hist.append({"team": r["team"], "season": r["season"], "week": r["week"], **s})
    starters_hist_df = pd.DataFrame(starters_hist)

    team_features = team_features.merge(starters_hist_df, on=["team", "season", "week"], how="left")
    team_features = team_features.merge(weather_buckets, on="game_id", how="left")
    team_features = team_features.merge(
        qb_panel[["passer_player_id", "weather_bucket", "season", "week", "qb_weather_epa", "qb_weather_cpoe"]],
        on=["passer_player_id", "weather_bucket", "season", "week"], how="left",
    )

    print("Valós meccs-statisztikák (dobott/futott yard, sack) betöltése...")
    actuals = build_team_game_actuals(pbp)

    print("Meccs-szintű tábla építése és modell tréningje...")
    matchup = build_matchup_table(team_features, schedule, actuals=actuals)
    model, feature_cols, metrics = train_pregame_model(matchup, test_season=2025)
    print(f"Feature-ök ({len(feature_cols)}): {feature_cols}")
    print(f"Teszt (2025) metrikák (győz/veszít): {metrics}")

    stat_models, stat_metrics = train_stat_models(matchup, feature_cols, test_season=2025)
    print(f"Stat-modellek teszt (2025) MAE: {stat_metrics}")

    model.save_model(str(MODEL_PATH))
    print(f"Modell elmentve: {MODEL_PATH}")

    target_season, target_week = determine_current_week(schedule)
    print(f"\nPredikció: {target_season} szezon {target_week}. hete (legkorábbi hét, "
          f"amelyben van még le nem játszott meccs)")
    pred_rows = build_prediction_rows_for_week(
        pbp, schedule, injuries, team_features, target_season, target_week,
    )

    # A CÉL-HÉT ÖSSZES meccse bekerül, a már lejátszottak is: ezeknél a
    # predikció ugyanúgy KIZÁRÓLAG a hét előtti (leakage-mentes) historikus
    # adatokból épül (lásd build_prediction_rows_for_week fent), a valós
    # eredményt csak UTÓLAG, összehasonlításképp csatoljuk hozzá - így
    # ellenőrizhető a modell tényleges hatékonysága már eldőlt meccseken is.
    upcoming = schedule[(schedule.season == target_season) & (schedule.week == target_week)]
    actuals_by_game = {(r["game_id"], r["team"]): r for _, r in actuals.iterrows()}
    all_predictions = []
    for _, g in upcoming.iterrows():
        home_row = pred_rows[pred_rows.team == g["home_team"]]
        away_row = pred_rows[pred_rows.team == g["away_team"]]
        if home_row.empty or away_row.empty:
            print(f"{g['away_team']} @ {g['home_team']}: nincs elég adat")
            continue
        home_row, away_row = home_row.iloc[0], away_row.iloc[0]
        home_team, away_team = g["home_team"], g["away_team"]

        feat_df = build_feature_row(home_row, away_row, feature_cols)
        home_win_prob = float(model.predict_proba(feat_df)[0, 1])

        print(f"\n=== {away_team} @ {home_team} ===")
        print(f"{home_team} győzelmi esélye: {home_win_prob:.1%}")
        print(f"Kezdő irányítók: {home_team}: {home_row.get('passer_player_name')} "
              f"[{home_row.get('confidence')}] | {away_team}: {away_row.get('passer_player_name')} "
              f"[{away_row.get('confidence')}]")

        stats = {}
        for side, team in (("home", home_team), ("away", away_team)):
            passing = float(stat_models[f"{side}_actual_passing_yards"].predict(feat_df)[0])
            rushing = float(stat_models[f"{side}_actual_rushing_yards"].predict(feat_df)[0])
            sacks = float(stat_models[f"{side}_actual_sacks_taken"].predict(feat_df)[0])
            stats[side] = {"passing_yards": passing, "rushing_yards": rushing, "sacks_taken": sacks}
            print(f"  {team}: {passing:.0f} dobott yard, {rushing:.0f} futott yard, "
                  f"{sacks:.1f} elszenvedett sack")

        print("Miért ez a predikció (top okok):")
        reasons = explain_prediction(model, feat_df, feature_cols, top_n=5)
        reason_texts = []
        for feature_name, contribution in reasons:
            contribution = float(contribution)
            text = humanize_reason(feature_name, contribution, home_team, away_team)
            reason_texts.append({"text": text, "contribution": contribution})
            print(f"  - {text} (hatás: {contribution:+.3f})")

        pred_entry = {
            "season": target_season, "week": target_week,
            "home_team": home_team, "away_team": away_team,
            "home_win_prob": round(home_win_prob, 4),
            "home_starter": {"name": home_row.get("passer_player_name"),
                              "confidence": home_row.get("confidence")},
            "away_starter": {"name": away_row.get("passer_player_name"),
                              "confidence": away_row.get("confidence")},
            "stats": stats,
            "reasons": reason_texts,
        }

        if pd.notna(g["result"]):
            game_id = g["game_id"]
            home_actual = actuals_by_game.get((game_id, home_team))
            away_actual = actuals_by_game.get((game_id, away_team))
            actual = {
                "home_score": int(g["home_score"]), "away_score": int(g["away_score"]),
                "home_win": bool(g["home_score"] > g["away_score"]),
                "stats": {
                    "home": {
                        "passing_yards": float(home_actual["actual_passing_yards"]) if home_actual is not None else None,
                        "rushing_yards": float(home_actual["actual_rushing_yards"]) if home_actual is not None else None,
                        "sacks_taken": float(home_actual["actual_sacks_taken"]) if home_actual is not None else None,
                    },
                    "away": {
                        "passing_yards": float(away_actual["actual_passing_yards"]) if away_actual is not None else None,
                        "rushing_yards": float(away_actual["actual_rushing_yards"]) if away_actual is not None else None,
                        "sacks_taken": float(away_actual["actual_sacks_taken"]) if away_actual is not None else None,
                    },
                },
            }
            pred_entry["actual"] = actual
            predicted_home_win = home_win_prob > 0.5
            hit = "TALÁLT" if predicted_home_win == actual["home_win"] else "TÉVEDETT"
            print(f"  [MÁR LEJÁTSZVA] Valós végeredmény: {home_team} {actual['home_score']} - "
                  f"{actual['away_score']} {away_team} -> a győztes-predikció {hit}")

        all_predictions.append(pred_entry)

    import json
    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=2)
    print(f"\nPredikciók elmentve: {PREDICTIONS_PATH} ({len(all_predictions)} meccs)")
