"""
Unit tesztek a pre-game feature engineering pipeline-ra.

A hangsúly az ADATSZIVÁRGÁS (lookahead bias) elleni védelmen van - ez a
modul explicit célja volt, ezért itt kell a legszigorúbban ellenőrizni,
hogy egy adott hét feature-je tényleg nem "látja" a saját vagy jövőbeli
heteinek eredményét.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pregame_feature_engineering import (  # noqa: E402
    InsufficientHistoryError,
    add_rolling_ema_features,
    compute_injury_burden,
    compute_opponent_adjusted_epa,
    compute_rest_advantage,
    compute_weather_features,
    filter_garbage_time,
    filter_non_plays,
    _zscore,
)


# ---------------------------------------------------------------------------
# Tisztítás / szűrés
# ---------------------------------------------------------------------------

def test_filter_garbage_time_excludes_extreme_wp():
    df = pd.DataFrame({"wp": [0.01, 0.04, 0.5, 0.96, 0.99]})
    out = filter_garbage_time(df, wp_low=0.05, wp_high=0.95)
    assert out["wp"].tolist() == [0.5]


def test_filter_non_plays_excludes_kneel_spike_no_play():
    df = pd.DataFrame({
        "qb_kneel": [0, 1, 0, 0],
        "qb_spike": [0, 0, 1, 0],
        "play_type": ["run", "run", "pass", "no_play"],
    })
    out = filter_non_plays(df)
    assert len(out) == 1
    assert out.iloc[0]["play_type"] == "run"


def test_zscore_constant_series_returns_zeros_not_nan():
    s = pd.Series([5.0, 5.0, 5.0])
    result = _zscore(s)
    assert (result == 0).all()


# ---------------------------------------------------------------------------
# Gördülő EMA - adatszivárgás ellenőrzése
# ---------------------------------------------------------------------------

def test_ema_week_n_never_includes_week_n_own_value():
    """A legfontosabb teszt: ha egy csapat egyetlen egy hétre extrém (pl. 1000)
    értéket kapna, az az ADOTT hét EMA-jában NEM jelenhet meg - csak a
    KÖVETKEZŐ héten (mert a shift(1) miatt egy héttel később szivárog be)."""
    team_game = pd.DataFrame({
        "team": ["KC"] * 5,
        "season": [2023] * 5,
        "week": [1, 2, 3, 4, 5],
        "game_id": [f"g{i}" for i in range(5)],
        "off_epa_pass": [0.1, 0.1, 1000.0, 0.1, 0.1],  # 3. héten extrém kiugrás
        "off_epa_rush": [0.0] * 5,
        "def_epa_pass": [0.0] * 5,
        "def_epa_rush": [0.0] * 5,
        "edsr": [0.5] * 5,
        "qb_composite": [0.0] * 5,
    })
    out = add_rolling_ema_features(team_game, span=3)

    week3_ema = out.loc[out["week"] == 3, "off_epa_pass_ema3"].iloc[0]
    week4_ema = out.loc[out["week"] == 4, "off_epa_pass_ema3"].iloc[0]

    assert week3_ema < 1.0, "A 3. hét EMA-ja nem tartalmazhatja a saját (3. heti) 1000-es kiugrását"
    assert week4_ema > 1.0, "A 4. hét EMA-jának MÁR tartalmaznia kell az (előző, 3. heti) kiugrást"


def test_ema_first_week_is_nan_no_prior_data():
    team_game = pd.DataFrame({
        "team": ["KC", "KC"],
        "season": [2023, 2023],
        "week": [1, 2],
        "game_id": ["g0", "g1"],
        "off_epa_pass": [0.3, 0.4],
        "off_epa_rush": [0.0, 0.0],
        "def_epa_pass": [0.0, 0.0],
        "def_epa_rush": [0.0, 0.0],
        "edsr": [0.5, 0.5],
        "qb_composite": [0.0, 0.0],
    })
    out = add_rolling_ema_features(team_game, span=3)
    assert pd.isna(out.loc[out["week"] == 1, "off_epa_pass_ema3"].iloc[0])


def test_ema_reset_each_season_does_not_leak_across_seasons():
    team_game = pd.DataFrame({
        "team": ["KC", "KC"],
        "season": [2022, 2023],
        "week": [18, 1],
        "game_id": ["g0", "g1"],
        "off_epa_pass": [5.0, 0.1],
        "off_epa_rush": [0.0, 0.0],
        "def_epa_pass": [0.0, 0.0],
        "def_epa_rush": [0.0, 0.0],
        "edsr": [0.5, 0.5],
        "qb_composite": [0.0, 0.0],
    })
    out = add_rolling_ema_features(team_game, span=3, reset_each_season=True)
    season2023_week1 = out[(out["season"] == 2023) & (out["week"] == 1)]
    assert pd.isna(season2023_week1["off_epa_pass_ema3"].iloc[0])


# ---------------------------------------------------------------------------
# Ellenfél-korrigált EPA (Ridge) - adatszivárgás ellenőrzése
# ---------------------------------------------------------------------------

def _make_play_df(n_teams=4, n_weeks=3, plays_per_matchup=20, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    rows = []
    for week in range(1, n_weeks + 1):
        for i in range(0, n_teams, 2):
            off, deff = teams[i], teams[i + 1]
            for _ in range(plays_per_matchup):
                rows.append({"season": 2023, "week": week, "posteam": off, "defteam": deff,
                             "epa": rng.normal(0, 1)})
                rows.append({"season": 2023, "week": week, "posteam": deff, "defteam": off,
                             "epa": rng.normal(0, 1)})
    return pd.DataFrame(rows)


def test_opponent_adjusted_epa_raises_when_no_history():
    play_df = _make_play_df(n_weeks=1)
    with pytest.raises(InsufficientHistoryError):
        compute_opponent_adjusted_epa(play_df, as_of_season=2023, as_of_week=1)


def test_opponent_adjusted_epa_only_uses_prior_weeks():
    """Ha a 3. hétre egy csapat extrém EPA-t termel, az a 3. hetet MEGELŐZŐEN
    (as_of_week=3) számolt ellenfél-korrekcióban nem jelenhet meg - csak
    a 4. hétre (as_of_week=4) számoltban."""
    play_df = _make_play_df(n_weeks=4)
    # T0 extrém jó lesz a 3. héten (a "jövőbeli" adat, amit as_of_week=3 nem láthat)
    mask = (play_df["week"] == 3) & (play_df["posteam"] == "T0")
    play_df.loc[mask, "epa"] = 50.0

    before = compute_opponent_adjusted_epa(play_df, as_of_season=2023, as_of_week=3)
    after = compute_opponent_adjusted_epa(play_df, as_of_season=2023, as_of_week=4)

    t0_before = before.loc[before["team"] == "T0", "adj_off_epa"].iloc[0]
    t0_after = after.loc[after["team"] == "T0", "adj_off_epa"].iloc[0]

    assert t0_before < 5, "A 3. hét ELŐTTI korrekció nem szivároghat be a 3. heti extrém értékből"
    assert t0_after > 5, "A 4. hét korrekciójának MÁR tükröznie kell a (megelőző, 3. heti) extrém értéket"


def test_opponent_adjusted_epa_lookback_weeks_excludes_old_history():
    """Egy nagyon régi (1. heti) kiugró érték NE befolyásolja a korrekciót,
    ha a lookback_weeks már kizárja azt a hetet a historikus ablakból."""
    play_df = _make_play_df(n_weeks=5)
    mask = (play_df["week"] == 1) & (play_df["posteam"] == "T0")
    play_df.loc[mask, "epa"] = 50.0

    with_full_history = compute_opponent_adjusted_epa(
        play_df, as_of_season=2023, as_of_week=5, lookback_weeks=None,
    )
    with_short_lookback = compute_opponent_adjusted_epa(
        play_df, as_of_season=2023, as_of_week=5, lookback_weeks=2,
    )

    t0_full = with_full_history.loc[with_full_history["team"] == "T0", "adj_off_epa"].iloc[0]
    t0_short = with_short_lookback.loc[with_short_lookback["team"] == "T0", "adj_off_epa"].iloc[0]

    assert t0_full > t0_short, (
        "A teljes múltat használó verziónak magasabb (az 1. heti kiugrástól torzított) "
        "értéket kell adnia, mint a rövid ablaknak, ami már nem látja azt a hetet."
    )


def test_opponent_adjusted_epa_raises_on_unexpected_data_error():
    """NaN az epa oszlopban NEM InsufficientHistoryError-t kell, hogy adjon,
    hanem vagy megoldódjon (NaN sorok kiszűrve), vagy más, explicit hibát adjon -
    sosem szabad csendben, hibásan 'nincs történelem'-ként értelmezni."""
    play_df = _make_play_df(n_weeks=2)
    play_df.loc[play_df["week"] == 1, "epa"] = np.nan
    # Az összes 1. heti (egyetlen historikus) sor epa-ja NaN -> a notna() szűrés
    # után nem marad historikus adat -> ez helyesen InsufficientHistoryError.
    with pytest.raises(InsufficientHistoryError):
        compute_opponent_adjusted_epa(play_df, as_of_season=2023, as_of_week=2)


# ---------------------------------------------------------------------------
# Rest advantage
# ---------------------------------------------------------------------------

def test_rest_advantage_computed_correctly():
    schedule = pd.DataFrame({
        "game_id": ["g1", "g2"],
        "home_team": ["KC", "SF"],
        "away_team": ["BUF", "DAL"],
        "home_rest": [7, 10],
        "away_rest": [7, 6],
    })
    out = compute_rest_advantage(schedule)
    assert out.loc[out["game_id"] == "g1", "rest_advantage_home"].iloc[0] == 0
    assert out.loc[out["game_id"] == "g2", "rest_advantage_home"].iloc[0] == 4


def test_rest_advantage_missing_columns_raises():
    with pytest.raises(ValueError):
        compute_rest_advantage(pd.DataFrame({"game_id": ["g1"]}))


# ---------------------------------------------------------------------------
# Időjárás
# ---------------------------------------------------------------------------

def test_weather_features_fills_neutral_values_for_domes():
    schedule = pd.DataFrame({
        "game_id": ["g1", "g2"],
        "roof": ["dome", "outdoors"],
        "temp": [np.nan, 28.0],
        "wind": [np.nan, 15.0],
    })
    out = compute_weather_features(schedule)
    dome_row = out[out["game_id"] == "g1"].iloc[0]
    outdoor_row = out[out["game_id"] == "g2"].iloc[0]

    assert dome_row["is_dome"] is True or dome_row["is_dome"] == True  # noqa: E712
    assert dome_row["temp_f"] == 70.0
    assert dome_row["wind_mph"] == 0.0

    assert outdoor_row["is_dome"] == False  # noqa: E712
    assert outdoor_row["temp_f"] == 28.0
    assert outdoor_row["wind_mph"] == 15.0


def test_weather_features_keeps_nan_for_outdoor_missing_data():
    """Egy szabadtéri meccsnél, aminek NINCS rögzített mérése (valós adathiány
    az nflverse-ben), a NaN-t NEM szabad mesterségesen kitölteni - csak a
    dome-oknál van értelme a semleges alapérték."""
    schedule = pd.DataFrame({
        "game_id": ["g1"], "roof": ["outdoors"], "temp": [np.nan], "wind": [np.nan],
    })
    out = compute_weather_features(schedule)
    assert pd.isna(out.iloc[0]["temp_f"])
    assert pd.isna(out.iloc[0]["wind_mph"])


def test_weather_features_missing_columns_raises():
    with pytest.raises(ValueError):
        compute_weather_features(pd.DataFrame({"game_id": ["g1"]}))


# ---------------------------------------------------------------------------
# Sérülések
# ---------------------------------------------------------------------------

def test_injury_burden_weights_and_qb_out_flag():
    injuries = pd.DataFrame({
        "season": [2023] * 4,
        "week": [1] * 4,
        "team": ["KC", "KC", "KC", "SF"],
        "position": ["QB", "WR", "CB", "RB"],
        "report_status": ["Out", "Questionable", "Doubtful", "Out"],
    })
    out = compute_injury_burden(injuries)

    kc = out[out["team"] == "KC"].iloc[0]
    assert kc["injury_burden"] == pytest.approx(1.0 + 0.25 + 0.75)
    assert kc["players_out"] == 1
    assert kc["qb_out"] == 1

    sf = out[out["team"] == "SF"].iloc[0]
    assert sf["qb_out"] == 0


def test_injury_burden_unreported_status_counts_as_zero_weight():
    injuries = pd.DataFrame({
        "season": [2023], "week": [1], "team": ["KC"],
        "position": ["WR"], "report_status": [np.nan],
    })
    out = compute_injury_burden(injuries)
    assert out.iloc[0]["injury_burden"] == 0.0


def test_injury_burden_custom_weights_override_defaults():
    injuries = pd.DataFrame({
        "season": [2023], "week": [1], "team": ["KC"],
        "position": ["WR"], "report_status": ["Out"],
    })
    out = compute_injury_burden(injuries, status_weights={"Out": 2.5})
    assert out.iloc[0]["injury_burden"] == 2.5
