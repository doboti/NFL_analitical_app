"""
Unit tesztek az irányító-szintű, időjárás-kondicionált profil modulra.

A hangsúly itt is az adatszivárgás elleni védelmen van (ugyanaz az elv,
mint a csapat-szintű modulnál), plusz a zsugorítás (shrinkage) helyes
matematikáján és az infer_likely_starter dokumentált korlátain.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pregame_feature_engineering import InsufficientHistoryError  # noqa: E402
from qb_profile_engineering import (  # noqa: E402
    assign_weather_bucket,
    build_qb_game_stats,
    compute_qb_weather_profile,
    infer_likely_starter,
)


def test_build_qb_game_stats_groups_by_passer_not_team():
    """Két különböző irányító UGYANANNÁL a csapatnál, UGYANANNÁL a meccsnél
    (csere történt) külön sort kell, hogy kapjon - ez a modul egész létjogosultsága."""
    pbp = pd.DataFrame({
        "game_id": ["g1", "g1", "g1"],
        "season": [2023, 2023, 2023],
        "week": [7, 7, 7],
        "posteam": ["ARI", "ARI", "ARI"],
        "passer_player_id": ["QB_A", "QB_A", "QB_B"],
        "passer_player_name": ["A.Starter", "A.Starter", "B.Backup"],
        "pass_attempt": [1, 1, 1],
        "epa": [0.5, -0.2, 0.1],
        "cpoe": [10.0, -5.0, 2.0],
    })
    out = build_qb_game_stats(pbp)
    assert len(out) == 2
    assert set(out["passer_player_id"]) == {"QB_A", "QB_B"}
    qb_a = out[out["passer_player_id"] == "QB_A"].iloc[0]
    assert qb_a["qb_dropbacks"] == 2
    assert qb_a["qb_epa"] == pytest.approx(0.15)


def test_build_qb_game_stats_ignores_non_pass_and_missing_passer():
    pbp = pd.DataFrame({
        "game_id": ["g1", "g1"],
        "season": [2023, 2023],
        "week": [1, 1],
        "posteam": ["ARI", "ARI"],
        "passer_player_id": [None, "QB_A"],
        "passer_player_name": [None, "A.Starter"],
        "pass_attempt": [0, 1],
        "epa": [0.3, 0.2],
        "cpoe": [None, 5.0],
    })
    out = build_qb_game_stats(pbp)
    assert len(out) == 1
    assert out.iloc[0]["passer_player_id"] == "QB_A"


def test_assign_weather_bucket_priorities():
    weather = pd.DataFrame({
        "game_id": ["dome_game", "windy_game", "cold_game", "mild_game", "unknown_game"],
        "is_dome": [True, False, False, False, False],
        "temp_f": [70.0, 30.0, 30.0, 65.0, None],
        "wind_mph": [0.0, 20.0, 5.0, 5.0, None],
    })
    out = assign_weather_bucket(weather).set_index("game_id")["weather_bucket"]
    assert out["dome_game"] == "dome"
    assert out["windy_game"] == "windy"  # szél elsőbbséget élvez a hideggel szemben
    assert out["cold_game"] == "cold"
    assert out["mild_game"] == "mild"
    assert out["unknown_game"] == "unknown"


# ---------------------------------------------------------------------------
# Zsugorítás (shrinkage) és adatszivárgás
# ---------------------------------------------------------------------------

def _qb_weather_rows(passer_id, passer_name, entries):
    """entries: list of (season, week, weather_bucket, epa, cpoe)"""
    rows = []
    for i, (season, week, bucket, epa, cpoe) in enumerate(entries):
        rows.append({
            "game_id": f"g{i}", "season": season, "week": week,
            "passer_player_id": passer_id, "passer_player_name": passer_name,
            "weather_bucket": bucket, "qb_epa": epa, "qb_cpoe": cpoe, "qb_dropbacks": 30,
        })
    return rows


def test_qb_weather_profile_shrinks_small_sample_toward_overall_mean():
    """Egy irányítónak sok 'mild' meccse van jó formával, és EGYETLEN 'cold'
    meccse rossz formával - a zsugorítás miatt a 'cold' becslésnek KÖZELEBB
    kell lennie az összesített átlaghoz, mint a nyers 1 meccses -2.0 értékhez."""
    entries = [(2023, w, "mild", 0.3, 5.0) for w in range(1, 6)]
    entries.append((2023, 6, "cold", -2.0, -20.0))
    qb_weather = pd.DataFrame(_qb_weather_rows("QB_A", "A.Test", entries))

    profile = compute_qb_weather_profile(qb_weather, as_of_season=2023, as_of_week=7, shrinkage_k=8.0)
    cold_row = profile[(profile["passer_player_id"] == "QB_A") & (profile["weather_bucket"] == "cold")].iloc[0]

    raw_cold_mean = -2.0
    overall_mean = (5 * 0.3 + (-2.0)) / 6
    assert raw_cold_mean < cold_row["qb_weather_epa"] < overall_mean + 0.01
    # jelentősen közelebb kell lennie az overall-hoz, mint a nyers 1-mintás értékhez
    assert abs(cold_row["qb_weather_epa"] - overall_mean) < abs(cold_row["qb_weather_epa"] - raw_cold_mean)


def test_qb_weather_profile_does_not_leak_future_weeks():
    entries = [(2023, 1, "mild", 0.2, 3.0), (2023, 2, "cold", 5.0, 40.0)]  # 2. hét = "jövő" a teszthez
    qb_weather = pd.DataFrame(_qb_weather_rows("QB_A", "A.Test", entries))

    # as_of_week=2 -> csak az 1. hét (mild) látható, a 2. heti (cold) extrém érték NEM.
    profile = compute_qb_weather_profile(qb_weather, as_of_season=2023, as_of_week=2, shrinkage_k=8.0)
    assert "cold" not in profile["weather_bucket"].values


def test_qb_weather_profile_raises_when_no_history():
    qb_weather = pd.DataFrame(_qb_weather_rows("QB_A", "A.Test", [(2023, 1, "mild", 0.2, 3.0)]))
    with pytest.raises(InsufficientHistoryError):
        compute_qb_weather_profile(qb_weather, as_of_season=2023, as_of_week=1)


# ---------------------------------------------------------------------------
# infer_likely_starter
# ---------------------------------------------------------------------------

_QB_GAME_STATS_COLUMNS = ["game_id", "season", "week", "team", "passer_player_id",
                          "passer_player_name", "qb_dropbacks"]


def _qb_game_stats_rows(team, entries):
    """entries: list of (game_id, season, week, passer_id, passer_name, dropbacks)"""
    if not entries:
        return pd.DataFrame(columns=_QB_GAME_STATS_COLUMNS)
    return pd.DataFrame([
        {"game_id": g, "season": s, "week": w, "team": team,
         "passer_player_id": pid, "passer_player_name": pname, "qb_dropbacks": db}
        for g, s, w, pid, pname, db in entries
    ])


def test_infer_likely_starter_healthy_case():
    qb_stats = _qb_game_stats_rows("ARI", [
        ("g1", 2023, 1, "QB_A", "A.Starter", 30),
        ("g2", 2023, 2, "QB_A", "A.Starter", 28),
    ])
    result = infer_likely_starter("ARI", 2023, 3, qb_stats, injuries_df=None)
    assert result["passer_player_id"] == "QB_A"
    assert result["confidence"] == "confirmed_healthy"


def test_infer_likely_starter_falls_back_to_backup_when_starter_out():
    qb_stats = _qb_game_stats_rows("ARI", [
        ("g1", 2023, 1, "QB_B", "B.Backup", 5),   # korábbi mérkőzés, ahol a backup is játszott
        ("g2", 2023, 2, "QB_A", "A.Starter", 30),
    ])
    injuries = pd.DataFrame({
        "season": [2023], "week": [3], "team": ["ARI"],
        "gsis_id": ["QB_A"], "report_status": ["Out"],
    })
    result = infer_likely_starter("ARI", 2023, 3, qb_stats, injuries_df=injuries)
    assert result["passer_player_id"] == "QB_B"
    assert result["confidence"] == "inferred_due_to_injury"


def test_infer_likely_starter_no_history_returns_unknown():
    qb_stats = _qb_game_stats_rows("ARI", [])
    result = infer_likely_starter("ARI", 2023, 1, qb_stats, injuries_df=None)
    assert result["confidence"] == "no_history"


def test_infer_likely_starter_injured_with_no_known_backup():
    qb_stats = _qb_game_stats_rows("ARI", [("g1", 2023, 1, "QB_A", "A.Starter", 30)])
    injuries = pd.DataFrame({
        "season": [2023], "week": [2], "team": ["ARI"],
        "gsis_id": ["QB_A"], "report_status": ["Out"],
    })
    result = infer_likely_starter("ARI", 2023, 2, qb_stats, injuries_df=injuries)
    assert result["confidence"] == "starter_injured_no_known_backup"
