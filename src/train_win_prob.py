"""Win Probability modell tréningje nflverse play-by-play adatokon (XGBoost)."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_GLOB = str(ROOT / "data" / "play_by_play_*.parquet")
MODEL_PATH = ROOT / "models" / "win_probability_xgb.json"
META_PATH = ROOT / "models" / "win_probability_meta.json"

FEATURES = [
    "qtr",
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "is_home",
]

TEST_SEASON = 2024


def load_data() -> pd.DataFrame:
    con = duckdb.connect()
    query = f"""
        SELECT
            season, game_id, qtr, down, ydstogo, yardline_100,
            score_differential, half_seconds_remaining, game_seconds_remaining,
            posteam_timeouts_remaining, defteam_timeouts_remaining,
            posteam_type, result, play_type
        FROM read_parquet('{DATA_GLOB}')
        WHERE posteam IS NOT NULL
          AND score_differential IS NOT NULL
          AND result IS NOT NULL
          AND result != 0
    """
    df = con.execute(query).df()
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["down"] = df["down"].fillna(0)
    df["is_home"] = (df["posteam_type"] == "home").astype(int)
    home_win = (df["result"] > 0).astype(int)
    df["label"] = np.where(df["is_home"] == 1, home_win, 1 - home_win)
    df = df.dropna(subset=FEATURES)
    return df


def main() -> None:
    print("Adatok betöltése (data/play_by_play_*.parquet)...")
    df = load_data()
    df = prepare(df)
    print(f"Betöltött sorok: {len(df):,}")

    train_df = df[df["season"] != TEST_SEASON]
    test_df = df[df["season"] == TEST_SEASON]
    print(f"Train: {len(train_df):,} sor | Test ({TEST_SEASON}): {len(test_df):,} sor")

    X_train, y_train = train_df[FEATURES], train_df["label"]
    X_test, y_test = test_df[FEATURES], test_df["label"]

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
    )
    print("Modell tréningje...")
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    ll = log_loss(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test log loss: {ll:.4f}")
    print(f"Test Brier score: {brier:.4f}")

    baseline_wp = test_df["score_differential"].notna()
    print(f"(Referencia: nflverse saját 'wp' oszlopa a data fájlokban elérhető validáláshoz)")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save_model(str(MODEL_PATH))
    with open(META_PATH, "w") as f:
        json.dump(
            {
                "features": FEATURES,
                "test_season": TEST_SEASON,
                "metrics": {"accuracy": acc, "log_loss": ll, "brier_score": brier},
                "train_rows": len(train_df),
                "test_rows": len(test_df),
            },
            f,
            indent=2,
        )
    print(f"Modell elmentve: {MODEL_PATH}")
    print(f"Metaadatok elmentve: {META_PATH}")


if __name__ == "__main__":
    main()
