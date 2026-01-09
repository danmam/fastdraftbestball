# playoff_draft_helper/data.py
import pandas as pd
import numpy as np

ODDS_COLS = [
    "Pick to Win Popularity (Wild Card)",
    "P_make_div",
    "P_make_conf",
    "P_make_sb",
]

def _pct_to_decimal(s: pd.Series) -> pd.Series:
    if s.dtype != "object":
        return s.astype(float)
    out = (
        s.astype(str).str.strip()
        .replace({"N/A": np.nan, "NA/": np.nan, "nan": np.nan})
        .str.rstrip("%")
    )
    return pd.to_numeric(out, errors="coerce") / 100.0

def load_data(players_csv, win_odds_csv, adp_csv):
    players = pd.read_csv(players_csv)
    win_odds = pd.read_csv(win_odds_csv)
    adp = pd.read_csv(adp_csv)

    # minimal hygiene: trim strings, do not alter projection numbers
    for col in ["Team", "Position", "Role", "Player"]:
        if col in players.columns:
            players[col] = players[col].astype(str).str.strip()

    win_odds["Team"] = win_odds["Team"].astype(str).str.strip()

    for col in ODDS_COLS:
        win_odds[col] = _pct_to_decimal(win_odds[col])

    win_odds["Has_WC_Game"] = ~win_odds["Pick to Win Popularity (Wild Card)"].isna()
    win_odds["Max_Games"] = np.where(win_odds["Has_WC_Game"], 4.0, 3.0)

    return players, win_odds, adp



