# playoff_draft_helper/scoring.py
import numpy as np

def compute_xgp(team_row) -> float:
    if bool(team_row["Has_WC_Game"]):
        return float(min(
            4.0,
            1 + team_row["P_make_div"]
              + team_row["P_make_conf"]
              + team_row["P_make_sb"]
        ))
    return float(min(
        3.0,
        1 + team_row["P_make_conf"]
          + team_row["P_make_sb"]
    ))

def tefp(player_row, team_row) -> float:
    xgp = compute_xgp(team_row)
    if bool(team_row["Has_WC_Game"]):
        return float(player_row["Wild Card Mean FPTS"] + (xgp - 1) * player_row["Div, Conf, SB Mean FPTS"])
    return float(xgp * player_row["Div, Conf, SB Mean FPTS"])

def ceiling_if_sb(player_row, team_row) -> float:
    max_games = float(team_row["Max_Games"])
    if bool(team_row["Has_WC_Game"]):
        return float(player_row["Wild Card Ceiling FPTS"] + (max_games - 1) * player_row["Div, Conf, SB  Ceiling FPTS"])
    return float(max_games * player_row["Div, Conf, SB  Ceiling FPTS"])

def ceiling_with_eff_games(player_row, team_row, eff_games: float) -> float:
    if np.isnan(eff_games):
        eff_games = float(team_row["Max_Games"])
    if bool(team_row["Has_WC_Game"]):
        return float(player_row["Wild Card Ceiling FPTS"] + (eff_games - 1) * player_row["Div, Conf, SB  Ceiling FPTS"])
    return float(eff_games * player_row["Div, Conf, SB  Ceiling FPTS"])

