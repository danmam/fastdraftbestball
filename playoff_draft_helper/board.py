# playoff_draft_helper/board.py
import pandas as pd
import numpy as np

from .bracket import conference_of
from .scoring import tefp, ceiling_if_sb, ceiling_with_eff_games

BOOSTERS = [2.0, 1.75, 1.5, 1.25, 1.0, 1.0]

def first_team_lock_by_conference(drafted_players_in_order: list[str], player_to_team: dict[str, str]):
    lock = {"NFC": None, "AFC": None}
    for name in drafted_players_in_order:
        t = player_to_team.get(name)
        if not t:
            continue
        conf = conference_of(t)
        if lock[conf] is None:
            lock[conf] = t
    return lock

def compute_board(
    players_df: pd.DataFrame,
    win_odds_df: pd.DataFrame,
    adp_df: pd.DataFrame,
    cond_by_champ_nfc: dict,
    exp_if_not_champ_nfc: dict,
    cond_by_champ_afc: dict,
    exp_if_not_champ_afc: dict,
    drafted_players_in_order: list[str],
    lock_override_nfc: str | None = None,
    lock_override_afc: str | None = None,
) -> pd.DataFrame:
    
    win_by_team = win_odds_df.set_index("Team")
    player_to_team = dict(zip(players_df["Player"], players_df["Team"]))

    auto_locks = first_team_lock_by_conference(drafted_players_in_order, player_to_team)
    lock_nfc = lock_override_nfc if lock_override_nfc else auto_locks["NFC"]
    lock_afc = lock_override_afc if lock_override_afc else auto_locks["AFC"]

    return out.sort_values("DraftPool_EffectiveCeiling", ascending=False), meta

def eff_games(team: str) -> float:
    conf = conference_of(team)

    # ---------- NFC ----------
    if conf == "NFC":
        # LOCKED
        if lock_nfc:
            if team == lock_nfc:
                # Expected games if the locked team does NOT win conference
                return float(exp_if_not_champ_nfc.get(team, np.nan))
            # Expected games for other teams given locked champ
            return float(cond_by_champ_nfc.get(lock_nfc, {}).get(team, np.nan))

        # NOT LOCKED YET → conditional on team winning conference
        return float(cond_by_champ_nfc.get(team, {}).get(team, np.nan))

    # ---------- AFC ----------
    if conf == "AFC":
        if lock_afc:
            if team == lock_afc:
                return float(exp_if_not_champ_afc.get(team, np.nan))
            return float(cond_by_champ_afc.get(lock_afc, {}).get(team, np.nan))

        return float(cond_by_champ_afc.get(team, {}).get(team, np.nan))

    return np.nan

    out = players_df.copy()
    out["Conference"] = out["Team"].apply(conference_of)

    out["TEFP"] = out.apply(lambda r: tefp(r, win_by_team.loc[r["Team"]]), axis=1)
    out["Ceiling_if_SB"] = out.apply(lambda r: ceiling_if_sb(r, win_by_team.loc[r["Team"]]), axis=1)

    out["EffGames_if_NOT_ConfChamp"] = out["Team"].map(lambda t: eff_games(t)).astype(float)
    out["Ceiling_if_NOT_ConfChamp"] = out.apply(
        lambda r: ceiling_with_eff_games(r, win_by_team.loc[r["Team"]], r["EffGames_if_NOT_ConfChamp"]),
        axis=1
    )

    # The number you draft/rank by (draft pool adjusted):
    # - If team is in the locked conference and not the lock => use NOT-champ ceiling.
    # - If team is the lock, you still want to see SB ceiling prominently.
    def draft_pool_ceiling(r) -> float:
        if r["Conference"] == "NFC" and lock_nfc and r["Team"] != lock_nfc:
            return r["Ceiling_if_NOT_ConfChamp"]
        if r["Conference"] == "AFC" and lock_afc and r["Team"] != lock_afc:
            return r["Ceiling_if_NOT_ConfChamp"]
        # If the team is locked (or no lock), default to SB ceiling (best-case) for that team.
        return r["Ceiling_if_SB"]

    out["DraftPool_EffectiveCeiling"] = out.apply(draft_pool_ceiling, axis=1)

    for m in BOOSTERS:
        out[f"Booster_{m}x"] = out["DraftPool_EffectiveCeiling"] * m

    # ADP merge
    if {"Name", "Rank"}.issubset(set(adp_df.columns)):
        out = out.merge(adp_df[["Name", "Rank"]], left_on="Player", right_on="Name", how="left")
        out.rename(columns={"Rank": "ADP_Rank"}, inplace=True)
    else:
        out["ADP_Rank"] = np.nan

    out["CeilingRank"] = out["DraftPool_EffectiveCeiling"].rank(ascending=False, method="min")
    out["Value_Gap"] = out["ADP_Rank"] - out["CeilingRank"]

    drafted_set = set(drafted_players_in_order)
    out["IsDrafted"] = out["Player"].isin(drafted_set)

    meta = {
        "AutoLock_NFC": auto_locks["NFC"],
        "AutoLock_AFC": auto_locks["AFC"],
        "UsingLock_NFC": lock_nfc,
        "UsingLock_AFC": lock_afc,
    }

    return out.sort_values("DraftPool_EffectiveCeiling", ascending=False), meta
