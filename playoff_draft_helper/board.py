import numpy as np
import pandas as pd

from playoff_draft_helper.bracket import conference_of
from playoff_draft_helper.scoring import (
    tefp,
    ceiling_if_sb,
    ceiling_with_eff_games,
)

BOOSTERS = [2.0, 1.75, 1.5, 1.25, 1.0, 1.0]


def compute_board(
    players_df: pd.DataFrame,
    win_odds_df: pd.DataFrame,
    adp_df: pd.DataFrame,
    cond_by_champ_nfc: dict,
    exp_if_not_champ_nfc: dict,
    cond_by_champ_afc: dict,
    exp_if_not_champ_afc: dict,
    drafted_players_in_order: list,
    lock_override_nfc: str | None = None,
    lock_override_afc: str | None = None,
):
    """
    Returns:
        board_df : DataFrame sorted by DraftPool_EffectiveCeiling
        meta     : dict describing auto-locks and active locks
    """

    # -----------------------------
    # Lookups
    # -----------------------------
    win_by_team = win_odds_df.set_index("Team")
    player_to_team = dict(zip(players_df["Player"], players_df["Team"]))

    # -----------------------------
    # Auto-locks (first drafted team per conference)
    # -----------------------------
    auto_lock = {"NFC": None, "AFC": None}

    for player in drafted_players_in_order:
        team = player_to_team.get(player)
        if team is None:
            continue
        conf = conference_of(team)
        if auto_lock[conf] is None:
            auto_lock[conf] = team

    lock_nfc = lock_override_nfc if lock_override_nfc else auto_lock["NFC"]
    lock_afc = lock_override_afc if lock_override_afc else auto_lock["AFC"]

    # -----------------------------
    # Effective games logic
    # -----------------------------
    def eff_games(team: str) -> float:
        conf = conference_of(team)

        # ---------- NFC ----------
        if conf == "NFC":
            if lock_nfc:
                if team == lock_nfc:
                    return exp_if_not_champ_nfc.get(team, np.nan)
                return cond_by_champ_nfc.get(lock_nfc, {}).get(team, np.nan)
            return cond_by_champ_nfc.get(team, {}).get(team, np.nan)

        # ---------- AFC ----------
        if conf == "AFC":
            if lock_afc:
                if team == lock_afc:
                    return exp_if_not_champ_afc.get(team, np.nan)
                return cond_by_champ_afc.get(lock_afc, {}).get(team, np.nan)
            return cond_by_champ_afc.get(team, {}).get(team, np.nan)

        return np.nan

    # =============================
    # BUILD BOARD
    # =============================
    out = players_df.copy()

    out["Conference"] = out["Team"].apply(conference_of)

    out["TEFP"] = out.apply(
        lambda r: tefp(r, win_by_team.loc[r["Team"]]),
        axis=1,
    )

    out["Ceiling_if_SB"] = out.apply(
        lambda r: ceiling_if_sb(r, win_by_team.loc[r["Team"]]),
        axis=1,
    )

    out["EffGames_if_NOT_ConfChamp"] = out["Team"].map(eff_games)

    out["Ceiling_if_NOT_ConfChamp"] = out.apply(
        lambda r: ceiling_with_eff_games(
            r,
            win_by_team.loc[r["Team"]],
            r["EffGames_if_NOT_ConfChamp"],
        ),
        axis=1,
    )

    # -----------------------------
    # Draft-pool effective ceiling
    # -----------------------------
    def draft_pool_ceiling(row):
        if row["Conference"] == "NFC" and lock_nfc and row["Team"] != lock_nfc:
            return row["Ceiling_if_NOT_ConfChamp"]
        if row["Conference"] == "AFC" and lock_afc and row["Team"] != lock_afc:
            return row["Ceiling_if_NOT_ConfChamp"]
        return row["Ceiling_if_SB"]

    out["DraftPool_EffectiveCeiling"] = out.apply(draft_pool_ceiling, axis=1)

    # -----------------------------
    # Boosters
    # -----------------------------
    for mult in BOOSTERS:
        out[f"Booster_{mult}x"] = out["DraftPool_EffectiveCeiling"] * mult

    # -----------------------------
    # ADP merge
    # -----------------------------
    if {"Name", "Rank"}.issubset(adp_df.columns):
        out = out.merge(
            adp_df[["Name", "Rank"]],
            left_on="Player",
            right_on="Name",
            how="left",
        )
        out.rename(columns={"Rank": "ADP_Rank"}, inplace=True)
    else:
        out["ADP_Rank"] = np.nan

    out["CeilingRank"] = out["DraftPool_EffectiveCeiling"].rank(
        ascending=False,
        method="min",
    )

    out["Value_Gap"] = out["ADP_Rank"] - out["CeilingRank"]

    out["IsDrafted"] = out["Player"].isin(set(drafted_players_in_order))

    # -----------------------------
    # Meta info
    # -----------------------------
    meta = {
        "AutoLock_NFC": auto_lock["NFC"],
        "AutoLock_AFC": auto_lock["AFC"],
        "UsingLock_NFC": lock_nfc,
        "UsingLock_AFC": lock_afc,
    }

    # -----------------------------
    # FINAL RETURN (ALWAYS)
    # -----------------------------
    return out.sort_values(
        "DraftPool_EffectiveCeiling",
        ascending=False,
    ), meta
