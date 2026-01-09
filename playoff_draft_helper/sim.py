# playoff_draft_helper/sim.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .bracket import BYE, WC_MATCHUPS, divisional_pairings, teams_in_conf


# ============================================================
# Round win probabilities
# ============================================================

@dataclass(frozen=True)
class RoundWinProbs:
    p_wc: dict[str, float]   # WC win prob for WC teams
    p_div: dict[str, float]  # Divisional win prob given they play in Div
    p_cc: dict[str, float]   # Conference Championship win prob given they play


def build_round_probs(win_odds_df) -> RoundWinProbs:
    """
    Uses reach probabilities:
      - P_make_div
      - P_make_conf
      - P_make_sb
      - Has_WC_Game

    IMPORTANT:
      - P_make_sb = probability of MAKING the Super Bowl (not winning it)
      - Super Bowl is treated as an extra game for both champs
    """
    P_Div = dict(zip(win_odds_df["Team"], win_odds_df["P_make_div"]))
    P_Conf = dict(zip(win_odds_df["Team"], win_odds_df["P_make_conf"]))
    P_SB = dict(zip(win_odds_df["Team"], win_odds_df["P_make_sb"]))
    Has_WC = dict(zip(win_odds_df["Team"], win_odds_df["Has_WC_Game"]))

    p_wc, p_div, p_cc = {}, {}, {}

    for team in win_odds_df["Team"]:
        if Has_WC[team]:
            p_wc[team] = float(P_Div[team])
            p_div[team] = float(P_Conf[team] / P_Div[team]) if P_Div[team] > 0 else 0.0
        else:
            p_wc[team] = np.nan
            p_div[team] = float(P_Conf[team])

        p_cc[team] = float(P_SB[team] / P_Conf[team]) if P_Conf[team] > 0 else 0.0

    return RoundWinProbs(p_wc=p_wc, p_div=p_div, p_cc=p_cc)


# ============================================================
# Core simulation helpers
# ============================================================

def _play(a: str, b: str, p_a: float, rng: np.random.Generator) -> str:
    return a if rng.random() < p_a else b


def simulate_conference_once(conf: str, probs: RoundWinProbs, rng: np.random.Generator):
    teams = list(teams_in_conf(conf))
    games_played = {t: 0 for t in teams}
    weeks = {"WC": [], "DIV": [], "CC": []}

    # Wild Card
    wc_winners = []
    for home, away in WC_MATCHUPS[conf]:
        weeks["WC"].append((home, away))
        games_played[home] += 1
        games_played[away] += 1
        wc_winners.append(_play(home, away, probs.p_wc[home], rng))

    # Divisional (reseeding)
    div_pairs = divisional_pairings(conf, wc_winners)
    div_winners = []
    for a, b in div_pairs:
        weeks["DIV"].append((a, b))
        games_played[a] += 1
        games_played[b] += 1
        div_winners.append(_play(a, b, probs.p_div[a], rng))

    # Conference Championship
    a, b = div_winners
    weeks["CC"].append((a, b))
    games_played[a] += 1
    games_played[b] += 1
    champ = _play(a, b, probs.p_cc[a], rng)

    return champ, games_played, weeks


def simulate_nfl_once(probs: RoundWinProbs, rng: np.random.Generator):
    nfc_champ, nfc_gp, nfc_weeks = simulate_conference_once("NFC", probs, rng)
    afc_champ, afc_gp, afc_weeks = simulate_conference_once("AFC", probs, rng)

    games_played = {}
    games_played.update(nfc_gp)
    games_played.update(afc_gp)

    weeks = {
        "WC": nfc_weeks["WC"] + afc_weeks["WC"],
        "DIV": nfc_weeks["DIV"] + afc_weeks["DIV"],
        "CC": nfc_weeks["CC"] + afc_weeks["CC"],
        "SB": [(nfc_champ, afc_champ)],
    }

    games_played[nfc_champ] += 1
    games_played[afc_champ] += 1

    return {"weeks": weeks, "games_played": games_played}


# ============================================================
# Lineup scoring (lineup‑locked boosters)
# ============================================================

BOOSTERS = [2.0, 1.75, 1.5, 1.25]


def assign_lineup_locked_boosters(lineup_df, value_col="FastPlayerValue"):
    df = lineup_df.copy()
    df["Booster"] = 1.0
    order = df.sort_values(value_col, ascending=False).index.tolist()
    for i, idx in enumerate(order[:4]):
        df.loc[idx, "Booster"] = BOOSTERS[i]
    return df


def score_lineup_one_universe(lineup_df, sim_out, value_col="FastPlayerValue"):
    df = assign_lineup_locked_boosters(lineup_df, value_col)
    df["BoostedValue"] = df[value_col] * df["Booster"]

    total = 0.0
    alive_counts = {}
    has_four_every_week = True

    for week, games in sim_out["weeks"].items():
        playing = {t for g in games for t in g}
        elig = df[df["Team"].isin(playing)]
        alive_counts[week] = len(elig)
        if len(elig) < 4:
            has_four_every_week = False
        total += elig.sort_values("BoostedValue", ascending=False).head(4)["BoostedValue"].sum()

    has_four_div_cc_sb = (
        alive_counts.get("DIV", 0) >= 4
        and alive_counts.get("CC", 0) >= 4
        and alive_counts.get("SB", 0) >= 4
    )

    return {
        "total_score": float(total),
        "has_four_every_week": has_four_every_week,
        "has_four_div_cc_sb": has_four_div_cc_sb,
    }


# ============================================================
# Public API: simulate many universes
# ============================================================

def simulate_lineup_many(
    lineup_df,
    probs: RoundWinProbs,
    n_sims: int = 5000,
    seed: int = 1,
):
    rng = np.random.default_rng(seed)

    totals = np.zeros(n_sims)
    ok_all = 0
    ok_late = 0

    for i in range(n_sims):
        sim = simulate_nfl_once(probs, rng)
        scored = score_lineup_one_universe(lineup_df, sim)
        totals[i] = scored["total_score"]
        if scored["has_four_every_week"]:
            ok_all += 1
        if scored["has_four_div_cc_sb"]:
            ok_late += 1

    return {
        "max_total": float(totals.max()),
        "p_four_every_week": ok_all / n_sims,
        "p_four_div_cc_sb": ok_late / n_sims,
        "q90": float(np.quantile(totals, 0.90)),
        "q95": float(np.quantile(totals, 0.95)),
    }
