print("LOADED sim.py WITH simulate_lineup_many")

# playoff_draft_helper/sim.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .bracket import BYE, WC_MATCHUPS, divisional_pairings, teams_in_conf

@dataclass(frozen=True)
class RoundWinProbs:
    p_wc: dict[str, float]   # WC win prob for WC teams
    p_div: dict[str, float]  # Divisional win prob given they are in Div game
    p_cc: dict[str, float]   # CC win prob given they are in CC game

def build_round_probs(win_odds_df) -> RoundWinProbs:
    # Use standardized column names
    P_Div = dict(zip(win_odds_df["Team"], win_odds_df["P_make_div"]))
    P_Conf = dict(zip(win_odds_df["Team"], win_odds_df["P_make_conf"]))
    P_SB = dict(zip(win_odds_df["Team"], win_odds_df["P_make_sb"]))
    Has_WC = dict(zip(win_odds_df["Team"], win_odds_df["Has_WC_Game"]))

    p_wc, p_div, p_cc = {}, {}, {}

    for team in win_odds_df["Team"].tolist():
        if Has_WC[team]:
            # WC teams: P_make_div is their WC win probability
            p_wc[team] = float(P_Div[team])
            p_div[team] = (
                float(P_Conf[team] / P_Div[team])
                if P_Div[team] and P_Div[team] > 0
                else 0.0
            )
        else:
            # Bye teams: no WC game
            p_wc[team] = np.nan
            p_div[team] = float(P_Conf[team])

        # IMPORTANT: this is probability of MAKING the Super Bowl,
        # not winning it — exactly as you specified
        p_cc[team] = (
            float(P_SB[team] / P_Conf[team])
            if P_Conf[team] and P_Conf[team] > 0
            else 0.0
        )

    return RoundWinProbs(p_wc=p_wc, p_div=p_div, p_cc=p_cc)

def _play(a: str, b: str, p_a: float, rng: np.random.Generator) -> str:
    return a if rng.random() < p_a else b

def simulate_conference_once(conf: str, probs: RoundWinProbs, rng: np.random.Generator):
    teams = list(teams_in_conf(conf))
    bye = BYE[conf]
    games_played = {t: 0 for t in teams}

    # Wild card
    wc_winners = []
    for home, away in WC_MATCHUPS[conf]:
        games_played[home] += 1
        games_played[away] += 1
        wc_winners.append(_play(home, away, probs.p_wc[home], rng))

    # Divisional (reseeding)
    div_pairs = divisional_pairings(conf, wc_winners)
    div_winners = []
    for a, b in div_pairs:
        games_played[a] += 1
        games_played[b] += 1
        div_winners.append(_play(a, b, probs.p_div[a], rng))

    # Conference championship
    a, b = div_winners
    games_played[a] += 1
    games_played[b] += 1
    champ = _play(a, b, probs.p_cc[a], rng)

    return champ, games_played

def simulate_conditionals(conf: str, probs: RoundWinProbs, n_sims: int = 200000, seed: int = 1):
    """
    Returns:
      cond_by_champ[champ][team] = E[games(team) | champ]
      exp_if_not_champ[team]     = E[games(team) | champ != team]
      champ_counts[champ]        = count of sims where champ occurred
    """
    rng = np.random.default_rng(seed)
    teams = list(teams_in_conf(conf))

    sums_by_champ = {c: {t: 0.0 for t in teams} for c in teams}
    champ_counts = {c: 0 for c in teams}

    sums_not_champ = {t: 0.0 for t in teams}
    cnt_not_champ = {t: 0 for t in teams}

    for _ in range(n_sims):
        champ, gp = simulate_conference_once(conf, probs, rng)

        champ_counts[champ] += 1
        row = sums_by_champ[champ]
        for t in teams:
            row[t] += gp[t]
            if t != champ:
                sums_not_champ[t] += gp[t]
                cnt_not_champ[t] += 1

    cond_by_champ = {}
    for champ in teams:
        denom = champ_counts[champ]
        if denom > 0:
            cond_by_champ[champ] = {t: sums_by_champ[champ][t] / denom for t in teams}

    exp_if_not_champ = {}
    for t in teams:
        denom = cnt_not_champ[t]
        exp_if_not_champ[t] = (sums_not_champ[t] / denom) if denom > 0 else np.nan

    return cond_by_champ, exp_if_not_champ, champ_counts
def simulate_lineup_many(
    lineup_df,
    probs: RoundWinProbs,
    n_sims: int = 50000,
    seed: int = 1,
    value_col: str = "FastPlayerValue",
    team_col: str = "Team",
):
    rng = np.random.default_rng(seed)

    totals = np.zeros(n_sims, dtype=float)
    ok_all = 0
    ok_late = 0

    for i in range(n_sims):
        sim_out = simulate_nfl_once(probs, rng)
        scored = score_lineup_one_universe(
            lineup_df=lineup_df,
            sim_out=sim_out,
            value_col=value_col,
            team_col=team_col,
        )
        totals[i] = scored["total_score"]
        if scored["has_four_every_week"]:
            ok_all += 1
        if scored["has_four_div_cc_sb"]:
            ok_late += 1

    return {
        "max_total": float(np.max(totals)) if n_sims else 0.0,
        "p_four_every_week": float(ok_all / n_sims) if n_sims else 0.0,
        "p_four_div_cc_sb": float(ok_late / n_sims) if n_sims else 0.0,
        "q50": float(np.quantile(totals, 0.50)) if n_sims else 0.0,
        "q80": float(np.quantile(totals, 0.80)) if n_sims else 0.0,
        "q90": float(np.quantile(totals, 0.90)) if n_sims else 0.0,
        "q95": float(np.quantile(totals, 0.95)) if n_sims else 0.0,
    }


