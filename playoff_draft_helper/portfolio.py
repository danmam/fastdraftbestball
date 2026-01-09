from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from playoff_draft_helper.sim import build_round_probs, simulate_lineup_many

BOOSTERS = [2.0, 1.75, 1.5, 1.25, 1.0, 1.0]


# -----------------------------
# Helpers: ownership parsing
# -----------------------------
def _parse_ownership_to_fraction(x) -> float:
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float)):
        return float(x) if float(x) <= 1 else float(x) / 100.0
    s = str(x).strip()
    if not s:
        return 0.0
    s = s.replace("%", "").strip()
    try:
        val = float(s)
        return val / 100.0 if val > 1 else val
    except Exception:
        return 0.0


# -----------------------------
# Team probability extraction
# -----------------------------
def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols:
            return cols[key]
    return None


def build_team_round_probs(win_odds_df: pd.DataFrame, bye_teams: set[str]) -> pd.DataFrame:
    df = win_odds_df.copy()

    if "Team" not in df.columns:
        raise ValueError("win_odds_df must contain a 'Team' column (team abbreviations).")

    col_div = _find_col(df, ["P_make_div", "Make Div", "MakeDiv", "Div", "Divisional", "Reach Div"])
    col_conf = _find_col(df, ["P_make_conf", "Make Conf", "MakeConf", "Conf", "Conference", "Reach Conf"])
    col_sb = _find_col(df, ["P_make_sb", "Make SB", "MakeSB", "Reach SB", "Super Bowl", "SuperBowl", "SB"])

    missing = [
        name
        for name, col in [
            ("P_make_div", col_div),
            ("P_make_conf", col_conf),
            ("P_make_sb", col_sb),
        ]
        if col is None
    ]
    if missing:
        raise ValueError(
            "win_odds_df is missing required team reach-probability columns: "
            + ", ".join(missing)
            + ". Add columns named P_make_div, P_make_conf, P_make_sb."
        )

    out = pd.DataFrame(
        {
            "Team": df["Team"].astype(str),
            "P_make_div": pd.to_numeric(df[col_div], errors="coerce").fillna(0.0),
            "P_make_conf": pd.to_numeric(df[col_conf], errors="coerce").fillna(0.0),
            "P_make_sb": pd.to_numeric(df[col_sb], errors="coerce").fillna(0.0),
        }
    )

    out["P_WC_play"] = out["Team"].apply(lambda t: 0.0 if t in bye_teams else 1.0)

    for c in ["P_make_div", "P_make_conf", "P_make_sb", "P_WC_play"]:
        out[c] = out[c].clip(0.0, 1.0)

    return out


# -----------------------------
# Player features for fast scoring
# -----------------------------
def build_player_pool(
    players_df: pd.DataFrame,
    adp_df: pd.DataFrame,
    team_round_probs: pd.DataFrame,
    bye_teams: set[str],
) -> pd.DataFrame:
    pool = players_df.copy()

    if "Player" not in pool.columns or "Team" not in pool.columns:
        raise ValueError("players_df must contain 'Player' and 'Team'.")

    own = adp_df.copy()
    if "Name" not in own.columns:
        raise ValueError("adp_df must contain a 'Name' column to merge ownership.")
    if "Ownership %" not in own.columns:
        raise ValueError("adp_df must contain an 'Ownership %' column.")

    own["OwnershipFrac"] = own["Ownership %"].apply(_parse_ownership_to_fraction)
    own = own[["Name", "OwnershipFrac"]].drop_duplicates()

    pool = pool.merge(own, left_on="Player", right_on="Name", how="left")
    pool.drop(columns=["Name"], inplace=True, errors="ignore")
    pool["OwnershipFrac"] = pool["OwnershipFrac"].fillna(0.0).clip(0.0, 1.0)

    pool["IsWildCardTeam"] = ~pool["Team"].isin(bye_teams)

    pool = pool.merge(team_round_probs, on="Team", how="left")
    for c in ["P_WC_play", "P_make_div", "P_make_conf", "P_make_sb"]:
        pool[c] = pool[c].fillna(0.0).clip(0.0, 1.0)

    if "TEFP" in pool.columns:
        pool["BasePerGame"] = pd.to_numeric(pool["TEFP"], errors="coerce").fillna(0.0)
    else:
        numeric_cols = [c for c in pool.columns if c.lower() in ("proj", "projection", "points", "fpts")]
        if not numeric_cols:
            raise ValueError("players_df must have TEFP or a projection column to compute BasePerGame.")
        pool["BasePerGame"] = pd.to_numeric(pool[numeric_cols[0]], errors="coerce").fillna(0.0)

    pool["FastExpectedTotal"] = pool["BasePerGame"] * (
        pool["P_WC_play"] + pool["P_make_div"] + pool["P_make_conf"] + pool["P_make_sb"]
    )

    if "DraftPool_EffectiveCeiling" in pool.columns:
        pool["CeilingProxy"] = pd.to_numeric(pool["DraftPool_EffectiveCeiling"], errors="coerce").fillna(0.0)
    elif "Ceiling_if_SB" in pool.columns:
        pool["CeilingProxy"] = pd.to_numeric(pool["Ceiling_if_SB"], errors="coerce").fillna(0.0)
    else:
        pool["CeilingProxy"] = pool["FastExpectedTotal"]

    pool["FastPlayerValue"] = 0.75 * pool["CeilingProxy"] + 0.25 * pool["FastExpectedTotal"]

    pool = (
        pool.sort_values("FastPlayerValue", ascending=False)
        .drop_duplicates(subset=["Player"], keep="first")
        .copy()
    )

    return pool


# -----------------------------
# Booster assignment & best-ball top-4 score
# -----------------------------
def assign_boosters_greedy_bestball(players: pd.DataFrame) -> pd.DataFrame:
    df = players.sort_values("FastPlayerValue", ascending=False).copy()
    df["Booster"] = BOOSTERS[: len(df)]
    return df


def score_lineup_fast(lineup_df: pd.DataFrame, k_dup: float = 900.0, eps_own: float = 0.001) -> dict:
    df = assign_boosters_greedy_bestball(lineup_df)
    df["BoostedValue"] = df["FastPlayerValue"] * df["Booster"]

    top4 = df.sort_values("BoostedValue", ascending=False).head(4)
    score_fast = float(top4["BoostedValue"].sum())

    own = df["OwnershipFrac"].clip(lower=eps_own).astype(float).values
    dup_pressure = float(np.prod(own))
    split_factor = float(1.0 / (1.0 + k_dup * dup_pressure))
    ew_fast = score_fast * split_factor

    return {
        "ScoreFast": score_fast,
        "DupPressure": dup_pressure,
        "SplitFactor": split_factor,
        "EWFast": ew_fast,
        "NumWildCard": int(df["IsWildCardTeam"].sum()),
        "NumTeams": int(df["Team"].nunique()),
    }


# -----------------------------
# Candidate generation (STRICT >= min_wc_players, no repair)
# -----------------------------
def generate_candidate_lineups(
    pool: pd.DataFrame,
    n_candidates: int,
    rng: np.random.Generator,
    min_wc_players: int = 3,
    max_players_per_team: int = 4,
    top_team_pool_size: int = 10,
) -> list[list[str]]:
    if "Player" not in pool.columns:
        raise ValueError("pool must contain Player")
    if "Team" not in pool.columns:
        raise ValueError("pool must contain Team")
    if "P_make_sb" not in pool.columns:
        raise ValueError("pool must contain P_make_sb (via team_round_probs merge).")
    if "IsWildCardTeam" not in pool.columns:
        raise ValueError("pool must contain IsWildCardTeam (via build_player_pool).")

    pool = (
        pool.sort_values("FastPlayerValue", ascending=False)
        .drop_duplicates(subset=["Player"], keep="first")
        .copy()
    )

    team_sb = (
        pool.groupby("Team")["P_make_sb"]
        .max()
        .sort_values(ascending=False)
        .head(top_team_pool_size)
    )
    top_teams = team_sb.index.tolist()
    if not top_teams:
        top_teams = pool["Team"].dropna().unique().tolist()

    weights = team_sb.values
    if weights.sum() <= 0:
        weights = np.ones(len(top_teams))
    weights = weights / weights.sum()

    by_team = {t: pool.loc[pool["Team"] == t].copy() for t in pool["Team"].unique()}
    is_wc = pool.set_index("Player")["IsWildCardTeam"]

    def sample_players_from_team(team: str, k: int) -> list[str]:
        df = by_team.get(team)
        if df is None or df.empty:
            return []
        w = df["FastPlayerValue"].clip(lower=0.0).values
        if w.sum() <= 0:
            w = np.ones(len(df))
        w = w / w.sum()
        picks = rng.choice(df["Player"].values, size=k, replace=False, p=w)
        return list(picks)

    shapes = [[3, 3], [3, 2, 1], [4, 2]]
    shape_weights = np.array([0.35, 0.40, 0.25])

    candidates: list[list[str]] = []
    attempts = 0
    max_attempts = n_candidates * 50

    while len(candidates) < n_candidates and attempts < max_attempts:
        attempts += 1

        shape = shapes[int(rng.choice(len(shapes), p=shape_weights))]
        n_teams = len(shape)

        chosen = []
        while len(chosen) < n_teams:
            t = str(rng.choice(top_teams, p=weights))
            if t not in chosen:
                chosen.append(t)

        lineup_players: list[str] = []
        ok = True
        for t, cnt in zip(chosen, shape):
            if cnt > max_players_per_team:
                ok = False
                break
            picks = sample_players_from_team(t, cnt)
            if len(picks) != cnt:
                ok = False
                break
            lineup_players.extend(picks)

        if not ok or len(set(lineup_players)) != 6:
            continue

        num_wc = int(is_wc.loc[lineup_players].sum())
        if num_wc < min_wc_players:
            continue

        candidates.append(lineup_players)

    return candidates


# -----------------------------
# Lineup constraint + leverage helpers
# -----------------------------
def _passes_constraints(idx: pd.DataFrame, players: list[str], min_wc_players: int, min_stack: int) -> bool:
    df = idx.loc[players]
    if int(df["IsWildCardTeam"].sum()) < min_wc_players:
        return False
    if int(df["Team"].value_counts().max()) < min_stack:
        return False
    return True


def _lineup_mean_ownership(idx: pd.DataFrame, players: list[str]) -> float:
    df = idx.loc[players]
    return float(df["OwnershipFrac"].astype(float).mean())


def _sample_k_with_leverage(
    rng: np.random.Generator,
    candidates: list[list[str]],
    idx: pd.DataFrame,
    k: int,
    beta: float = 2.0,
) -> list[list[str]]:
    if len(candidates) <= k:
        return candidates

    owns = np.array([_lineup_mean_ownership(idx, p) for p in candidates], dtype=float)
    leverage = (1.0 - np.clip(owns, 0.0, 1.0)) ** beta
    if leverage.sum() <= 0:
        leverage = np.ones(len(candidates), dtype=float)
    probs = leverage / leverage.sum()

    sel_ix = rng.choice(len(candidates), size=k, replace=False, p=probs)
    return [candidates[i] for i in sel_ix]


# -----------------------------
# Portfolio selection (simulation-driven)
# -----------------------------
def optimize_portfolio_10(
    pool: pd.DataFrame,
    win_odds_df: pd.DataFrame,
    n_candidates: int = 4000,
    k_shortlist: int = 200,
    n_sims: int = 500,
    shortlist_size: int = 400,  # kept for backward compat; not used
    k_dup: float = 900.0,
    overlap_lambda: float = 0.35,
    rng_seed: int = 1,
    min_wc_players: int = 3,
    min_stack: int = 2,
    leverage_beta: float = 2.0,
    feasibility_gate_div_cc_sb: float = 0.03,
    bye_teams: set[str] | None = None,
    rams_team: str = "LAR",
    rams_heavy_threshold: int = 3,
    max_rams_heavy_portfolio: int = 3,
    max_rams_any_portfolio: int = 5,
) -> dict:
    if bye_teams is None:
        bye_teams = {"SEA", "DEN"}

    # Keep development entropy: same rng_seed doesn't mean same lineups every run
    rng = np.random.default_rng(rng_seed + int(np.random.randint(0, 1_000_000)))

    pool = (
        pool.sort_values("FastPlayerValue", ascending=False)
        .drop_duplicates(subset=["Player"], keep="first")
        .copy()
    )
    idx = pool.set_index("Player")

    probs = build_round_probs(win_odds_df)
    print("STEP 1: Generating candidates...")
    # 1) Generate candidates (strict WC constraint)
    candidates = generate_candidate_lineups(
        pool=pool,
        n_candidates=n_candidates,
        rng=rng,
        min_wc_players=min_wc_players,
    )
    print(f"STEP 2: {len(candidates)} candidates generated")
    if not candidates:
        raise ValueError("No candidates generated.")

    # 2) Apply additional structural constraint: min stack (>=3 from one team)
    constrained = [c for c in candidates if _passes_constraints(idx, c, min_wc_players, min_stack)]
    if not constrained:
        raise ValueError("No candidates passed constraints (min_wc_players/min_stack).")

    # 3) Shortlist K=200 using leverage-weighted randomness
    shortlist_lineups = _sample_k_with_leverage(
        rng=rng,
        candidates=constrained,
        idx=idx,
        k=k_shortlist,
        beta=leverage_beta,
    )

    # 4) FastScore (secondary signal / diagnostics only)
    scored_rows = []
    for lineup in shortlist_lineups:
        lineup_df = idx.loc[lineup].reset_index()
        s = score_lineup_fast(lineup_df, k_dup=k_dup)

        team_counts = lineup_df["Team"].value_counts()
        max_stack_count = int(team_counts.max())
        mean_own = float(lineup_df["OwnershipFrac"].astype(float).mean())
        num_rams = int((lineup_df["Team"] == rams_team).sum())

        scored_rows.append(
            {
                "Players": tuple(lineup),
                "EWFast": float(s["EWFast"]),
                "ScoreFast": float(s["ScoreFast"]),
                "SplitFactor": float(s["SplitFactor"]),
                "DupPressure": float(s["DupPressure"]),
                "NumWildCard": int(s["NumWildCard"]),
                "NumTeams": int(s["NumTeams"]),
                "MaxStack": max_stack_count,
                "MeanOwnership": mean_own,
                "NumRams": num_rams,
                "HasAnyRams": num_rams > 0,
                "IsRamsHeavy": num_rams >= rams_heavy_threshold,
            }
        )

    candidates_scored = (
        pd.DataFrame(scored_rows)
        .sort_values("EWFast", ascending=False)
        .reset_index(drop=True)
    )
    print("CANDIDATES_SCORED SIZE:", len(candidates_scored))
    # 5) Week-by-week simulation on shortlist
    sim_rows = [] 
    n_lineups = len(candidates_scored) 
    
    for i, (_, r) in enumerate(candidates_scored.iterrows()): 
        if i % 5 == 0: 
            print(f"[SIM] Starting lineup {i+1}/{n_lineups}")
        players = list(r["Players"])
        lineup_df = idx.loc[players].reset_index()

        sim = simulate_lineup_many(
            lineup_df=lineup_df,
            probs=probs,
            n_sims=n_sims,
            seed=int(rng.integers(1, 1_000_000_000)),
        )
        print(f"[SIM] Finished lineup {i+1}/{n_lineups}")
        sim_rows.append(
            {
                "Players": r["Players"],
                "MaxTotalSim": float(sim["max_total"]),
                "P4EveryWeek": float(sim["p_four_every_week"]),
                "P4DivCCSB": float(sim["p_four_div_cc_sb"]),
                "Q90": float(sim["q90"]),
                "Q95": float(sim["q95"]),
            }
        )

    sim_df = pd.DataFrame(sim_rows)
    scored = candidates_scored.merge(sim_df, on="Players", how="left")

    feasible = scored.loc[scored["P4DivCCSB"] >= feasibility_gate_div_cc_sb].copy()
    if feasible.empty:
        feasible = scored.copy()

    feasible = feasible.sort_values(["MaxTotalSim", "Q95", "EWFast"], ascending=False).reset_index(drop=True)

    # 6) Greedy portfolio selection: simulated ceiling primary, overlap penalty, Rams caps
    selected = []
    selected_sets = []
    rams_any_selected = 0
    rams_heavy_selected = 0

    def _overlap_penalty(players_set: set[str]) -> float:
        return float(sum(len(players_set & s) for s in selected_sets))

    def _try_select(row) -> bool:
        nonlocal rams_any_selected, rams_heavy_selected

        if bool(row["HasAnyRams"]) and rams_any_selected >= max_rams_any_portfolio:
            return False
        if bool(row["IsRamsHeavy"]) and rams_heavy_selected >= max_rams_heavy_portfolio:
            return False

        players_set = set(row["Players"])
        obj = float(row["MaxTotalSim"]) - overlap_lambda * _overlap_penalty(players_set)

        selected.append((obj, row))
        selected_sets.append(players_set)

        if bool(row["HasAnyRams"]):
            rams_any_selected += 1
        if bool(row["IsRamsHeavy"]):
            rams_heavy_selected += 1

        return True

    for _, row in feasible.iterrows():
        if len(selected) >= 10:
            break
        _try_select(row)

    if len(selected) < 10:
        for _, row in scored.sort_values(["MaxTotalSim", "Q95", "EWFast"], ascending=False).iterrows():
            if len(selected) >= 10:
                break
            _try_select(row)

    selected_rows = [r for _, r in sorted(selected, key=lambda x: x[0], reverse=True)[:10]]

    # 7) Build display lineups + exposures
    portfolio_lineups = []
    portfolio_summary_rows = []

    for i, r in enumerate(selected_rows, start=1):
        lineup_players = list(r["Players"])
        lineup_df = idx.loc[lineup_players].reset_index()

        lineup_df = assign_boosters_greedy_bestball(lineup_df)
        lineup_df["BoostedValue"] = lineup_df["FastPlayerValue"].astype(float) * lineup_df["Booster"].astype(float)

        top4 = lineup_df.sort_values("BoostedValue", ascending=False).head(4)

        portfolio_lineups.append(lineup_df)

        portfolio_summary_rows.append(
            {
                "Entry": i,
                "MaxTotalSim": float(r["MaxTotalSim"]),
                "P4DivCCSB": float(r["P4DivCCSB"]),
                "P4EveryWeek": float(r["P4EveryWeek"]),
                "Q95": float(r["Q95"]),
                "EWFast": float(r["EWFast"]),
                "MeanOwnership": float(r["MeanOwnership"]),
                "NumWildCard": int(r["NumWildCard"]),
                "NumTeams": int(r["NumTeams"]),
                "MaxStack": int(r["MaxStack"]),
                "NumRams": int(r["NumRams"]),
                "IsRamsHeavy": bool(r["IsRamsHeavy"]),
                "Top4BoostedSum": float(top4["BoostedValue"].sum()),
                "Players": ", ".join(lineup_players),
            }
        )

    portfolio_summary = pd.DataFrame(portfolio_summary_rows)

    all_players = list(itertools.chain.from_iterable(df["Player"] for df in portfolio_lineups))
    exp_players = pd.Series(all_players).value_counts().rename_axis("Player").reset_index(name="Count")
    exp_players["Exposure"] = exp_players["Count"] / 10.0
    exp_players = exp_players.merge(pool[["Player", "Team", "OwnershipFrac"]], on="Player", how="left")

    all_teams = list(itertools.chain.from_iterable(df["Team"] for df in portfolio_lineups))
    exp_teams = pd.Series(all_teams).value_counts().rename_axis("Team").reset_index(name="Count")
    exp_teams["Exposure"] = exp_teams["Count"] / (10.0 * 6.0)

    return {
        "portfolio_lineups": portfolio_lineups,
        "portfolio_summary": portfolio_summary,
        "exposure_players": exp_players,
        "exposure_teams": exp_teams,
        "candidates_scored": scored,
    }
