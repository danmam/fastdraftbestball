from __future__ import annotations

import itertools
import math
import numpy as np
import pandas as pd

BOOSTERS = [2.0, 1.75, 1.5, 1.25, 1.0, 1.0]


# -----------------------------
# Helpers: ownership parsing
# -----------------------------
def _parse_ownership_to_fraction(x) -> float:
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float)):
        # assume already fraction if <=1 else percent
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
    """
    Returns a team-level table with:
      - P_WC_play: 1 if not bye team else 0
      - P_make_div
      - P_make_conf
      - P_make_sb   (IMPORTANT: make SB, not win SB)
    This tries to locate columns in win_odds_df. If you already have these as columns,
    name them with something like:
      Team, P_make_div, P_make_conf, P_make_sb
    """
    df = win_odds_df.copy()

    if "Team" not in df.columns:
        raise ValueError("win_odds_df must contain a 'Team' column (team abbreviations).")

    # Prefer explicit columns if present
    col_div = _find_col(df, ["P_make_div", "Make Div", "MakeDiv", "Div", "Divisional", "Reach Div"])
    col_conf = _find_col(df, ["P_make_conf", "Make Conf", "MakeConf", "Conf", "Conference", "Reach Conf"])
    col_sb = _find_col(df, ["P_make_sb", "Make SB", "MakeSB", "Reach SB", "Super Bowl", "SuperBowl", "SB"])

    # If your file only has win-by-round, you can extend this later.
    # For now, we require these to exist to avoid silently using P(win SB).
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
            + ". Add columns named P_make_div, P_make_conf, P_make_sb (probability team reaches each round)."
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

    # Sanity clamps
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
    """
    Creates a modeling pool with:
      - OwnershipFrac (from adp file, merged by player name)
      - IsWildCardTeam
      - FastExpectedTotal (week-aware expected points proxy)
    Requires players_df has at least: Player, Team, TEFP (or something we can use), DraftPool_EffectiveCeiling or Ceiling_if_SB.
    """
    pool = players_df.copy()

    if "Player" not in pool.columns or "Team" not in pool.columns:
        raise ValueError("players_df must contain 'Player' and 'Team'.")

    # Merge ownership by Name->Player
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

    # Wild card flag
    pool["IsWildCardTeam"] = ~pool["Team"].isin(bye_teams)

    # Team round reach probs
    pool = pool.merge(team_round_probs, on="Team", how="left")
    for c in ["P_WC_play", "P_make_div", "P_make_conf", "P_make_sb"]:
        pool[c] = pool[c].fillna(0.0).clip(0.0, 1.0)

    # Base per-game expectation proxy
    if "TEFP" in pool.columns:
        pool["BasePerGame"] = pd.to_numeric(pool["TEFP"], errors="coerce").fillna(0.0)
    else:
        # fallback if TEFP not present; you can tighten this if your schema differs
        numeric_cols = [c for c in pool.columns if c.lower() in ("proj", "projection", "points", "fpts")]
        if not numeric_cols:
            raise ValueError("players_df must have TEFP or a projection column to compute BasePerGame.")
        pool["BasePerGame"] = pd.to_numeric(pool[numeric_cols[0]], errors="coerce").fillna(0.0)

    # Week-aware expected total points proxy:
    # WC contributes if they play, later rounds contribute by reach probabilities.
    # This is intentionally about MAKING rounds, not winning SB.
    pool["FastExpectedTotal"] = pool["BasePerGame"] * (
        pool["P_WC_play"] + pool["P_make_div"] + pool["P_make_conf"] + pool["P_make_sb"]
    )

    # A ceiling proxy for tail weighting (optional but useful)
    if "DraftPool_EffectiveCeiling" in pool.columns:
        pool["CeilingProxy"] = pd.to_numeric(pool["DraftPool_EffectiveCeiling"], errors="coerce").fillna(0.0)
    elif "Ceiling_if_SB" in pool.columns:
        pool["CeilingProxy"] = pd.to_numeric(pool["Ceiling_if_SB"], errors="coerce").fillna(0.0)
    else:
        pool["CeilingProxy"] = pool["FastExpectedTotal"]

    # Tail-weighted per-player value (fast)
    pool["FastPlayerValue"] = 0.75 * pool["CeilingProxy"] + 0.25 * pool["FastExpectedTotal"]

    # Critical invariant for downstream selection/indexing:
    # We must have a single row per Player to avoid ambiguous .loc returning a Series.
    pool = pool.sort_values("FastPlayerValue", ascending=False).drop_duplicates(subset=["Player"], keep="first")

    return pool


# -----------------------------
# Booster assignment & best-ball top-4 score
# -----------------------------
def assign_boosters_greedy_bestball(players: pd.DataFrame) -> pd.DataFrame:
    """
    Input: 6-row df for lineup players, must include FastPlayerValue.
    Output: same df with Booster assigned.
    Greedy rule: higher booster to higher FastPlayerValue.
    """
    df = players.sort_values("FastPlayerValue", ascending=False).copy()
    df["Booster"] = BOOSTERS[: len(df)]
    return df


def score_lineup_fast(lineup_df: pd.DataFrame, k_dup: float = 900.0, eps_own: float = 0.001) -> dict:
    """
    lineup_df: 6-row df with columns: Player, Team, FastPlayerValue, OwnershipFrac, IsWildCardTeam
    Returns a dict with:
      - ScoreFast (top4 after boosters)
      - DupPressure
      - SplitFactor
      - EWFast
      - NumWildCard
      - NumTeams
    """
    df = assign_boosters_greedy_bestball(lineup_df)

    df["BoostedValue"] = df["FastPlayerValue"] * df["Booster"]

    # best ball: top 4 AFTER boosters
    top4 = df.sort_values("BoostedValue", ascending=False).head(4)
    score_fast = float(top4["BoostedValue"].sum())

    # duplication pressure (ownership)
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
# Candidate generation
# -----------------------------
def generate_candidate_lineups(
    pool: pd.DataFrame,
    n_candidates: int,
    rng: np.random.Generator,
    min_wc_players: int = 4,
    max_players_per_team: int = 4,
    top_team_pool_size: int = 10,
) -> list[list[str]]:
    """
    Generates candidates without locking teams, but biased toward teams likely to make SB
    while still allowing multi-team builds (3-3, 3-2-1, etc.).

    Structural rules enforced at generation-time:
      - 6 players total
      - >= min_wc_players from WC teams
      - no team exceeds max_players_per_team (prevents 5-1 unless you allow it)
    """
    if "Player" not in pool.columns:
        raise ValueError("pool must contain Player")
    if "Team" not in pool.columns:
        raise ValueError("pool must contain Team")
    if "P_make_sb" not in pool.columns:
        raise ValueError("pool must contain P_make_sb (via team_round_probs merge).")

    # Ensure uniqueness so idx.loc[lineup] always returns exactly 6 rows
    pool = pool.sort_values("FastPlayerValue", ascending=False).drop_duplicates(subset=["Player"], keep="first").copy()

    # candidate primary teams = top by P_make_sb
    team_sb = (
        pool.groupby("Team")["P_make_sb"]
        .max()
        .sort_values(ascending=False)
        .head(top_team_pool_size)
    )
    top_teams = team_sb.index.tolist()
    if not top_teams:
        top_teams = pool["Team"].dropna().unique().tolist()

    # weighted team sampling by P_make_sb (not win SB)
    weights = team_sb.values
    if weights.sum() <= 0:
        weights = np.ones(len(top_teams))
    weights = weights / weights.sum()

    # pre-split players by team
    by_team = {t: pool.loc[pool["Team"] == t].copy() for t in pool["Team"].unique()}
    wc_pool = pool.loc[pool["IsWildCardTeam"]].copy()
    bye_pool = pool.loc[~pool["IsWildCardTeam"]].copy()

    # fast lookup: player -> is wildcard flag (scalar)
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

    def sample_satellite(k: int, exclude_players: set[str]) -> list[str]:
        df = pool.loc[~pool["Player"].isin(exclude_players)]
        if df.empty:
            return []
        w = df["FastPlayerValue"].clip(lower=0.0).values
        if w.sum() <= 0:
            w = np.ones(len(df))
        w = w / w.sum()
        picks = rng.choice(df["Player"].values, size=k, replace=False, p=w)
        return list(picks)

    candidates: list[list[str]] = []

    # shapes: 3-3, 3-2-1, 4-2 (team counts)
    shapes = [
        [3, 3],
        [3, 2, 1],
        [4, 2],
    ]
    shape_weights = np.array([0.35, 0.40, 0.25])

    attempts = 0
    while len(candidates) < n_candidates and attempts < n_candidates * 50:
        attempts += 1

        shape = shapes[int(rng.choice(len(shapes), p=shape_weights))]

        # choose distinct teams biased by P_make_sb
        n_teams = len(shape)
        chosen = []
        while len(chosen) < n_teams:
            t = str(rng.choice(top_teams, p=weights))
            if t not in chosen:
                chosen.append(t)

        lineup_players: list[str] = []
        team_counts = {}

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
            team_counts[t] = cnt

        if not ok or len(set(lineup_players)) != 6:
            continue

        # Enforce min WC players
        num_wc = int(is_wc.loc[lineup_players].sum())
        if num_wc < min_wc_players:
            # Try to swap in WC satellites if possible
            # Simple repair: replace random bye-team player(s) with WC players
            lineup_set = set(lineup_players)
            bye_players = [p for p in lineup_players if not bool(is_wc.loc[p])]

            if not bye_players:
                continue

            needed = min_wc_players - num_wc
            repair_ok = True
            for _ in range(needed):
                if not bye_players:
                    repair_ok = False
                    break
                out_p = bye_players.pop()

                df_wc = wc_pool.loc[~wc_pool["Player"].isin(lineup_set)]
                if df_wc.empty:
                    repair_ok = False
                    break

                w = df_wc["FastPlayerValue"].clip(lower=0.0).values
                if w.sum() <= 0:
                    w = np.ones(len(df_wc))
                w = w / w.sum()
                in_p = str(rng.choice(df_wc["Player"].values, p=w))

                lineup_players.remove(out_p)
                lineup_players.append(in_p)
                lineup_set.remove(out_p)
                lineup_set.add(in_p)

            if not repair_ok:
                continue

            num_wc = int(is_wc.loc[lineup_players].sum())
            if num_wc < min_wc_players:
                continue

        candidates.append(lineup_players)

    return candidates


# -----------------------------
# Portfolio selection
# -----------------------------
def optimize_portfolio_10(
    pool: pd.DataFrame,
    n_candidates: int = 20000,
    shortlist_size: int = 400,
    k_dup: float = 900.0,
    overlap_lambda: float = 0.35,
    rng_seed: int = 1,
    min_wc_players: int = 4,
    bye_teams: set[str] | None = None,
    rams_team: str = "LAR",
    rams_heavy_threshold: int = 3,
    max_rams_heavy_portfolio: int = 3,
    max_rams_any_portfolio: int = 5,
) -> dict:
    """
    Portfolio optimizer with HARD Rams exposure caps:
      - max 3 lineups with >=3 Rams
      - max 5 lineups with any Rams
    """

    if bye_teams is None:
        bye_teams = {"SEA", "DEN"}

    rng = np.random.default_rng(rng_seed)

    # Ensure unique players
    pool = (
        pool.sort_values("FastPlayerValue", ascending=False)
        .drop_duplicates(subset=["Player"], keep="first")
        .copy()
    )

    idx = pool.set_index("Player")

    # -----------------------------
    # Generate candidates
    # -----------------------------
    candidates = generate_candidate_lineups(
        pool=pool,
        n_candidates=n_candidates,
        rng=rng,
        min_wc_players=min_wc_players,
    )

    if not candidates:
        raise ValueError("No candidates generated.")

    # -----------------------------
    # Score candidates
    # -----------------------------
    scored_rows = []

    for lineup in candidates:
        lineup_df = idx.loc[lineup].reset_index()
        s = score_lineup_fast(lineup_df, k_dup=k_dup)

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
                "NumRams": num_rams,
                "HasAnyRams": num_rams > 0,
                "IsRamsHeavy": num_rams >= rams_heavy_threshold,
            }
        )

    scored = (
        pd.DataFrame(scored_rows)
        .sort_values("EWFast", ascending=False)
        .reset_index(drop=True)
    )

    shortlist = scored.head(shortlist_size).copy()

    # -----------------------------
    # Greedy portfolio selection
    # -----------------------------
    selected = []
    selected_sets = []

    rams_any_selected = 0
    rams_heavy_selected = 0

    def overlap_penalty(players_set: set[str]) -> float:
        return sum(len(players_set & s) for s in selected_sets)

    def try_select(row) -> bool:
        nonlocal rams_any_selected, rams_heavy_selected

        if row["HasAnyRams"] and rams_any_selected >= max_rams_any_portfolio:
            return False
        if row["IsRamsHeavy"] and rams_heavy_selected >= max_rams_heavy_portfolio:
            return False

        players = list(row["Players"])
        players_set = set(players)

        obj = float(row["EWFast"]) - overlap_lambda * overlap_penalty(players_set)

        selected.append((obj, row))
        selected_sets.append(players_set)

        if row["HasAnyRams"]:
            rams_any_selected += 1
        if row["IsRamsHeavy"]:
            rams_heavy_selected += 1

        return True

    # Primary pass
    for _, row in shortlist.iterrows():
        if len(selected) >= 10:
            break
        try_select(row)

    # Backfill if needed
    if len(selected) < 10:
        for _, row in scored.iloc[shortlist_size:].iterrows():
            if len(selected) >= 10:
                break
            try_select(row)

    selected_rows = [r for _, r in sorted(selected, key=lambda x: x[0], reverse=True)[:10]]

    # -----------------------------
    # HARD ASSERT: caps must hold
    # -----------------------------
    def _num_rams(players_tuple: tuple[str, ...]) -> int:
        df = idx.loc[list(players_tuple)]
        return int((df["Team"].astype(str).str.strip() == rams_team).sum())
    
    actual_any = 0
    actual_heavy = 0
    for r in selected_rows:
        n = _num_rams(r["Players"])
        if n > 0:
            actual_any += 1
        if n >= rams_heavy_threshold:
            actual_heavy += 1
    
    if actual_any > max_rams_any_portfolio or actual_heavy > max_rams_heavy_portfolio:
        raise ValueError(
            f"Rams caps violated: any={actual_any} (cap {max_rams_any_portfolio}), "
            f"heavy={actual_heavy} (cap {max_rams_heavy_portfolio}). "
            f"Check selection logic or that this function is actually running."
        )

    # -----------------------------
    # Build final lineups
    # -----------------------------
    portfolio_lineups = []
    portfolio_summary_rows = []

    for i, r in enumerate(selected_rows, start=1):
        lineup_players = list(r["Players"])
        lineup_df = idx.loc[lineup_players].reset_index()
        lineup_df = assign_boosters_greedy_bestball(lineup_df)

        lineup_df["BoostedValue"] = lineup_df["FastPlayerValue"] * lineup_df["Booster"]
        top4 = lineup_df.sort_values("BoostedValue", ascending=False).head(4)

        portfolio_lineups.append(lineup_df)

        portfolio_summary_rows.append(
            {
                "Entry": i,
                "EWFast": r["EWFast"],
                "ScoreFast": r["ScoreFast"],
                "SplitFactor": r["SplitFactor"],
                "DupPressure": r["DupPressure"],
                "NumWildCard": r["NumWildCard"],
                "NumTeams": r["NumTeams"],
                "NumRams": r["NumRams"],
                "IsRamsHeavy": r["IsRamsHeavy"],
                "Top4BoostedSum": float(top4["BoostedValue"].sum()),
                "Players": ", ".join(lineup_players),
            }
        )
    st.write(result["portfolio_summary"].columns.tolist())
    portfolio_summary = pd.DataFrame(portfolio_summary_rows)

    # -----------------------------
    # Exposures
    # -----------------------------
    all_players = list(itertools.chain.from_iterable(df["Player"] for df in portfolio_lineups))
    exp_players = (
        pd.Series(all_players)
        .value_counts()
        .rename_axis("Player")
        .reset_index(name="Count")
    )
    exp_players["Exposure"] = exp_players["Count"] / 10.0
    exp_players = exp_players.merge(
        pool[["Player", "Team", "OwnershipFrac"]], on="Player", how="left"
    )

    all_teams = list(itertools.chain.from_iterable(df["Team"] for df in portfolio_lineups))
    exp_teams = (
        pd.Series(all_teams)
        .value_counts()
        .rename_axis("Team")
        .reset_index(name="Count")
    )
    exp_teams["Exposure"] = exp_teams["Count"] / (10.0 * 6.0)

return {
    "portfolio_lineups": portfolio_lineups,
    "portfolio_summary": portfolio_summary,
    "exposure_players": exp_players,
    "exposure_teams": exp_teams,
    "candidates_scored": shortlist,
}

        "exposure_players": exp_players,
        "exposure_teams": exp_teams,
        "candidates_scored": shortlist,
    }
