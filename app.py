import streamlit as st
import pandas as pd

from playoff_draft_helper.data import load_data
from playoff_draft_helper.sim import build_round_probs, simulate_conditionals
from playoff_draft_helper.board import compute_board
from playoff_draft_helper.bracket import NFC_TEAMS, AFC_TEAMS
from playoff_draft_helper.portfolio import (
    build_team_round_probs,
    build_player_pool,
    optimize_portfolio_10,
)

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("🏈 Playoff Best Ball Draft Assistant")

# --------------------------------------------------
# Session state (draft memory)
# --------------------------------------------------
if "drafted_by_you" not in st.session_state:
    st.session_state.drafted_by_you = []

if "drafted_by_others" not in st.session_state:
    st.session_state.drafted_by_others = []

def reset_draft():
    st.session_state.drafted_by_you = []
    st.session_state.drafted_by_others = []

# --------------------------------------------------
# File uploads
# --------------------------------------------------
with st.expander("📂 Data Inputs", expanded=False):
    players_fp = st.file_uploader("Player Projections CSV", type=["csv"])
    win_odds_fp = st.file_uploader("Win Odds CSV", type=["csv"])
    adp_fp = st.file_uploader("ADP CSV", type=["csv"])

if not (players_fp and win_odds_fp and adp_fp):
    st.info("Upload all three CSV files to begin.")
    st.stop()

# --------------------------------------------------
# Load + simulate (cached)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_simulate(players_fp, win_odds_fp, adp_fp):
    players, win_odds, adp = load_data(players_fp, win_odds_fp, adp_fp)

    probs = build_round_probs(win_odds)

    cond_nfc, exp_not_nfc, _ = simulate_conditionals(
        "NFC", probs, n_sims=150_000, seed=1
    )
    cond_afc, exp_not_afc, _ = simulate_conditionals(
        "AFC", probs, n_sims=150_000, seed=2
    )

    return players, win_odds, adp, cond_nfc, exp_not_nfc, cond_afc, exp_not_afc


players, win_odds, adp, cond_nfc, exp_not_nfc, cond_afc, exp_not_afc = load_and_simulate(
    players_fp, win_odds_fp, adp_fp
)

# --------------------------------------------------
# TOP BAR: Pick info + locks
# --------------------------------------------------
top_left, top_mid, top_right = st.columns([2, 2, 2])

with top_left:
    current_pick = st.number_input(
        "Current Pick", min_value=1, max_value=60, value=1
    )
    next_pick = st.number_input(
        "Next Pick",
        min_value=current_pick + 1,
        max_value=60,
        value=current_pick + 10,
    )

with top_mid:
    lock_nfc = st.selectbox("NFC Lock", ["(auto)"] + sorted(NFC_TEAMS))
    lock_afc = st.selectbox("AFC Lock", ["(auto)"] + sorted(AFC_TEAMS))

lock_nfc_val = None if lock_nfc == "(auto)" else lock_nfc
lock_afc_val = None if lock_afc == "(auto)" else lock_afc

with top_right:
    st.markdown("### Your Roster")
    st.write(st.session_state.drafted_by_you)

    if st.button("🔄 Reset Draft"):
        reset_draft()
        st.rerun()

# --------------------------------------------------
# Compute board
# --------------------------------------------------
board, recommendations, meta = compute_board(
    players_df=players,
    win_odds_df=win_odds,
    adp_df=adp,
    cond_by_champ_nfc=cond_nfc,
    exp_if_not_champ_nfc=exp_not_nfc,
    cond_by_champ_afc=cond_afc,
    exp_if_not_champ_afc=exp_not_afc,
    drafted_players_in_order=st.session_state.drafted_by_you,
    drafted_by_others=st.session_state.drafted_by_others,
    current_pick=current_pick,
    next_pick=next_pick,
    lock_override_nfc=lock_nfc_val,
    lock_override_afc=lock_afc_val,
)

# --------------------------------------------------
# Portfolio optimizer
# --------------------------------------------------
st.subheader("🎯 Portfolio Optimizer (10 Entries)")

colA, colB, colC, colD = st.columns(4)

with colA:
    n_candidates = st.number_input(
        "Candidates", min_value=2000, max_value=100000, value=20000, step=2000
    )

with colB:
    shortlist_size = st.number_input(
        "Shortlist", min_value=50, max_value=2000, value=400, step=50
    )

with colC:
    k_dup = st.number_input(
        "Dup penalty k", min_value=0.0, max_value=3000.0, value=900.0, step=50.0
    )

with colD:
    overlap_lambda = st.number_input(
        "Overlap λ", min_value=0.0, max_value=2.0, value=0.35, step=0.05
    )

bye_teams = {"SEA", "DEN"}

if st.button("Generate optimized 10 entries"):
    team_probs = build_team_round_probs(win_odds, bye_teams=bye_teams)

    base_df = board if "TEFP" in board.columns else players

    pool = build_player_pool(
        players_df=base_df,
        adp_df=adp,
        team_round_probs=team_probs,
        bye_teams=bye_teams,
    )

    # THIS LINE MUST EXIST
    result = optimize_portfolio_10(
        pool=pool,
        n_candidates=int(n_candidates),
        shortlist_size=int(shortlist_size),
        k_dup=float(k_dup),
        overlap_lambda=float(overlap_lambda),
        rng_seed=1,
        min_wc_players=4,
        bye_teams=bye_teams,
    )

    st.subheader("DEBUG: Rams caps in selected portfolio")
    summary = result["portfolio_summary"].copy()
    st.write("Lineups with any LAR:", int((summary["NumRams"] > 0).sum()))
    st.write("Lineups with >=3 LAR:", int((summary["NumRams"] >= 3).sum()))
    
    # =============================
    # Screened set inspection
    # =============================
    screened = result["candidates_scored"]

    st.subheader("🔍 Screened Lineups (Top 200)")
    st.dataframe(
        screened.head(200),
        use_container_width=True,
        height=500,
    )

    st.subheader("📉 EWFast decay (Top 200)")
    st.line_chart(screened["EWFast"].head(200))

    teams = (
        screened["Players"]
        .apply(lambda ps: pd.Series(ps))
        .stack()
        .map(players.set_index("Player")["Team"])
    )

    team_counts = teams.value_counts().reset_index()
    team_counts.columns = ["Team", "Appearances"]

    st.subheader("🏈 Team frequency in screened set")
    st.dataframe(team_counts.head(10), use_container_width=True)

    st.session_state["portfolio_result"] = result
    #st.rerun()


# --------------------------------------------------
# Results
# --------------------------------------------------
if "portfolio_result" in st.session_state:
    result = st.session_state["portfolio_result"]

    st.markdown("### Portfolio summary")
    st.dataframe(result["portfolio_summary"], use_container_width=True, height=260)

    st.markdown("### Player exposures (10 entries)")
    st.dataframe(result["exposure_players"], use_container_width=True, height=260)

    st.markdown("### Team exposures")
    st.dataframe(result["exposure_teams"], use_container_width=True, height=220)

    st.markdown("### Entries")
    for i, df in enumerate(result["portfolio_lineups"], start=1):
        with st.expander(f"Entry {i}", expanded=False):
            st.dataframe(
                df[
                    [
                        "Player",
                        "Team",
                        "OwnershipFrac",
                        "FastPlayerValue",
                        "Booster",
                        "BoostedValue",
                    ]
                ],
                use_container_width=True,
                height=240,
            )

# --------------------------------------------------
# Recommendations
# --------------------------------------------------
left, right = st.columns([3, 2])

with left:
    st.subheader("🔥 Top 10 Recommended Picks")
    st.dataframe(
        recommendations[
            [
                "Player",
                "Team",
                "DraftPool_EffectiveCeiling",
                "VOR",
                "GoneProb",
                "MustHave",
            ]
        ],
        height=350,
        use_container_width=True,
    )

with right:
    st.subheader("✍️ Draft Actions")

    for _, row in recommendations.iterrows():
        st.markdown(f"**{row['Player']} ({row['Team']})**")

        c1, c2 = st.columns(2)

        with c1:
            if st.button(f"Draft Me", key=f"me_{row['Player']}"):
                st.session_state.drafted_by_you.append(row["Player"])
                st.rerun()

        with c2:
            if st.button(f"Drafted by Other", key=f"other_{row['Player']}"):
                st.session_state.drafted_by_others.append(row["Player"])
                st.rerun()

# --------------------------------------------------
# ADP pressure view
# --------------------------------------------------
st.subheader("📈 Next 10 by ADP")

adp_view = (
    board.loc[board["Available"]]
    .sort_values("ADP_Rank")
    .head(10)
)

st.dataframe(
    adp_view[
        ["Player", "Team", "ADP_Rank", "DraftPool_EffectiveCeiling"]
    ],
    height=300,
    use_container_width=True,
)

# --------------------------------------------------
# Full board
# --------------------------------------------------
with st.expander("📊 Full Draft Board"):
    st.dataframe(
        board[
            [
                "Player",
                "Team",
                "DraftPool_EffectiveCeiling",
                "VOR",
                "GoneProb",
                "DraftPriority",
                "MustHave",
                "DraftedByYou",
                "DraftedByOthers",
            ]
        ],
        height=500,
        use_container_width=True,
    )







