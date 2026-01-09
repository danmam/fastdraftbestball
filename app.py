import streamlit as st
import pandas as pd

from playoff_draft_helper.data import load_data
from playoff_draft_helper.sim import build_round_probs, simulate_conditionals
from playoff_draft_helper.board import compute_board
from playoff_draft_helper.bracket import NFC_TEAMS, AFC_TEAMS

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

# --------------------------------------------------
# File uploads (collapsed once loaded)
# --------------------------------------------------
with st.expander("📂 Data Inputs", expanded=False):
    players_fp = st.file_uploader("Player Projections CSV", type=["csv"])
    win_odds_fp = st.file_uploader("Win Odds CSV", type=["csv"])
    adp_fp = st.file_uploader("ADP CSV", type=["csv"])

# --------------------------------------------------
# Stop early if data missing
# --------------------------------------------------
if not (players_fp and win_odds_fp and adp_fp):
    st.info("Upload all three CSV files to begin.")
    st.stop()

# --------------------------------------------------
# Load + simulate (cached by Streamlit)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_simulate(players_fp, win_odds_fp, adp_fp):
    players, win_odds, adp = load_data(players_fp, win_odds_fp, adp_fp)
    probs = build_round_probs(win_odds)

    cond_nfc, exp_not_nfc, _ = simulate_conditionals("NFC", probs, n_sims=150_000, seed=1)
    cond_afc, exp_not_afc, _ = simulate_conditionals("AFC", probs, n_sims=150_000, seed=2)

    return players, win_odds, adp, cond_nfc, exp_not_nfc, cond_afc, exp_not_afc

players, win_odds, adp, cond_nfc, exp_not_nfc, cond_afc, exp_not_afc = load_and_simulate(
    players_fp, win_odds_fp, adp_fp
)

# --------------------------------------------------
# TOP BAR: Pick info + locks
# --------------------------------------------------
top_left, top_mid, top_right = st.columns([2, 2, 2])

with top_left:
    current_pick = st.number_input("Current Pick", min_value=1, max_value=60, value=1)
    next_pick = st.number_input("Next Pick", min_value=current_pick + 1, max_value=60, value=current_pick + 10)

with top_mid:
    lock_nfc = st.selectbox("NFC Lock", ["(auto)"] + sorted(NFC_TEAMS))
    lock_afc = st.selectbox("AFC Lock", ["(auto)"] + sorted(AFC_TEAMS))

lock_nfc_val = None if lock_nfc == "(auto)" else lock_nfc
lock_afc_val = None if lock_afc == "(auto)" else lock_afc

with top_right:
    st.markdown("### Your Roster")
    st.write(st.session_state.drafted_by_you)

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
# MAIN PANEL: Recommendations + Actions
# --------------------------------------------------
left, right = st.columns([3, 2])

with left:
    st.subheader("🔥 Top 10 Recommended Picks")

    st.dataframe(
        recommendations[
            ["Player", "Team", "DraftPool_EffectiveCeiling", "VOR", "GoneProb", "MustHave"]
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
# OPTIONAL: Full board (collapsed)
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

