import streamlit as st
import pandas as pd

from playoff_draft_helper.data import load_data
from playoff_draft_helper.sim import build_round_probs, simulate_conditionals
from playoff_draft_helper.board import compute_board
from playoff_draft_helper.bracket import NFC_TEAMS, AFC_TEAMS

# --------------------------------------------------
# Streamlit setup
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("Playoff Best Ball Draft Helper — Simulation-Based EV")

st.markdown(
    """
This tool recommends playoff best ball picks using:
- **True bracket simulation**
- **Conditional expected games**
- **Draft-pool suppression after locks**
- **Value Over Replacement + ADP urgency**

Draft decisions are optimized for **first-place equity**, not median outcomes.
"""
)

# --------------------------------------------------
# File uploads
# --------------------------------------------------
players_fp = st.file_uploader(
    "Player Fantasy Projections.csv (exact schema)",
    type=["csv"],
)

win_odds_fp = st.file_uploader(
    "Playoff Pick Em Popularity and Win Odds by Round.csv",
    type=["csv"],
)

adp_fp = st.file_uploader(
    "FastDraft ADP.csv",
    type=["csv"],
)

# --------------------------------------------------
# Simulation controls
# --------------------------------------------------
n_sims = st.slider(
    "Monte Carlo simulations per conference",
    min_value=50_000,
    max_value=500_000,
    value=200_000,
    step=50_000,
)

seed = st.number_input(
    "Random seed",
    min_value=0,
    max_value=10_000_000,
    value=1,
)

# --------------------------------------------------
# Main logic
# --------------------------------------------------
if players_fp and win_odds_fp and adp_fp:
    players, win_odds, adp = load_data(players_fp, win_odds_fp, adp_fp)

    probs = build_round_probs(win_odds)

    with st.spinner("Simulating NFC bracket..."):
        cond_nfc, exp_not_nfc, champ_counts_nfc = simulate_conditionals(
            "NFC",
            probs,
            n_sims=n_sims,
            seed=seed,
        )

    with st.spinner("Simulating AFC bracket..."):
        cond_afc, exp_not_afc, champ_counts_afc = simulate_conditionals(
            "AFC",
            probs,
            n_sims=n_sims,
            seed=seed + 1,
        )

    # --------------------------------------------------
    # Draft state inputs
    # --------------------------------------------------
    st.subheader("Draft State")

    drafted_by_you = st.multiselect(
        "Players drafted by YOU (in exact draft order)",
        options=players["Player"].tolist(),
        default=[],
    )

    drafted_by_others = st.multiselect(
        "Players drafted by OTHER teams",
        options=[
            p for p in players["Player"].tolist()
            if p not in drafted_by_you
        ],
        default=[],
    )

    col1, col2 = st.columns(2)

    with col1:
        current_pick = st.number_input(
            "Current pick number",
            min_value=1,
            max_value=60,
            value=1,
        )

    with col2:
        next_pick = st.number_input(
            "Next pick number",
            min_value=current_pick + 1,
            max_value=60,
            value=current_pick + 10,
        )

    # --------------------------------------------------
    # Lock overrides
    # --------------------------------------------------
    st.subheader("Conference Lock Overrides")

    col3, col4 = st.columns(2)

    with col3:
        lock_nfc = st.selectbox(
            "NFC lock",
            options=["(auto)"] + sorted(list(NFC_TEAMS)),
            index=0,
        )

    with col4:
        lock_afc = st.selectbox(
            "AFC lock",
            options=["(auto)"] + sorted(list(AFC_TEAMS)),
            index=0,
        )

    lock_nfc_val = None if lock_nfc == "(auto)" else lock_nfc
    lock_afc_val = None if lock_afc == "(auto)" else lock_afc

    # --------------------------------------------------
    # Compute board + recommendations
    # --------------------------------------------------
    board, recommendations, meta = compute_board(
        players_df=players,
        win_odds_df=win_odds,
        adp_df=adp,
        cond_by_champ_nfc=cond_nfc,
        exp_if_not_champ_nfc=exp_not_nfc,
        cond_by_champ_afc=cond_afc,
        exp_if_not_champ_afc=exp_not_afc,
        drafted_players_in_order=drafted_by_you,
        drafted_by_others=drafted_by_others,
        current_pick=current_pick,
        next_pick=next_pick,
        lock_override_nfc=lock_nfc_val,
        lock_override_afc=lock_afc_val,
    )

    # --------------------------------------------------
    # Lock summary
    # --------------------------------------------------
    st.markdown(
        f"""
**Auto Locks:**  
• NFC: `{meta['AutoLock_NFC']}`  
• AFC: `{meta['AutoLock_AFC']}`  

**Active Locks:**  
• NFC: `{meta['UsingLock_NFC']}`  
• AFC: `{meta['UsingLock_AFC']}`  

**Replacement Ceiling:** `{meta['ReplacementCeiling']:.2f}`
"""
    )

    # --------------------------------------------------
    # Top-10 recommendations
    # --------------------------------------------------
    st.subheader("Top 10 Recommended Picks")

    st.dataframe(
        recommendations[
            [
                "Player",
                "Team",
                "Conference",
                "DraftPool_EffectiveCeiling",
                "VOR",
                "GoneProb",
                "DraftPriority",
                "MustHave",
                "ADP_Rank",
            ]
        ],
        use_container_width=True,
        height=350,
    )

    # --------------------------------------------------
    # MUST-HAVES
    # --------------------------------------------------
    must_haves = recommendations[recommendations["MustHave"]]

    if not must_haves.empty:
        st.warning("🚨 MUST‑HAVE PICKS (High EV + Will Be Gone)")
        st.dataframe(
            must_haves[
                [
                    "Player",
                    "Team",
                    "DraftPool_EffectiveCeiling",
                    "VOR",
                    "GoneProb",
                    "ADP_Rank",
                ]
            ],
            use_container_width=True,
        )

    # --------------------------------------------------
    # Full draft board
    # --------------------------------------------------
    st.subheader("Full Draft Board")

    st.dataframe(
        board[
            [
                "Player",
                "Team",
                "Conference",
                "Position",
                "Role",
                "DraftPool_EffectiveCeiling",
                "VOR",
                "GoneProb",
                "DraftPriority",
                "MustHave",
                "ADP_Rank",
                "DraftedByYou",
                "DraftedByOthers",
            ]
        ],
        use_container_width=True,
        height=600,
    )

    # --------------------------------------------------
    # Explanation
    # --------------------------------------------------
    with st.expander("How recommendations are calculated"):
        st.markdown(
            """
**Draft Priority** balances two forces:

• **Value Over Replacement (VOR)**  
  How much ceiling you gain vs waiting.

• **Gone Probability**  
  Likelihood the player is drafted before your next pick.

**MUST‑HAVE** = Positive VOR *and* high gone probability.

This framework optimizes **first‑place equity**, not median outcomes.
"""
        )

else:
    st.info("Upload all three CSV files to begin.")
