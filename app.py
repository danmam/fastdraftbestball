# app.py
import streamlit as st

from playoff_draft_helper.data import load_data
from playoff_draft_helper.sim import build_round_probs, simulate_conditionals
from playoff_draft_helper.board import compute_board
from playoff_draft_helper.bracket import NFC_TEAMS, AFC_TEAMS

st.set_page_config(layout="wide")
st.title("Playoff Best Ball Draft Helper — Simulation-based ceiling reductions")

st.markdown(
"""
**Default behavior:** first drafted team in each conference is assumed to win that conference (SB representative).  
**Draft pool ceilings update accordingly:** other same-conference teams get reduced via bracket simulation conditioning.  
You can **override locks manually** if you want to pivot.
"""
)

players_fp = st.file_uploader("Player Fantasy Projections.csv (exact schema)", type=["csv"])
win_odds_fp = st.file_uploader("Playoff Pick Em Popularity and Win Odds by Round.csv", type=["csv"])
adp_fp = st.file_uploader("FastDraft ADP.csv", type=["csv"])

n_sims = st.slider("Monte Carlo sims per conference", 50000, 500000, 200000, step=50000)
seed = st.number_input("RNG seed", min_value=0, max_value=10_000_000, value=1)

if players_fp and win_odds_fp and adp_fp:
    players, win_odds, adp = load_data(players_fp, win_odds_fp, adp_fp)

    probs = build_round_probs(win_odds)

    with st.spinner("Simulating NFC conditional expectations..."):
        cond_nfc, exp_not_nfc, champ_counts_nfc = simulate_conditionals("NFC", probs, n_sims=n_sims, seed=seed)
    with st.spinner("Simulating AFC conditional expectations..."):
        cond_afc, exp_not_afc, champ_counts_afc = simulate_conditionals("AFC", probs, n_sims=n_sims, seed=seed + 1)

    st.subheader("Draft state")

    drafted_players = st.multiselect(
        "Drafted players (in exact draft order; first = earliest)",
        options=players["Player"].tolist(),
        default=[]
    )

    st.subheader("Lock overrides")

    c1, c2 = st.columns(2)
    with c1:
        lock_nfc = st.selectbox(
            "NFC lock override (optional)",
            options=["(auto)"] + sorted(list(NFC_TEAMS)),
            index=0
        )
    with c2:
        lock_afc = st.selectbox(
            "AFC lock override (optional)",
            options=["(auto)"] + sorted(list(AFC_TEAMS)),
            index=0
        )

    lock_nfc_val = None if lock_nfc == "(auto)" else lock_nfc
    lock_afc_val = None if lock_afc == "(auto)" else lock_afc

    board, meta = compute_board(
        players_df=players,
        win_odds_df=win_odds,
        adp_df=adp,
        cond_by_champ_nfc=cond_nfc,
        exp_if_not_champ_nfc=exp_not_nfc,
        cond_by_champ_afc=cond_afc,
        exp_if_not_champ_afc=exp_not_afc,
        drafted_players_in_order=drafted_players,
        lock_override_nfc=lock_nfc_val,
        lock_override_afc=lock_afc_val,
    )

    st.markdown(f"""
**Auto locks:** NFC={meta["AutoLock_NFC"]}, AFC={meta["AutoLock_AFC"]}  
**Using locks:** NFC={meta["UsingLock_NFC"]}, AFC={meta["UsingLock_AFC"]}
""")

    st.subheader("Draft board (sorted by DraftPool_EffectiveCeiling)")

    show_cols = [
        "Player", "Team", "Conference", "Position", "Role",
        "TEFP",
        "Ceiling_if_SB",
        "EffGames_if_NOT_ConfChamp",
        "Ceiling_if_NOT_ConfChamp",
        "DraftPool_EffectiveCeiling",
        "Booster_2.0x", "Booster_1.75x",
        "ADP_Rank", "Value_Gap"
    ]

    st.dataframe(
        board.loc[~board["IsDrafted"], show_cols],
        use_container_width=True,
        height=700
    )

    with st.expander("Champion frequency sanity check"):
        st.write("If a team has near-zero champ counts, increase sims for stable conditioning.")
        st.write({"NFC": champ_counts_nfc, "AFC": champ_counts_afc})

else:
    st.info("Upload the three CSVs to begin (same schemas as your files).")
