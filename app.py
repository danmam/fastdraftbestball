import streamlit as st
import pandas as pd
import numpy as np

from playoff_draft_helper.data import load_data
from playoff_draft_helper.sim import build_round_probs
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

players, win_odds, adp = load_data(players_fp, win_odds_fp, adp_fp)

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
cond_nfc = {} 
exp_not_nfc = {} 
cond_afc = {} 
exp_not_afc = {}
board, recommendations, meta = compute_board(
    players_df=players,
    win_odds_df=win_odds,
    adp_df=adp,
    cond_by_champ_nfc=cond_nfc,
    exp_if_not_champ_nfc=exp_not_nfc,
    cond_by_champ_afc=cond_afc,
    exp_if_not_champ_afc=exp_not_afc,  # Fixed: was exp_if_not_afc
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

# Template selector
st.markdown("##### Quick Start Templates")
template_col1, template_col2 = st.columns([3, 1])

with template_col1:
    template = st.selectbox(
        "Load preset configuration:",
        ["Custom", "Tournament GPP", "Balanced", "Cash/Low Variance", "Contrarian Chaos"],
        help="Quick-load common strategy profiles"
    )

with template_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    load_template = st.button("Load Template")

# Template configurations
if load_template or template != "Custom":
    if template == "Tournament GPP":
        st.session_state.template_config = {
            "min_wc": 3, "min_stack": 2, "max_per_team": 4,
            "k_dup": 1200.0, "overlap": 0.5, "leverage": 2.5,
            "max_heavy": 2, "max_any": 5
        }
    elif template == "Balanced":
        st.session_state.template_config = {
            "min_wc": 2, "min_stack": 2, "max_per_team": 3,
            "k_dup": 900.0, "overlap": 0.35, "leverage": 2.0,
            "max_heavy": 3, "max_any": 5
        }
    elif template == "Cash/Low Variance":
        st.session_state.template_config = {
            "min_wc": 1, "min_stack": 2, "max_per_team": 3,
            "k_dup": 500.0, "overlap": 0.2, "leverage": 1.5,
            "max_heavy": 5, "max_any": 7
        }
    elif template == "Contrarian Chaos":
        st.session_state.template_config = {
            "min_wc": 4, "min_stack": 3, "max_per_team": 4,
            "k_dup": 1500.0, "overlap": 0.8, "leverage": 3.5,
            "max_heavy": 1, "max_any": 3
        }

# Get template config if exists
if "template_config" not in st.session_state:
    st.session_state.template_config = {}

tc = st.session_state.template_config

# Basic settings
st.markdown("##### Generation Settings")
colA, colB, colC, colD, colE = st.columns(5)

with colA:
    n_candidates = st.number_input(
        "Candidates", min_value=2000, max_value=100000, value=20000, step=2000,
        help="Number of random lineups to generate before filtering"
    )

with colB:
    shortlist_size = st.number_input(
        "Shortlist", min_value=50, max_value=2000, value=400, step=50,
        help="Number of top candidates to simulate in detail"
    )

with colC:
    n_sims = st.number_input(
        "Simulations", min_value=100, max_value=10000, value=1000, step=100,
        help="Number of bracket simulations per lineup"
    )

with colD:
    k_dup = st.number_input(
        "Dup penalty k", min_value=0.0, max_value=3000.0, value=900.0, step=50.0,
        help="Penalty for duplicate/chalky ownership"
    )

with colE:
    overlap_lambda = st.number_input(
        "Overlap λ", min_value=0.0, max_value=2.0, value=0.35, step=0.05,
        help="Player overlap penalty between lineups"
    )

# Lineup construction constraints
with st.expander("⚙️ Lineup Construction Rules", expanded=False):
    st.markdown("##### Stack and Team Constraints")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        min_wc_players = st.number_input(
            "Min Wild Card Players",
            min_value=0,
            max_value=6,
            value=tc.get("min_wc", 2),
            step=1,
            help="Minimum players from Wild Card weekend teams (no bye week)"
        )
    
    with col2:
        min_stack = st.number_input(
            "Min Stack Size",
            min_value=2,
            max_value=4,
            value=tc.get("min_stack", 2),
            step=1,
            help="Minimum players from any single team in a lineup"
        )
    
    with col3:
        max_players_per_team = st.number_input(
            "Max Players Per Team",
            min_value=2,
            max_value=6,
            value=tc.get("max_per_team", 4),
            step=1,
            help="Maximum players allowed from any single team"
        )
    
    with col4:
        leverage_beta = st.number_input(
            "Leverage Beta",
            min_value=0.5,
            max_value=5.0,
            value=tc.get("leverage", 2.0),
            step=0.5,
            help="Higher = more preference for low-ownership lineups"
        )
    
    st.markdown("##### Team-Specific Caps")
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        rams_team = st.text_input(
            "Cap Team Code",
            value="LAR",
            help="Team abbreviation to apply exposure caps (e.g., LAR, PHI)"
        )
    
    with col6:
        rams_heavy_threshold = st.number_input(
            "Heavy Stack Threshold",
            min_value=2,
            max_value=6,
            value=3,
            step=1,
            help=f"Number of {rams_team} players to be considered 'heavy'"
        )
    
    with col7:
        max_rams_heavy = st.number_input(
            "Max Heavy Stacks",
            min_value=0,
            max_value=10,
            value=tc.get("max_heavy", 3),
            step=1,
            help=f"Max lineups with {rams_heavy_threshold}+ {rams_team} players"
        )
    
    with col8:
        max_rams_any = st.number_input(
            "Max Any Exposure",
            min_value=0,
            max_value=10,
            value=tc.get("max_any", 5),
            step=1,
            help=f"Max lineups with any {rams_team} players"
        )
    
    st.markdown("##### Advanced Filters")
    col9, col10 = st.columns(2)
    
    with col9:
        feasibility_gate = st.number_input(
            "Feasibility Gate (P4 Div/CC/SB)",
            min_value=0.0,
            max_value=1.0,
            value=0.03,
            step=0.01,
            format="%.3f",
            help="Minimum probability of having 4 players in Div/CC/SB rounds"
        )
    
    with col10:
        n_workers = st.number_input(
            "Parallel Workers",
            min_value=0,  # Changed: allow 0 for auto-detect
            max_value=16,
            value=0,
            step=1,
            help="CPU cores for parallel processing (0 = auto-detect)"
        )
        n_workers = None if n_workers == 0 else n_workers

bye_teams = {"SEA", "DEN"}

# Validation warnings
if min_wc_players + (6 - max_players_per_team) > 6:
    st.warning(f"⚠️ Your constraints may be impossible: Min WC Players ({min_wc_players}) + diversity requirements may exceed 6 players.")

if min_stack > max_players_per_team:
    st.error(f"❌ Invalid: Min Stack ({min_stack}) cannot be larger than Max Players Per Team ({max_players_per_team})")
    st.stop()

if max_rams_heavy > max_rams_any:
    st.warning(f"⚠️ Warning: Max Heavy Stacks ({max_rams_heavy}) should be ≤ Max Any Exposure ({max_rams_any})")

# Display constraint summary
with st.expander("📋 Current Constraint Summary", expanded=False):
    st.markdown(f"""
    **Lineup Construction:**
    - Each lineup requires **{min_wc_players}-6** Wild Card players
    - Each lineup requires **≥{min_stack}** players from at least one team
    - No team can have **>{max_players_per_team}** players in a lineup
    
    **Portfolio Diversification:**
    - Player overlap penalty: **{overlap_lambda}**
    - Ownership penalty: **{k_dup}**
    - Leverage weighting: **{leverage_beta}**
    
    **{rams_team} Exposure Caps:**
    - Max lineups with any {rams_team} players: **{max_rams_any}/10**
    - Max lineups with {rams_heavy_threshold}+ {rams_team} players: **{max_rams_heavy}/10**
    
    **Quality Filters:**
    - Min P(4 players Div/CC/SB): **{feasibility_gate:.1%}**
    """)

if st.button("Generate optimized 10 entries"):
    team_probs = build_team_round_probs(win_odds, bye_teams=bye_teams)

    base_df = board if "TEFP" in board.columns else players

    pool = build_player_pool(
        players_df=base_df,
        adp_df=adp,
        team_round_probs=team_probs,
        bye_teams=bye_teams,
    )

    # Progress tracking
    progress_bar = st.progress(0, text="Initializing optimization...")
    status_text = st.empty()
    
    def update_progress(message):
        status_text.text(message)
    
    try:
        # Update progress at start
        progress_bar.progress(5, text="🎲 Pre-generating bracket simulations...")
        status_text.info(f"Generating {n_sims} NFL playoff bracket simulations (will be reused for all lineups)")
        
        # Run optimized portfolio generation
        progress_bar.progress(10, text="🎯 Starting portfolio optimization...")
        
        result = optimize_portfolio_10(
            pool=pool,
            win_odds_df=win_odds,
            n_candidates=n_candidates,
            k_shortlist=shortlist_size,
            n_sims=n_sims,
            k_dup=float(k_dup),
            overlap_lambda=float(overlap_lambda),
            rng_seed=1,
            min_wc_players=min_wc_players,
            min_stack=min_stack,
            max_players_per_team=max_players_per_team,
            leverage_beta=leverage_beta,
            feasibility_gate_div_cc_sb=feasibility_gate,
            bye_teams=bye_teams,
            rams_team=rams_team,
            rams_heavy_threshold=rams_heavy_threshold,
            max_rams_heavy_portfolio=max_rams_heavy,
            max_rams_any_portfolio=max_rams_any,
            n_workers=n_workers,
            progress_callback=update_progress,
        )
        
        progress_bar.progress(100, text="✓ Optimization complete!")
        
        # Display results with performance info
        st.success(f"✓ Generated 10 optimized lineups")
        st.info(f"⚡ Performance: Used {n_sims} bracket simulations (cached and reused across {shortlist_size} candidate lineups)")
        
        # Key metrics explanation
        with st.expander("ℹ️ Understanding the Results", expanded=False):
            st.markdown("""
            **Portfolio Summary Columns:**
            - **MaxTotalSim**: Highest total score across all simulated brackets (ceiling)
            - **Q95**: 95th percentile score (high-but-realistic outcome)
            - **P4DivCCSB**: Probability of having 4+ players in Divisional, CC, and SB rounds
            - **NumWildCard**: Number of Wild Card weekend players in lineup
            - **MaxStack**: Largest team stack size
            - **NumRams**: Number of players from the capped team
            - **MeanOwnership**: Average ownership of players in lineup
            
            **Strategy:**
            - Higher MaxTotalSim = higher ceiling but may be less stable
            - Higher P4DivCCSB = more likely to "survive" deep into playoffs
            - Lower MeanOwnership = more contrarian/differentiated
            """)
        
        st.subheader("📊 Portfolio Summary")
        st.dataframe(
            result["portfolio_summary"][[
                "Entry", "MaxTotalSim", "Q95", "P4DivCCSB", 
                "NumWildCard", "MaxStack", "NumRams", "MeanOwnership"
            ]],
            use_container_width=True,
            height=280,
        )
        
        st.subheader("👥 Player Exposures")
        st.dataframe(
            result["exposure_players"].head(20),
            use_container_width=True,
            height=300,
        )
        
        st.subheader("🏈 Team Exposures")
        st.dataframe(
            result["exposure_teams"],
            use_container_width=True,
            height=220,
        )
        
        st.subheader("📋 Final 10 Lineups")
        
        for i, lineup_df in enumerate(result["portfolio_lineups"], start=1):
            with st.expander(f"Lineup {i} - Max: {result['portfolio_summary'].iloc[i-1]['MaxTotalSim']:.1f}", expanded=(i==1)):
                st.dataframe(
                    lineup_df[["Player", "Team", "FastPlayerValue", "Booster", "BoostedValue"]],
                    use_container_width=True,
                    hide_index=True,
                )
        
        # Analysis section
        st.subheader("🔍 Candidate Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top 50 Candidates by Simulated Max**")
            st.dataframe(
                result["candidates_scored"][["MaxTotalSim", "Q95", "P4DivCCSB", "NumWildCard", "MaxStack"]].head(50),
                use_container_width=True,
                height=300,
            )
        
        with col2:
            st.markdown("**Score Distribution**")
            st.line_chart(
                result["candidates_scored"]["MaxTotalSim"].head(100),
                height=280,
            )
        
    except Exception as e:
        st.error(f"Optimization failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    finally:
        progress_bar.empty()
        status_text.empty()


# --------------------------------------------------
# Results (if stored in session state from previous runs)
# --------------------------------------------------
if "portfolio_result" in st.session_state:
    result = st.session_state["portfolio_result"]

    st.markdown("### 💾 Saved Portfolio")
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
