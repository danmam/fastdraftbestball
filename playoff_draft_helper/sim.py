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
