# playoff_draft_helper/bracket.py
NFC_TEAMS = {"SEA", "CHI", "GB", "PHI", "SF", "CAR", "LAR"}
AFC_TEAMS = {"DEN", "NE", "LAC", "JAX", "BUF", "PIT", "HOU"}

SEED = {
    "SEA": 1, "CHI": 2, "PHI": 3, "CAR": 4, "LAR": 5, "SF": 6, "GB": 7,
    "DEN": 1, "NE": 2, "JAX": 3, "PIT": 4, "HOU": 5, "BUF": 6, "LAC": 7,
}

WC_MATCHUPS = {
    "NFC": [("CHI", "GB"), ("PHI", "SF"), ("CAR", "LAR")],
    "AFC": [("NE", "LAC"), ("JAX", "BUF"), ("PIT", "HOU")],
}

BYE = {"NFC": "SEA", "AFC": "DEN"}

def conference_of(team: str) -> str:
    if team in NFC_TEAMS:
        return "NFC"
    if team in AFC_TEAMS:
        return "AFC"
    raise ValueError(f"Unknown team: {team}")

def teams_in_conf(conf: str) -> set[str]:
    return NFC_TEAMS if conf == "NFC" else AFC_TEAMS

def divisional_pairings(conf: str, wc_winners: list[str]) -> list[tuple[str, str]]:
    bye = BYE[conf]
    lowest = max(wc_winners, key=lambda t: SEED[t])  # worst seed number
    others = [t for t in wc_winners if t != lowest]
    return [(bye, lowest), (others[0], others[1])]
