"""Financial-specific sentiment lexicon for VADER enhancement."""

# Positive financial terms (score: 0.5 to 3.0)
POSITIVE_TERMS = {
    # Strong positive (2.0 - 3.0)
    "bullish": 2.5,
    "surge": 2.5,
    "soar": 2.5,
    "skyrocket": 3.0,
    "rally": 2.0,
    "boom": 2.5,
    "breakout": 2.0,
    "all-time high": 2.5,
    "record high": 2.5,
    "outperform": 2.0,
    "beat expectations": 2.5,
    "beat estimates": 2.5,
    "exceeds expectations": 2.5,
    # Moderate positive (1.0 - 2.0)
    "upgrade": 1.8,
    "buy rating": 1.5,
    "strong buy": 2.0,
    "growth": 1.2,
    "profit": 1.5,
    "gains": 1.2,
    "recovery": 1.5,
    "rebound": 1.5,
    "momentum": 1.2,
    "uptrend": 1.5,
    "positive": 1.0,
    "optimistic": 1.2,
    "confident": 1.0,
    "expansion": 1.2,
    "innovation": 1.0,
    # Mild positive (0.5 - 1.0)
    "stable": 0.5,
    "steady": 0.5,
    "solid": 0.8,
    "healthy": 0.8,
    "improving": 0.8,
}

# Negative financial terms (score: -0.5 to -3.0)
NEGATIVE_TERMS = {
    # Strong negative (-2.0 to -3.0)
    "bearish": -2.5,
    "crash": -3.0,
    "collapse": -3.0,
    "plunge": -2.5,
    "plummet": -2.5,
    "tank": -2.5,
    "meltdown": -3.0,
    "bankruptcy": -3.0,
    "default": -2.5,
    "fraud": -3.0,
    "scandal": -2.5,
    "miss expectations": -2.5,
    "miss estimates": -2.5,
    "below expectations": -2.0,
    # Moderate negative (-1.0 to -2.0)
    "downgrade": -1.8,
    "sell rating": -1.5,
    "strong sell": -2.0,
    "loss": -1.5,
    "losses": -1.5,
    "decline": -1.5,
    "drop": -1.2,
    "fall": -1.2,
    "slump": -1.5,
    "selloff": -1.8,
    "sell-off": -1.8,
    "downtrend": -1.5,
    "recession": -2.0,
    "layoffs": -1.5,
    "lawsuit": -1.5,
    "investigation": -1.2,
    "probe": -1.0,
    # Mild negative (-0.5 to -1.0)
    "concern": -0.8,
    "concerns": -0.8,
    "uncertainty": -0.8,
    "volatile": -0.5,
    "volatility": -0.5,
    "risk": -0.5,
    "risky": -0.8,
    "weak": -0.8,
    "slowdown": -0.8,
}

# Intensifiers specific to finance
INTENSIFIERS = {
    "significantly": 1.5,
    "dramatically": 1.8,
    "sharply": 1.5,
    "substantially": 1.3,
    "massively": 1.8,
    "slightly": 0.5,
    "marginally": 0.5,
    "modestly": 0.7,
}

# Negation handling (these flip sentiment)
NEGATIONS = {
    "not",
    "no",
    "never",
    "neither",
    "nobody",
    "nothing",
    "nowhere",
    "hardly",
    "barely",
    "scarcely",
    "doesn't",
    "isn't",
    "wasn't",
    "shouldn't",
    "wouldn't",
    "couldn't",
    "won't",
    "can't",
    "don't",
}


def get_financial_lexicon() -> dict:
    """Get combined financial lexicon for VADER.

    Returns:
        Dictionary mapping terms to sentiment scores
    """
    lexicon = {}
    lexicon.update(POSITIVE_TERMS)
    lexicon.update(NEGATIVE_TERMS)
    return lexicon
