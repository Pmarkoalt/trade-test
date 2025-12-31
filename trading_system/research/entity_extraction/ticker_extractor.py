"""Ticker/symbol extraction from news articles."""

import re
from typing import List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


# Known tickers and their variations
TICKER_ALIASES = {
    # Equities
    "AAPL": ["apple", "iphone", "ipad", "mac", "tim cook"],
    "MSFT": ["microsoft", "windows", "azure", "satya nadella"],
    "GOOGL": ["google", "alphabet", "youtube", "android", "sundar pichai"],
    "AMZN": ["amazon", "aws", "prime", "jeff bezos", "andy jassy"],
    "NVDA": ["nvidia", "geforce", "jensen huang"],
    "META": ["meta", "facebook", "instagram", "whatsapp", "mark zuckerberg"],
    "TSLA": ["tesla", "elon musk", "model s", "model 3", "model y"],
    "JPM": ["jpmorgan", "jp morgan", "jamie dimon"],
    "V": ["visa"],
    "JNJ": ["johnson & johnson", "johnson and johnson"],
    "WMT": ["walmart", "walmart inc"],
    "PG": ["procter & gamble", "procter and gamble"],
    "MA": ["mastercard"],
    "UNH": ["unitedhealth", "united health"],
    "HD": ["home depot"],
    "DIS": ["disney", "walt disney"],
    "BAC": ["bank of america", "bofa"],
    "ADBE": ["adobe"],
    "NFLX": ["netflix"],
    "CRM": ["salesforce"],
    "PYPL": ["paypal"],
    "INTC": ["intel"],
    "CMCSA": ["comcast"],
    "PEP": ["pepsico", "pepsi"],
    "COST": ["costco"],
    "TMO": ["thermo fisher"],
    "AVGO": ["broadcom"],
    "CSCO": ["cisco"],
    "ABT": ["abbott"],
    "DHR": ["danaher"],
    # Crypto
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth", "ether"],
    "BNB": ["binance coin", "bnb"],
    "XRP": ["ripple", "xrp"],
    "ADA": ["cardano", "ada"],
    "SOL": ["solana", "sol"],
    "DOT": ["polkadot", "dot"],
    "MATIC": ["polygon", "matic"],
    "LTC": ["litecoin", "ltc"],
    "LINK": ["chainlink", "link"],
}

# Build reverse lookup
ALIAS_TO_TICKER = {}
for ticker, aliases in TICKER_ALIASES.items():
    for alias in aliases:
        ALIAS_TO_TICKER[alias.lower()] = ticker


class TickerExtractor:
    """Extract stock/crypto tickers from text."""

    def __init__(self, valid_tickers: Optional[Set[str]] = None):
        """Initialize extractor.

        Args:
            valid_tickers: Set of valid tickers to recognize.
                          If None, uses built-in list.
        """
        self.valid_tickers = valid_tickers or set(TICKER_ALIASES.keys())

        # Pattern for explicit ticker mentions (e.g., $AAPL, AAPL:, (AAPL))
        self.ticker_pattern = re.compile(
            r'(?:\$([A-Z]{1,5})|'  # $AAPL
            r'\b([A-Z]{1,5})(?::|(?=\s+(?:stock|shares|price|falls|rises|surges|drops)))|'  # AAPL: or AAPL stock
            r'\(([A-Z]{1,5})\))'  # (AAPL)
        )

    def extract(self, text: str) -> List[str]:
        """Extract tickers from text.

        Args:
            text: Text to search

        Returns:
            List of found ticker symbols (deduplicated)
        """
        if not text:
            return []

        found_tickers = set()
        text_lower = text.lower()

        # 1. Find explicit ticker mentions
        for match in self.ticker_pattern.finditer(text):
            for group in match.groups():
                if group and group.upper() in self.valid_tickers:
                    found_tickers.add(group.upper())

        # 2. Find company name mentions
        for alias, ticker in ALIAS_TO_TICKER.items():
            if ticker in self.valid_tickers and alias in text_lower:
                # Verify it's a word boundary match
                pattern = r'\b' + re.escape(alias) + r'\b'
                if re.search(pattern, text_lower):
                    found_tickers.add(ticker)

        return list(found_tickers)

    def extract_with_context(self, text: str) -> List[Tuple[str, str]]:
        """Extract tickers with surrounding context.

        Args:
            text: Text to search

        Returns:
            List of (ticker, context) tuples
        """
        results = []
        tickers = self.extract(text)

        for ticker in tickers:
            # Find context around ticker mention
            context = self._get_context(text, ticker)
            results.append((ticker, context))

        return results

    def _get_context(self, text: str, ticker: str, window: int = 50) -> str:
        """Get text context around ticker mention.

        Args:
            text: Full text
            ticker: Ticker to find
            window: Characters before/after to include

        Returns:
            Context string
        """
        text_lower = text.lower()
        aliases = [ticker.lower()] + [a for a, t in ALIAS_TO_TICKER.items() if t == ticker]

        for alias in aliases:
            idx = text_lower.find(alias)
            if idx >= 0:
                start = max(0, idx - window)
                end = min(len(text), idx + len(alias) + window)
                context = text[start:end].strip()
                if start > 0:
                    context = "..." + context
                if end < len(text):
                    context = context + "..."
                return context

        return ""
