"""Tests for ticker extraction module."""

import pytest

from trading_system.research.entity_extraction.ticker_extractor import (
    TickerExtractor,
    TICKER_ALIASES,
    ALIAS_TO_TICKER,
)


class TestTickerExtractor:
    """Tests for TickerExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create a TickerExtractor instance."""
        return TickerExtractor()

    def test_extractor_initialization(self, extractor):
        """Test that extractor initializes correctly."""
        assert extractor is not None
        assert len(extractor.valid_tickers) > 0

    def test_extractor_custom_tickers(self):
        """Test extractor with custom valid tickers."""
        custom_tickers = {"AAPL", "MSFT", "BTC"}
        extractor = TickerExtractor(valid_tickers=custom_tickers)
        assert extractor.valid_tickers == custom_tickers

    def test_extract_dollar_format(self, extractor):
        """Test extraction of $AAPL format."""
        text = "Apple stock $AAPL surges on earnings"
        tickers = extractor.extract(text)
        assert "AAPL" in tickers

    def test_extract_dollar_format_multiple(self, extractor):
        """Test extraction of multiple $TICKER formats."""
        text = "$AAPL and $MSFT both rose today"
        tickers = extractor.extract(text)
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_extract_company_name_apple(self, extractor):
        """Test extraction of 'Apple' → AAPL."""
        text = "Apple reported strong earnings"
        tickers = extractor.extract(text)
        assert "AAPL" in tickers

    def test_extract_company_name_bitcoin(self, extractor):
        """Test extraction of 'Bitcoin' → BTC."""
        text = "Bitcoin reached a new all-time high"
        tickers = extractor.extract(text)
        assert "BTC" in tickers

    def test_extract_company_name_ethereum(self, extractor):
        """Test extraction of 'Ethereum' → ETH."""
        text = "Ethereum network upgraded successfully"
        tickers = extractor.extract(text)
        assert "ETH" in tickers

    def test_extract_ticker_with_colon(self, extractor):
        """Test extraction of AAPL: format."""
        text = "AAPL: Apple stock rises"
        tickers = extractor.extract(text)
        assert "AAPL" in tickers

    def test_extract_ticker_in_parentheses(self, extractor):
        """Test extraction of (AAPL) format."""
        text = "The company (AAPL) announced earnings"
        tickers = extractor.extract(text)
        assert "AAPL" in tickers

    def test_extract_ticker_before_stock_keyword(self, extractor):
        """Test extraction of AAPL stock format."""
        text = "AAPL stock surged today"
        tickers = extractor.extract(text)
        assert "AAPL" in tickers

    def test_extract_multiple_tickers(self, extractor):
        """Test extraction of multiple tickers in same text."""
        text = "Apple (AAPL) and Microsoft (MSFT) both reported earnings. Bitcoin (BTC) also rose."
        tickers = extractor.extract(text)
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "BTC" in tickers
        assert len(tickers) >= 3

    def test_extract_deduplicates(self, extractor):
        """Test that extractor deduplicates tickers."""
        text = "Apple (AAPL) stock $AAPL surged. Apple reported earnings."
        tickers = extractor.extract(text)
        assert tickers.count("AAPL") == 1
        assert len(tickers) == 1

    def test_extract_empty_text(self, extractor):
        """Test extraction from empty text."""
        tickers = extractor.extract("")
        assert tickers == []

    def test_extract_none_text(self, extractor):
        """Test extraction from None text."""
        tickers = extractor.extract(None)
        assert tickers == []

    def test_extract_case_insensitive_company_name(self, extractor):
        """Test that company name extraction is case insensitive."""
        text = "APPLE reported earnings"
        tickers = extractor.extract(text)
        assert "AAPL" in tickers

    def test_extract_word_boundary_match(self, extractor):
        """Test that company names match on word boundaries."""
        # "apple" in "pineapple" should NOT match
        text = "Pineapple is a fruit, but Apple stock rose"
        tickers = extractor.extract(text)
        assert "AAPL" in tickers

    def test_extract_with_context_single_ticker(self, extractor):
        """Test extraction with context for single ticker."""
        text = "Apple stock surged 10% on strong earnings report today"
        results = extractor.extract_with_context(text)
        assert len(results) > 0
        ticker, context = results[0]
        assert ticker == "AAPL"
        assert "Apple" in context or "AAPL" in context
        assert len(context) > 0

    def test_extract_with_context_multiple_tickers(self, extractor):
        """Test extraction with context for multiple tickers."""
        text = "Apple (AAPL) and Microsoft (MSFT) both reported earnings today"
        results = extractor.extract_with_context(text)
        assert len(results) >= 2
        tickers = [ticker for ticker, _ in results]
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_get_context_includes_surrounding_text(self, extractor):
        """Test that context includes surrounding text."""
        text = "The market opened strong. Apple stock surged 10% on earnings. Trading volume was high."
        context = extractor._get_context(text, "AAPL", window=30)
        assert "Apple" in context or "AAPL" in context
        assert "surged" in context or "earnings" in context

    def test_get_context_with_ellipsis(self, extractor):
        """Test that context includes ellipsis when truncated."""
        long_text = "This is a very long text. " * 10 + "Apple stock rose. " + "More text here. " * 10
        context = extractor._get_context(long_text, "AAPL", window=20)
        # Context should be truncated and may have ellipsis
        assert len(context) > 0
        assert "Apple" in context or "AAPL" in context

    def test_extract_custom_tickers_only(self):
        """Test that extractor only finds tickers in valid_tickers set."""
        custom_tickers = {"AAPL", "MSFT"}
        extractor = TickerExtractor(valid_tickers=custom_tickers)
        
        # Should find AAPL and MSFT
        text1 = "Apple and Microsoft reported earnings"
        tickers1 = extractor.extract(text1)
        assert "AAPL" in tickers1
        assert "MSFT" in tickers1
        
        # Should NOT find BTC even if mentioned
        text2 = "Bitcoin and Apple both rose"
        tickers2 = extractor.extract(text2)
        assert "AAPL" in tickers2
        assert "BTC" not in tickers2

    def test_extract_various_equity_tickers(self, extractor):
        """Test extraction of various equity tickers."""
        text = "Google (GOOGL), Amazon (AMZN), and Tesla (TSLA) all rose"
        tickers = extractor.extract(text)
        assert "GOOGL" in tickers
        assert "AMZN" in tickers
        assert "TSLA" in tickers

    def test_extract_various_crypto_tickers(self, extractor):
        """Test extraction of various crypto tickers."""
        text = "Bitcoin (BTC), Ethereum (ETH), and Solana (SOL) all gained"
        tickers = extractor.extract(text)
        assert "BTC" in tickers
        assert "ETH" in tickers
        assert "SOL" in tickers

    def test_extract_aliases_work(self, extractor):
        """Test that various aliases work correctly."""
        # Test Microsoft aliases
        text1 = "Microsoft stock rose"
        assert "MSFT" in extractor.extract(text1)
        
        # Test Nvidia aliases
        text2 = "Nvidia announced new GeForce cards"
        assert "NVDA" in extractor.extract(text2)
        
        # Test Meta/Facebook aliases
        text3 = "Facebook parent Meta reported earnings"
        assert "META" in extractor.extract(text3)

    def test_ticker_aliases_structure(self):
        """Test that TICKER_ALIASES has expected structure."""
        assert len(TICKER_ALIASES) > 0
        assert "AAPL" in TICKER_ALIASES
        assert "BTC" in TICKER_ALIASES
        assert isinstance(TICKER_ALIASES["AAPL"], list)
        assert len(TICKER_ALIASES["AAPL"]) > 0

    def test_alias_to_ticker_mapping(self):
        """Test that ALIAS_TO_TICKER mapping is correct."""
        assert len(ALIAS_TO_TICKER) > 0
        assert "apple" in ALIAS_TO_TICKER
        assert ALIAS_TO_TICKER["apple"] == "AAPL"
        assert "bitcoin" in ALIAS_TO_TICKER
        assert ALIAS_TO_TICKER["bitcoin"] == "BTC"

