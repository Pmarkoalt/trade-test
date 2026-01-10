"""Equity universe selection and management.

Supports:
- SP500 universe (from file or hardcoded list)
- NASDAQ-100 universe
- Custom equity universes
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# Hardcoded SP500 subset for MVP (top liquid names)
# In production, this should be loaded from a file or API
SP500_CORE_UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE",
    # Consumer
    "WMT", "HD", "MCD", "NKE", "COST", "SBUX",
    # Industrials
    "BA", "CAT", "UPS", "HON", "GE",
    # Energy
    "XOM", "CVX", "COP",
    # Communications
    "DIS", "NFLX", "CMCSA",
    # Materials
    "LIN", "APD",
]


def load_sp500_universe(universe_file: Optional[str] = None) -> List[str]:
    """Load SP500 universe from file or use hardcoded list.
    
    Args:
        universe_file: Optional path to CSV file with SP500 symbols
                      Expected format: CSV with 'symbol' or 'ticker' column
    
    Returns:
        List of SP500 symbols
    """
    if universe_file:
        file_path = Path(universe_file)
        if not file_path.exists():
            logger.warning(f"SP500 universe file not found: {universe_file}, using hardcoded list")
            return SP500_CORE_UNIVERSE.copy()
        
        try:
            df = pd.read_csv(file_path)
            
            # Try common column names
            symbol_col = None
            for col in ['symbol', 'Symbol', 'ticker', 'Ticker', 'SYMBOL', 'TICKER']:
                if col in df.columns:
                    symbol_col = col
                    break
            
            if symbol_col is None:
                # Use first column
                symbol_col = df.columns[0]
                logger.info(f"Using first column '{symbol_col}' as symbol column")
            
            symbols = df[symbol_col].dropna().astype(str).str.upper().str.strip().tolist()
            
            logger.info(f"Loaded {len(symbols)} symbols from {universe_file}")
            return symbols
            
        except Exception as e:
            logger.error(f"Error loading SP500 universe from {universe_file}: {e}")
            logger.warning("Falling back to hardcoded SP500 core universe")
            return SP500_CORE_UNIVERSE.copy()
    
    else:
        logger.info(f"Using hardcoded SP500 core universe ({len(SP500_CORE_UNIVERSE)} symbols)")
        return SP500_CORE_UNIVERSE.copy()


def filter_universe_by_data_availability(
    universe: List[str], 
    available_data: Dict[str, pd.DataFrame],
    min_bars: int = 200
) -> List[str]:
    """Filter universe to only symbols with sufficient data.
    
    Args:
        universe: List of symbols to filter
        available_data: Dictionary mapping symbol -> DataFrame
        min_bars: Minimum number of bars required
    
    Returns:
        Filtered list of symbols
    """
    filtered = []
    
    for symbol in universe:
        if symbol in available_data:
            df = available_data[symbol]
            if len(df) >= min_bars:
                filtered.append(symbol)
            else:
                logger.debug(f"Excluding {symbol}: only {len(df)} bars (need {min_bars})")
        else:
            logger.debug(f"Excluding {symbol}: no data available")
    
    logger.info(f"Filtered universe: {len(filtered)}/{len(universe)} symbols have sufficient data")
    
    return filtered


def select_equity_universe(
    universe_type: str = "SP500",
    universe_file: Optional[str] = None,
    available_data: Optional[Dict[str, pd.DataFrame]] = None,
    min_bars: int = 200
) -> List[str]:
    """Select equity universe based on type.
    
    Args:
        universe_type: Type of universe ("SP500", "NASDAQ-100", "custom")
        universe_file: Path to universe file (for custom or to override defaults)
        available_data: Optional data dictionary to filter by availability
        min_bars: Minimum bars required per symbol
    
    Returns:
        List of selected symbols
    """
    if universe_type == "SP500":
        universe = load_sp500_universe(universe_file)
    elif universe_type == "NASDAQ-100":
        # For now, use SP500 core (can be extended later)
        logger.warning("NASDAQ-100 not yet implemented, using SP500 core universe")
        universe = load_sp500_universe(universe_file)
    elif universe_type == "custom":
        if not universe_file:
            raise ValueError("custom universe_type requires universe_file")
        universe = load_sp500_universe(universe_file)
    else:
        raise ValueError(f"Unknown universe_type: {universe_type}")
    
    # Filter by data availability if provided
    if available_data is not None:
        universe = filter_universe_by_data_availability(universe, available_data, min_bars)
    
    return universe
