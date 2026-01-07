"""Reusable daily recommendation generation utilities.

This module provides a single, reusable entry point for producing daily
recommendations that can be called from:
- FastAPI endpoint (/signals/daily)
- Scheduler jobs (daily_signals_job)
- CLI scripts / adhoc runs

It intentionally returns structured data (recommendations + metadata) and
lets the caller decide how to deliver it (email, Slack, DB, etc.).
"""

import os
from dataclasses import asdict
from datetime import date as date_type
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..configs.run_config import RunConfig
from ..data.loader import load_universe
from ..data_pipeline.config import DataPipelineConfig
from ..data_pipeline.live_data_fetcher import LiveDataFetcher
from ..research.config import ResearchConfig
from ..research.news_analyzer import NewsAnalyzer
from ..signals.config import SignalConfig
from ..signals.live_signal_generator import LiveSignalGenerator
from ..strategies.strategy_loader import load_strategies_from_run_config


def _recommendation_to_dict(rec) -> Dict[str, Any]:
    # Recommendation is a dataclass
    d = asdict(rec)
    # Normalize datetime fields
    if "generated_at" in d and hasattr(d["generated_at"], "isoformat"):
        d["generated_at"] = d["generated_at"].isoformat()
    return d


def _resolve_config_path(path: Optional[str]) -> Optional[str]:
    """Resolve a config path that may be repo-relative into a container path.

    This is needed in Docker where the host ./configs directory is mounted at
    /app/custom_configs (see docker-compose.n8n.yml).
    """
    if not path:
        return None

    p = Path(path)
    if p.exists():
        return str(p)

    basename = p.name
    candidates = [
        Path("/app/custom_configs") / basename,
        Path("/app/configs") / basename,
        Path("configs") / basename,
        Path("custom_configs") / basename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return path


async def generate_daily_recommendations(
    *,
    asset_class: str,
    universe: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    lookback_days: int = 252,
    news_lookback_hours: int = 168,
    max_recommendations: int = 5,
    min_conviction: str = "MEDIUM",
    run_config_path: Optional[str] = None,
    current_date: Optional[date_type] = None,
    debug: bool = False,
    include_candidates: bool = False,
    candidate_limit: int = 25,
    candidate_news: bool = True,
) -> Dict[str, Any]:
    """Generate daily recommendations with live OHLCV + live news.

    This function expects API keys via environment variables:
    - MASSIVE_API_KEY (Polygon)
    - ALPHA_VANTAGE_API_KEY (AlphaVantage)
    - NEWSAPI_KEY (optional)

    Args:
        asset_class: "equity" or "crypto"
        universe: Universe name, e.g. "NASDAQ-100", "SP500", "crypto". If None, defaults by asset_class.
        lookback_days: OHLCV lookback days for feature computation
        news_lookback_hours: News lookback hours for sentiment
        max_recommendations: Max recommendations to return
        min_conviction: "LOW"/"MEDIUM"/"HIGH"
        run_config_path: Optional run config path. Defaults to RUN_CONFIG_PATH env var.
        current_date: Optional date override.

    Returns:
        Dict containing metadata + recommendations (as plain dicts).
    """
    asset_class = asset_class.lower()
    if asset_class not in {"equity", "crypto"}:
        raise ValueError("asset_class must be 'equity' or 'crypto'")

    if current_date is None:
        current_date = date_type.today()

    # Strategies
    config_path = run_config_path or os.getenv("RUN_CONFIG_PATH")
    if not config_path:
        # Prefer docker-mounted custom_configs path when running in the n8n stack
        docker_default = "/app/custom_configs/production_run_config.yaml"
        repo_default = "configs/production_run_config.yaml"
        config_path = docker_default if Path(docker_default).exists() else repo_default

    # If user provided RUN_CONFIG_PATH but it doesn't exist in container, try common docker mount locations.
    p = Path(config_path)
    if not p.exists():
        basename = p.name
        candidates = [
            Path("/app/custom_configs") / basename,
            Path("/app/configs") / basename,
            Path("configs") / basename,
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = str(candidate)
                break

    if not Path(config_path).exists():
        raise ValueError(f"Run config not found: {config_path}")

    run_config = RunConfig.from_yaml(config_path)
    equity_config_path = (
        run_config.strategies.equity.config_path
        if run_config.strategies.equity and run_config.strategies.equity.enabled
        else None
    )
    crypto_config_path = (
        run_config.strategies.crypto.config_path
        if run_config.strategies.crypto and run_config.strategies.crypto.enabled
        else None
    )

    equity_config_path = _resolve_config_path(equity_config_path)
    crypto_config_path = _resolve_config_path(crypto_config_path)

    strategies = load_strategies_from_run_config(
        equity_config_path=equity_config_path if asset_class == "equity" else None,
        crypto_config_path=crypto_config_path if asset_class == "crypto" else None,
    )

    # Universe / symbols
    if symbols is not None and len(symbols) > 0:
        symbols = [s.upper() for s in symbols]
        universe_type = "custom"
    else:
        universe_type = universe or ("NASDAQ-100" if asset_class == "equity" else "crypto")
        try:
            symbols = load_universe(universe_type)
        except Exception:
            symbols = []

        # Fallback: if universe CSV is missing in Docker, use the strategy's universe field
        if not symbols:
            fallback_universe = None
            if asset_class == "equity" and run_config.strategies.equity and run_config.strategies.equity.enabled:
                fallback_universe = run_config.strategies.equity.universe
            if asset_class == "crypto" and run_config.strategies.crypto and run_config.strategies.crypto.enabled:
                fallback_universe = run_config.strategies.crypto.universe

            if isinstance(fallback_universe, list) and fallback_universe:
                symbols = [s.upper() for s in fallback_universe]
                universe_type = "strategy_universe"

    if not symbols:
        raise ValueError(
            f"No symbols available. Provide request.symbols explicitly or ensure universe files are present for universe={universe_type}."
        )

    # Ensure strategies use an explicit list universe in live mode.
    # Some configs set universe as a string (e.g. "NASDAQ-100"); in that case,
    # the TechnicalSignalGenerator's `symbol in strategy.universe` check will fail.
    for strategy in strategies:
        if isinstance(strategy.universe, str):
            strategy.universe = list(symbols)

    # Data fetcher
    dp_config = DataPipelineConfig(
        massive_api_key=os.getenv("MASSIVE_API_KEY"),
        alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
        cache_path=Path(os.getenv("DATA_CACHE_PATH", "data/cache")),
        cache_ttl_hours=int(os.getenv("CACHE_TTL_HOURS", "24")),
    )
    data_fetcher = LiveDataFetcher(dp_config)

    # News analyzer
    news_analyzer: Optional[NewsAnalyzer] = None
    try:
        research_config = ResearchConfig(
            enabled=True,
            newsapi_key=os.getenv("NEWSAPI_KEY"),
            alpha_vantage_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
            massive_api_key=os.getenv("MASSIVE_API_KEY"),
            lookback_hours=news_lookback_hours,
            max_articles_per_symbol=int(os.getenv("MAX_ARTICLES_PER_SYMBOL", "10")),
        )
        news_analyzer = NewsAnalyzer(research_config)
    except Exception:
        news_analyzer = None

    # Signal generator
    signal_config = SignalConfig(
        max_recommendations=max_recommendations,
        min_conviction=min_conviction,
        news_enabled=True,
        news_lookback_hours=news_lookback_hours,
    )

    signal_generator = LiveSignalGenerator(
        strategies=strategies,
        signal_config=signal_config,
        news_analyzer=news_analyzer,
        tracking_db=None,
    )

    # OHLCV
    async with data_fetcher:
        ohlcv_data: Dict[str, pd.DataFrame] = await data_fetcher.fetch_daily_data(
            symbols=symbols,
            asset_class=asset_class,
            lookback_days=lookback_days,
        )

    # Normalize OHLCV frames to match indicator expectations:
    # - DatetimeIndex
    # - columns: open, high, low, close, volume
    normalized_ohlcv: Dict[str, pd.DataFrame] = {}
    for sym, df in ohlcv_data.items():
        if df is None or df.empty:
            continue

        df_norm = df.copy()

        # If the data comes with a 'date' column (common for live fetchers), promote it to index
        if "date" in df_norm.columns:
            df_norm["date"] = pd.to_datetime(df_norm["date"])
            df_norm = df_norm.set_index("date")

        # Drop redundant symbol column if present
        if "symbol" in df_norm.columns:
            df_norm = df_norm.drop(columns=["symbol"])

        df_norm = df_norm.sort_index()

        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(c in df_norm.columns for c in required_cols):
            continue

        normalized_ohlcv[sym] = df_norm

    ohlcv_data = normalized_ohlcv

    # Choose a usable signal date from the data (avoids empty results on non-trading days)
    used_date = current_date
    try:
        latest_dates = []
        for df in ohlcv_data.values():
            if df is None or df.empty:
                continue
            if len(df.index) == 0:
                continue
            latest_dates.append(pd.Timestamp(df.index[-1]).date())
        if latest_dates:
            used_date = max(latest_dates)
    except Exception:
        used_date = current_date

    recommendations = await signal_generator.generate_recommendations(
        ohlcv_data=ohlcv_data,
        current_date=used_date,
        portfolio_state=None,
    )

    candidates: Optional[List[Dict[str, Any]]] = None
    if include_candidates:
        candidates = []
        try:
            # Build candidate list based on eligibility and proximity to breakout levels
            from ..indicators.feature_computer import compute_features, compute_features_for_date

            # Optionally pre-fetch news analysis once (for the subset we return)
            news_analysis = None
            if candidate_news and news_analyzer is not None:
                try:
                    news_analysis = await news_analyzer.analyze_symbols(
                        symbols=list(symbols),
                        lookback_hours=news_lookback_hours,
                    )
                except Exception:
                    news_analysis = None

            for sym in symbols:
                if sym not in ohlcv_data:
                    continue

                df = ohlcv_data[sym]
                if df is None or df.empty:
                    continue

                asset = "equity"
                for st in strategies:
                    if sym in st.universe:
                        asset = st.asset_class
                        break

                feats_df = compute_features(df, symbol=sym, asset_class=asset, use_cache=False, optimize_memory=False)
                feat_row = compute_features_for_date(feats_df, pd.Timestamp(used_date))
                if feat_row is None:
                    continue

                # Distances to breakout levels
                dist_20 = None
                dist_55 = None
                if feat_row.highest_close_20d is not None and feat_row.highest_close_20d > 0:
                    dist_20 = float((feat_row.highest_close_20d - feat_row.close) / feat_row.highest_close_20d)
                if feat_row.highest_close_55d is not None and feat_row.highest_close_55d > 0:
                    dist_55 = float((feat_row.highest_close_55d - feat_row.close) / feat_row.highest_close_55d)

                # Evaluate first matching strategy (for reasons)
                eligible = None
                eligibility_reasons: List[str] = []
                has_trigger = None
                trigger_type = None
                clearance = 0.0

                for st in strategies:
                    if st.asset_class != asset:
                        continue
                    if sym not in st.universe:
                        continue
                    try:
                        eligible, eligibility_reasons = st.check_eligibility(feat_row)
                    except Exception as e:
                        eligible, eligibility_reasons = False, [f"eligibility_error: {e}"]
                    try:
                        breakout_type, clearance = st.check_entry_triggers(feat_row)
                        has_trigger = breakout_type is not None
                        trigger_type = (
                            breakout_type.value
                            if breakout_type is not None and hasattr(breakout_type, "value")
                            else str(breakout_type)
                        )
                    except Exception as e:
                        has_trigger = False
                        trigger_type = None
                        eligibility_reasons = list(eligibility_reasons) + [f"trigger_error: {e}"]
                    break

                # Candidate score: closer to breakout + eligibility bonus
                proximity = None
                if dist_20 is not None and dist_55 is not None:
                    proximity = min(dist_20, dist_55)
                elif dist_20 is not None:
                    proximity = dist_20
                elif dist_55 is not None:
                    proximity = dist_55

                if proximity is None:
                    continue

                candidate_score = float(max(0.0, 1.0 - proximity))
                if eligible:
                    candidate_score += 0.10
                if has_trigger:
                    candidate_score += 0.25

                item: Dict[str, Any] = {
                    "symbol": sym,
                    "asset_class": asset,
                    "close": float(feat_row.close),
                    "ma50": float(feat_row.ma50) if feat_row.ma50 is not None else None,
                    "roc60": float(feat_row.roc60) if feat_row.roc60 is not None else None,
                    "dist_to_20d_high": dist_20,
                    "dist_to_55d_high": dist_55,
                    "eligible": bool(eligible) if eligible is not None else None,
                    "eligibility_reasons": eligibility_reasons,
                    "has_trigger": bool(has_trigger) if has_trigger is not None else None,
                    "trigger_type": trigger_type,
                    "clearance": float(clearance),
                    "candidate_score": candidate_score,
                }

                # Optional news enrichment
                if news_analysis is not None:
                    try:
                        score, reasoning = news_analyzer.get_news_score_for_signal(sym, news_analysis)  # type: ignore[union-attr]
                        item["news_score"] = float(score)
                        item["news_reasoning"] = reasoning
                        summary = getattr(news_analysis, "symbol_summaries", {}).get(sym)
                        if summary is not None:
                            item["news_headlines"] = getattr(summary, "top_headlines", [])
                            item["news_sentiment_label"] = getattr(getattr(summary, "sentiment_label", None), "value", None)
                            item["news_article_count"] = getattr(summary, "article_count", None)
                    except Exception:
                        pass

                candidates.append(item)

            candidates.sort(key=lambda x: x.get("candidate_score", 0.0), reverse=True)
            candidates = candidates[: max(1, int(candidate_limit))]
        except Exception:
            candidates = []

    debug_payload: Optional[Dict[str, Any]] = None
    if debug:
        debug_payload = {
            "used_date": str(used_date),
            "symbols_requested": list(symbols),
            "ohlcv_symbols_available": list(ohlcv_data.keys()),
            "per_symbol": {},
        }
        try:
            from ..indicators.feature_computer import compute_features, compute_features_for_date

            for sym in symbols:
                item: Dict[str, Any] = {"has_ohlcv": sym in ohlcv_data}
                if sym not in ohlcv_data:
                    debug_payload["per_symbol"][sym] = item
                    continue

                df = ohlcv_data[sym]
                item["rows"] = int(len(df))
                item["last_index"] = str(df.index[-1]) if len(df.index) else None

                asset = "equity"
                for st in strategies:
                    if sym in st.universe:
                        asset = st.asset_class
                        break
                item["asset_class"] = asset

                try:
                    feats_df = compute_features(df, symbol=sym, asset_class=asset, use_cache=False, optimize_memory=False)
                    feat_row = compute_features_for_date(feats_df, pd.Timestamp(used_date))
                except Exception as e:
                    item["feature_error"] = str(e)
                    debug_payload["per_symbol"][sym] = item
                    continue

                item["feature_row_found"] = feat_row is not None
                if feat_row is None:
                    debug_payload["per_symbol"][sym] = item
                    continue

                item["is_valid_for_entry"] = bool(feat_row.is_valid_for_entry())

                # Evaluate each strategy for eligibility/trigger
                strat_details = []
                for st in strategies:
                    if st.asset_class != asset:
                        continue
                    if sym not in st.universe:
                        continue
                    try:
                        eligible, reasons = st.check_eligibility(feat_row)
                    except Exception as e:
                        eligible, reasons = False, [f"eligibility_error: {e}"]
                    try:
                        breakout_type, clearance = st.check_entry_triggers(feat_row)
                        has_trigger = breakout_type is not None
                    except Exception as e:
                        breakout_type, clearance, has_trigger = None, 0.0, False
                        reasons = list(reasons) + [f"trigger_error: {e}"]

                    strat_details.append(
                        {
                            "strategy": st.name,
                            "eligible": bool(eligible),
                            "eligibility_reasons": reasons,
                            "has_trigger": bool(has_trigger),
                            "trigger_type": breakout_type.value
                            if breakout_type is not None and hasattr(breakout_type, "value")
                            else str(breakout_type),
                            "clearance": float(clearance),
                        }
                    )

                item["strategies"] = strat_details
                debug_payload["per_symbol"][sym] = item
        except Exception as e:
            debug_payload["error"] = str(e)

    return {
        "status": "ok",
        "asset_class": asset_class,
        "universe": universe_type,
        "symbol_count": len(symbols),
        "recommendation_count": len(recommendations),
        "used_date": str(used_date),
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "recommendations": [_recommendation_to_dict(r) for r in recommendations],
        "candidates": candidates,
        "debug": debug_payload,
    }
