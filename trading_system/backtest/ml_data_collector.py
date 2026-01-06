"""ML Data Collector for backtest feature accumulation.

This module handles:
1. Feature extraction when signals are generated
2. Storage of feature vectors with signal metadata
3. Outcome labeling when trades close (R-multiple, win/loss)
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from ..ml.feature_engineering import MLFeatureEngineer
from ..ml_refinement.config import FeatureVector
from ..ml_refinement.storage.feature_db import FeatureDatabase
from ..models.features import FeatureRow
from ..models.positions import Position
from ..models.signals import Signal


class MLDataCollector:
    """Collects and stores ML training data during backtests.
    
    This class bridges the gap between the backtest engine and the ML
    training pipeline by:
    1. Extracting features when signals are generated
    2. Storing feature vectors in the feature database
    3. Updating targets when trades close with actual outcomes
    
    Example:
        collector = MLDataCollector(db_path="features.db")
        
        # On signal generation
        signal_id = collector.record_signal(signal, features)
        
        # On trade close
        collector.record_outcome(signal_id, r_multiple, return_pct)
    """
    
    def __init__(
        self,
        db_path: str = "features.db",
        enabled: bool = True,
        feature_engineer: Optional[MLFeatureEngineer] = None,
    ):
        """Initialize ML data collector.
        
        Args:
            db_path: Path to SQLite feature database
            enabled: Whether data collection is enabled
            feature_engineer: Optional pre-configured feature engineer
        """
        self.enabled = enabled
        self.db_path = db_path
        self._db: Optional[FeatureDatabase] = None
        
        # Feature engineer for extracting ML features
        self.feature_engineer = feature_engineer or MLFeatureEngineer(
            include_raw_features=True,
            include_derived_features=True,
            include_technical_indicators=True,
            include_volume_indicators=True,
            include_volatility_features=True,
            include_market_regime_features=True,
            include_cross_asset_features=True,
            normalize_features=False,  # Don't normalize during collection
        )
        
        # Track signal_id to position mapping
        self._signal_to_position: Dict[str, str] = {}  # signal_id -> position_key
        self._position_to_signal: Dict[str, str] = {}  # position_key -> signal_id
        
        # Statistics
        self.signals_recorded = 0
        self.outcomes_recorded = 0
        
    @property
    def db(self) -> FeatureDatabase:
        """Get or create database connection."""
        if self._db is None:
            self._db = FeatureDatabase(self.db_path)
            # Initialize database if needed
            self._initialize_db()
        return self._db
    
    def _initialize_db(self) -> None:
        """Initialize database schema if needed."""
        try:
            self._db.initialize()
            logger.info(f"ML data collector initialized with database: {self.db_path}")
        except Exception as e:
            logger.warning(f"Database initialization warning (may already exist): {e}")
    
    def record_signal(
        self,
        signal: Signal,
        features: FeatureRow,
        position_key: Optional[str] = None,
    ) -> Optional[str]:
        """Record a signal and its features for ML training.
        
        Called when a signal is generated and an order is created.
        
        Args:
            signal: The trading signal that was generated
            features: FeatureRow at the time of signal generation
            position_key: Optional position key for tracking (symbol or (asset_class, symbol))
            
        Returns:
            signal_id if recorded, None if disabled or error
        """
        if not self.enabled:
            return None
            
        try:
            # Generate unique signal ID
            signal_id = str(uuid.uuid4())
            
            # Extract ML features from FeatureRow
            feature_dict = self._extract_features(features, signal)
            
            # Create feature vector
            fv = FeatureVector(
                signal_id=signal_id,
                timestamp=signal.date.isoformat() if hasattr(signal.date, 'isoformat') else str(signal.date),
                features=feature_dict,
                target=None,  # Will be set when trade closes
                target_binary=None,
            )
            
            # Store in database
            self.db.store_feature_vector(
                fv=fv,
                symbol=signal.symbol,
                asset_class=signal.asset_class,
                signal_type=signal.triggered_on.value if signal.triggered_on else "unknown",
                conviction=str(signal.score) if signal.score else "0.0",
            )
            
            # Track position mapping
            if position_key:
                self._signal_to_position[signal_id] = position_key
                self._position_to_signal[position_key] = signal_id
            else:
                # Default position key is just the symbol
                self._signal_to_position[signal_id] = signal.symbol
                self._position_to_signal[signal.symbol] = signal_id
            
            self.signals_recorded += 1
            logger.debug(f"Recorded signal {signal_id[:8]} for {signal.symbol} at {signal.date}")
            
            return signal_id
            
        except Exception as e:
            logger.error(f"Error recording signal for {signal.symbol}: {e}", exc_info=True)
            return None
    
    def record_outcome(
        self,
        position: Position,
        position_key: Optional[str] = None,
    ) -> bool:
        """Record the outcome of a closed trade.
        
        Called when a position is closed to update the feature vector
        with the actual R-multiple and win/loss outcome.
        
        Args:
            position: The closed Position object
            position_key: Optional position key used during signal recording
            
        Returns:
            True if outcome was recorded, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            # Find signal_id for this position
            key = position_key or position.symbol
            signal_id = self._position_to_signal.get(key)
            
            if signal_id is None:
                logger.debug(f"No signal_id found for position {key}")
                return False
            
            # Calculate R-multiple
            r_multiple = position.compute_r_multiple()
            if r_multiple is None:
                # Fallback calculation
                if position.initial_stop_price and position.entry_price:
                    risk = abs(position.entry_price - position.initial_stop_price)
                    if risk > 0 and position.exit_price:
                        pnl_per_share = position.exit_price - position.entry_price
                        r_multiple = pnl_per_share / risk
            
            # Calculate return percentage
            return_pct = None
            if position.entry_price and position.exit_price and position.entry_price > 0:
                return_pct = (position.exit_price - position.entry_price) / position.entry_price
            
            # Update target in database
            if r_multiple is not None:
                success = self.db.update_target(
                    signal_id=signal_id,
                    r_multiple=float(r_multiple),
                    return_pct=float(return_pct) if return_pct else None,
                )
                
                if success:
                    self.outcomes_recorded += 1
                    logger.debug(
                        f"Recorded outcome for {position.symbol}: R={r_multiple:.2f}, "
                        f"return={return_pct*100:.2f}%" if return_pct else f"R={r_multiple:.2f}"
                    )
                    
                    # Clean up tracking
                    del self._position_to_signal[key]
                    del self._signal_to_position[signal_id]
                    
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error recording outcome for {position.symbol}: {e}", exc_info=True)
            return False
    
    def _extract_features(self, features: FeatureRow, signal: Signal) -> Dict[str, float]:
        """Extract ML features from FeatureRow and Signal.
        
        Args:
            features: FeatureRow at signal time
            signal: The generated signal
            
        Returns:
            Dictionary of feature name -> value
        """
        feature_dict: Dict[str, float] = {}
        
        # Extract features using the feature engineer
        try:
            feature_series = self.feature_engineer.transform(features)
            feature_dict = feature_series.to_dict()
        except Exception as e:
            logger.warning(f"Feature engineer transform failed: {e}, using manual extraction")
        
        # Add signal-specific features
        feature_dict["signal_score"] = float(signal.score) if signal.score else 0.0
        feature_dict["breakout_clearance"] = float(signal.breakout_clearance) if signal.breakout_clearance else 0.0
        feature_dict["breakout_strength"] = float(signal.breakout_strength) if signal.breakout_strength else 0.0
        feature_dict["momentum_strength"] = float(signal.momentum_strength) if signal.momentum_strength else 0.0
        feature_dict["diversification_bonus"] = float(signal.diversification_bonus) if signal.diversification_bonus else 0.0
        feature_dict["capacity_passed"] = 1.0 if signal.capacity_passed else 0.0
        
        # Add raw price features
        feature_dict["entry_price"] = float(signal.entry_price) if signal.entry_price else 0.0
        feature_dict["stop_price"] = float(signal.stop_price) if signal.stop_price else 0.0
        
        # Risk-reward ratio
        if signal.entry_price and signal.stop_price and signal.stop_price > 0:
            risk = abs(signal.entry_price - signal.stop_price)
            if risk > 0:
                feature_dict["risk_amount"] = float(risk)
        
        # Add trigger type as one-hot encoded features
        trigger_types = ["fast_breakout", "slow_breakout", "both"]
        for tt in trigger_types:
            feature_dict[f"trigger_{tt}"] = 1.0 if signal.triggered_on and signal.triggered_on.value == tt else 0.0
        
        # Filter out any NaN values (replace with 0)
        import math
        feature_dict = {
            k: 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
            for k, v in feature_dict.items()
        }
        
        return feature_dict
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics.
        
        Returns:
            Dictionary with collection stats
        """
        stats = {
            "enabled": self.enabled,
            "signals_recorded": self.signals_recorded,
            "outcomes_recorded": self.outcomes_recorded,
            "pending_outcomes": len(self._position_to_signal),
        }
        
        if self.enabled and self._db:
            try:
                stats["total_samples"] = self.db.count_samples(require_target=False)
                stats["labeled_samples"] = self.db.count_samples(require_target=True)
            except Exception:
                pass
        
        return stats
    
    def close(self) -> None:
        """Close database connection and log statistics."""
        if self._db:
            stats = self.get_statistics()
            logger.info(
                f"ML Data Collector closed: {stats['signals_recorded']} signals, "
                f"{stats['outcomes_recorded']} outcomes, "
                f"{stats.get('total_samples', 'N/A')} total samples in DB"
            )
            self._db.close()
            self._db = None
