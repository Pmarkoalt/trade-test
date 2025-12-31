"""Standardized signal object for strategies.

This module re-exports Signal and related types from models for convenience.
Strategies should import from here or directly from models.
"""

# Re-export from models (signal is a data model used throughout the system)
from ...models.signals import BreakoutType, Signal, SignalSide

__all__ = ["Signal", "SignalSide", "BreakoutType"]
