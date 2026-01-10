"""Strategy bucket implementations for daily signal generation.

Buckets:
- Bucket A: Safe S&P (conservative equity strategy)
- Bucket B: Top-cap crypto (aggressive crypto strategy)
"""

from .bucket_a_safe_sp import SafeSPStrategy
from .bucket_b_crypto_topcat import TopCapCryptoStrategy

__all__ = [
    "SafeSPStrategy",
    "TopCapCryptoStrategy",
]
