from __future__ import annotations

import time
import random
from typing import Dict, List, Tuple, Optional


__all__ = ["BudgetTimer", "select_candidates", "rng_from_seed"]


class BudgetTimer:
    """
    Enforce a wall-clock budget in milliseconds.
    - deadline: absolute perf_counter deadline (seconds)
    - time_left_ms(): int milliseconds remaining (0 if expired)
    - expired(): bool
    """
    def __init__(self, max_ms_budget: int) -> None:
        try:
            ms = int(max_ms_budget)
        except Exception:
            ms = 0
        ms = max(0, ms)
        self.start_s: float = time.perf_counter()
        # If budget is 0, we treat it as "no budget limit"
        self._has_budget: bool = ms > 0
        self.deadline: float = self.start_s + (ms / 1000.0 if self._has_budget else 0.0)

    def time_left_ms(self) -> int:
        if not self._has_budget:
            # effectively infinite
            return 2**31 - 1
        remaining = (self.deadline - time.perf_counter()) * 1000.0
        return int(remaining) if remaining > 0 else 0

    def expired(self) -> bool:
        if not self._has_budget:
            return False
        return time.perf_counter() >= self.deadline


def select_candidates(features: Dict[str, float], max_features: int = 512) -> List[Tuple[str, float]]:
    """
    Return features sorted by activation strength (descending), capped to protect runtime.
    """
    try:
        mf = int(max_features)
    except Exception:
        mf = 512
    mf = max(1, mf)
    items = list(features.items())
    # sort descending by activation strength
    items.sort(key=lambda kv: float(kv[1]), reverse=True)
    return items[:mf]


def rng_from_seed(seed: Optional[int]) -> random.Random:
    """
    Deterministic RNG when seed is provided, otherwise a fresh Random instance.
    """
    if seed is None:
        return random.Random()
    try:
        return random.Random(int(seed))
    except Exception:
        return random.Random(0)