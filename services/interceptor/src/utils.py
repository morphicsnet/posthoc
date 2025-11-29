"""
Shared utilities for interceptor capture pipeline:
- topk_quantize
- pack_shard
- shard_key
- high-resolution timers
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union, Any
import heapq
import json
import time

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

VectorLike = Union[Sequence[float], List[float], "np.ndarray", bytes, memoryview, bytearray]


def _to_list(x: VectorLike) -> List[float]:
    if np is not None and isinstance(x, np.ndarray):
        return x.astype(float, copy=False).tolist()
    if isinstance(x, (bytes, bytearray, memoryview)):
        # Try to decode as JSON array of floats
        try:
            s = bytes(x).decode("utf-8")
            arr = json.loads(s)
            if isinstance(arr, list):
                return [float(v) for v in arr]
        except Exception:
            pass
        raise TypeError("Unsupported raw bytes format for activation vector; expected JSON-encoded list.")
    # Assume generic sequence
    return [float(v) for v in x]  # type: ignore[arg-type]


def topk_quantize(x: VectorLike, k: int) -> Tuple[List[int], List[float]]:
    """
    Select top-k by absolute value and return (indices, values).
    Stable order by index ascending to improve downstream compression effectiveness.
    """
    if k <= 0:
        return [], []
    if np is not None and isinstance(x, np.ndarray):
        arr = x.astype(float, copy=False)
        if getattr(arr, "ndim", 1) != 1:
            arr = arr.reshape(-1)
        k_eff = min(k, int(arr.size))
        if k_eff <= 0:
            return [], []
        idx = np.argpartition(np.abs(arr), -k_eff)[-k_eff:]
        idx = np.asarray(idx, dtype=int)
        vals = arr[idx]
        order = np.argsort(idx)
        idx_sorted = idx[order].astype(int).tolist()
        vals_sorted = vals[order].astype(float).tolist()
        return idx_sorted, vals_sorted
    # Fallback pure-Python
    lst = _to_list(x)
    k_eff = min(k, len(lst))
    if k_eff <= 0:
        return [], []
    # heapq.nlargest returns items in descending key order; we then sort indices
    top = heapq.nlargest(k_eff, enumerate(lst), key=lambda p: abs(p[1]))
    idx_sorted = sorted((i for i, _ in top))
    top_dict = {i: v for i, v in top}
    vals_sorted = [float(top_dict[i]) for i in idx_sorted]
    return idx_sorted, vals_sorted


def pack_shard(payload: Any, compress: str = "json") -> bytes:
    """
    Pack a shard payload (typically a dict or list) into bytes.
    compress:
      - 'json' (default): UTF-8 JSON without whitespace
      - 'none': same as json (no binary codec to keep stdlib-only)
    """
    if compress not in ("json", "none"):
        raise ValueError(f"Unsupported compress={compress}")
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def shard_key(trace_id: str, window_idx: int, namespace: str = "activations") -> str:
    """
    Deterministic shard key used by the capture pipeline to write window batches.
    """
    return f"{namespace}:{trace_id}:win:{int(window_idx):06d}"


def now_ns() -> int:
    return time.perf_counter_ns()


def ns_to_ms(ns: int) -> float:
    return ns / 1e6


__all__ = [
    "topk_quantize",
    "pack_shard",
    "shard_key",
    "now_ns",
    "ns_to_ms",
]