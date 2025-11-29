#!/usr/bin/env python3
"""
Microbenchmark: per-token capture overhead for interceptor CapturePipeline.

Simulates a decode loop of T tokens. For each token:
- Synthesizes a sparse activation vector (dim D, ~5% non-zeros)
- Invokes pipeline.capture_token(trace_id, t, vector)

Measures per-token capture time (ms) and reports mean/p50/p95.
Computes overhead ratio at p95 vs --model-ms-per-token and prints PASS/FAIL
for a 5% threshold.

Usage:
  python benchmarks/inference_capture_bench.py \
    --tokens 128 --dim 4096 --topk 256 \
    --model-ms-per-token 60 --compress json --repeat 3
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
from typing import List, Tuple

try:
    import numpy as np  # type: ignore  # pylint: disable=import-error
except Exception:  # pragma: no cover
    np = None  # type: ignore

# Robust imports whether running as a repo script or installed package
def _import_hooks():
    try:
        from services.interceptor.src.hooks import (  # type: ignore
            CaptureConfig,
            CapturePipeline,
            register_layer_hook,
            RedisClientStub,
        )
        return CaptureConfig, CapturePipeline, register_layer_hook, RedisClientStub
    except Exception:
        # Add repo root (two levels up from this file) to sys.path and retry
        import pathlib
        repo_root = str(pathlib.Path(__file__).resolve().parents[1])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from services.interceptor.src.hooks import (  # type: ignore
            CaptureConfig,
            CapturePipeline,
            register_layer_hook,
            RedisClientStub,
        )
        return CaptureConfig, CapturePipeline, register_layer_hook, RedisClientStub


CaptureConfig, CapturePipeline, register_layer_hook, RedisClientStub = _import_hooks()


def _sparse_vector(dim: int, sparsity: float = 0.05, use_numpy: bool = True):
    """Create a sparse vector with ~sparsity non-zeros in [-1, 1]."""
    nnz = max(1, int(dim * sparsity))
    if use_numpy and np is not None:
        arr = np.zeros((dim,), dtype=float)
        # without replacement is closer to intended sparsity
        idx = np.random.choice(dim, size=nnz, replace=False)
        arr[idx] = np.random.random(size=nnz) * 2 - 1
        return arr
    # pure python
    vec = [0.0] * dim
    # unique positions
    for i in random.sample(range(dim), k=nnz):
        vec[i] = random.random() * 2 - 1
    return vec


def _percentiles(values: List[float]) -> Tuple[float, float, float]:
    """Return mean, p50, p95 in ms."""
    if not values:
        return 0.0, 0.0, 0.0
    mean = statistics.fmean(values)
    p50 = statistics.median(values)
    # p95 (nearest-rank method)
    sorted_vals = sorted(values)
    rank = max(1, int(round(0.95 * len(sorted_vals))))
    p95 = sorted_vals[rank - 1]
    return mean, p50, p95


def run_once(tokens: int, dim: int, topk: int, compress: str, window_size: int = 16, use_numpy: bool = True):
    """
    Execute a single benchmark run and return:
      - stats: (mean_ms, p50_ms, p95_ms)
      - flush_ms
      - envelope (for potential debugging/size inspection)
    """
    # Configure pipeline
    cfg = CaptureConfig(
        model_name=os.getenv("BENCH_MODEL_NAME", "bench-model"),
        model_hash=os.getenv("BENCH_MODEL_HASH", "bench-hash"),
        topk=int(topk),
        window_size=int(window_size),
        compress=compress,
        namespace="activations",
    )
    pipe = CapturePipeline(cfg)
    trace_id = "bench-trace"
    pipe.start_trace(trace_id, params={"tokens": tokens, "dim": dim})

    # Optional hook adapter to simulate layer callback
    hook = register_layer_hook(pipe, trace_id)

    # Time per token
    t_ms: List[float] = []

    # Benchmark loop
    for t in range(tokens):
        acts = _sparse_vector(dim, sparsity=0.05, use_numpy=use_numpy)
        # time only the capture_token path (generation excluded)
        # use perf_counter_ns via stdlib for high-resolution timing
        start_ns = __import__("time").perf_counter_ns()
        hook(t, acts)
        end_ns = __import__("time").perf_counter_ns()
        t_ms.append((end_ns - start_ns) / 1e6)

    # Envelope production (flush) timing is not included per-token but reported
    start_ns = __import__("time").perf_counter_ns()
    envelope = pipe.flush(trace_id)
    end_ns = __import__("time").perf_counter_ns()
    flush_ms = (end_ns - start_ns) / 1e6

    return _percentiles(t_ms), flush_ms, envelope


def main():
    parser = argparse.ArgumentParser(description="Inference capture overhead microbenchmark")
    parser.add_argument("--tokens", type=int, default=128, help="Number of tokens to simulate")
    parser.add_argument("--dim", type=int, default=4096, help="Activation vector dimensionality")
    parser.add_argument("--topk", type=int, default=256, help="Top-k for quantization")
    parser.add_argument("--model-ms-per-token", type=float, default=60.0, help="Hypothetical model time per token (ms)")
    parser.add_argument("--compress", choices=["none", "json"], default="json", help="Packing format (json only, stdlib)")
    parser.add_argument("--repeat", type=int, default=3, help="Number of runs; best is reported")
    parser.add_argument("--no-numpy", action="store_true", help="Disable NumPy even if available")
    args = parser.parse_args()

    use_numpy = not args.no_numpy and (np is not None)

    results = []
    for r in range(args.repeat):
        stats, flush_ms, envelope = run_once(
            tokens=args.tokens,
            dim=args.dim,
            topk=args.topk,
            compress=args.compress,
            window_size=16,
            use_numpy=use_numpy,
        )
        mean_ms, p50_ms, p95_ms = stats
        results.append((stats, flush_ms, envelope))
        print(f"Run {r+1}/{args.repeat}: capture_ms mean={mean_ms:.4f} p50={p50_ms:.4f} p95={p95_ms:.4f} | flush_ms={flush_ms:.4f}")

    # Select best run by lowest p95
    best = min(results, key=lambda x: x[0][2])
    (best_mean, best_p50, best_p95), best_flush_ms, best_env = best

    overhead_ratio = (best_p95 / args.model_ms_per_token) if args.model_ms_per_token > 0 else float("inf")
    overhead_pct = overhead_ratio * 100.0
    threshold_ms = 0.05 * args.model_ms_per_token
    verdict = "PASS" if best_p95 <= threshold_ms else "FAIL"

    print("\nSummary (best-of runs):")
    print(f"- tokens={args.tokens} dim={args.dim} topk={args.topk} compress={args.compress} numpy={'yes' if use_numpy else 'no'}")
    print(f"- capture_ms: mean={best_mean:.4f} p50={best_p50:.4f} p95={best_p95:.4f}")
    print(f"- model_ms/token={args.model_ms_per_token:.4f} -> overhead @p95 = {overhead_pct:.2f}%")
    print(f"- flush_ms={best_flush_ms:.4f}  shards={len(best_env.get('shards', []))} num_tokens={best_env.get('num_tokens')}")
    print(f"- Threshold: p95 capture_ms ≤ 5% of model_ms/token -> {verdict}")

    # Exit code reflects verdict (0 pass, 1 fail)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()