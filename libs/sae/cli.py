from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

# Import-guarded: this module is part of the repo, but execution path may vary
try:
    from libs.sae.loader import SAEConfig, load_dictionary, project_topk  # type: ignore
except Exception as e:  # pragma: no cover
    print(f"[sae.cli] Failed to import loader: {e}", file=sys.stderr)
    raise


def _gen_probe_vec(length: int, seed: int) -> "List[float] | 'np.ndarray'":
    if length <= 0:
        return []  # type: ignore
    # Prefer numpy if available for speed and deterministic seeding
    if np is not None:
        rng = np.random.default_rng(seed)  # type: ignore
        # standard_normal to get both signs; encode as float32 for compute path
        return rng.standard_normal(length, dtype=np.float32)  # type: ignore
    # Fallback: Python stdlib
    import random as _rand  # local import to avoid polluting global state
    r = _rand.Random(seed)
    return [r.uniform(-1.0, 1.0) for _ in range(length)]


def main() -> int:
    p = argparse.ArgumentParser(description="SAE dictionary smoke CLI: load meta and one layer, do top-k projection.")
    p.add_argument("--root", required=True, help="Path to dictionary root or parent directory containing the dictionary")
    p.add_argument("--dict", dest="dictionary_name", default="sae-gpt4-2m", help="Dictionary name directory (default: sae-gpt4-2m)")
    p.add_argument("--layer", type=int, required=True, help="Layer index to load (e.g., 12)")
    p.add_argument("--topk", type=int, default=10, help="Top-k features to print (default: 10)")
    p.add_argument("--probe", type=int, default=None, help="Optional probe vector length; will be ignored if it mismatches the layer hidden_dim")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device preference (default: auto)")
    p.add_argument("--seed", type=int, default=123, help="Seed for random probe vector (default: 123)")
    args = p.parse_args()

    cfg = SAEConfig(
        root_path=Path(args.root),
        dictionary_name=args.dictionary_name,
        device=args.device,
        dtype="fp16",
        prefer_sparse=True,
        cache_layers=2,
    )

    try:
        t0 = time.perf_counter()
        dct = load_dictionary(cfg)
        t1 = time.perf_counter()
    except Exception as e:
        print(f"[sae.cli] Failed to load dictionary meta: {e}", file=sys.stderr)
        return 2

    print(f"[sae.cli] Loaded dictionary '{args.dictionary_name}' from '{Path(args.root)}' in {(t1 - t0)*1000:.2f} ms")
    meta = dct.meta or {}
    # Print a brief meta summary
    print(f"[sae.cli] meta.version={meta.get('version')} model_hash={meta.get('model_hash')} dtype={meta.get('dtype')} format={meta.get('format')}")

    try:
        t2 = time.perf_counter()
        layer = dct.get_layer(int(args.layer))
        t3 = time.perf_counter()
    except Exception as e:
        print(f"[sae.cli] Failed to load layer {args.layer}: {e}", file=sys.stderr)
        return 3

    rows, hidden_dim = layer.shape
    print(f"[sae.cli] Layer {args.layer}: features={rows} hidden_dim={hidden_dim} nnz={layer.nnz}")

    # Determine probe length
    if args.probe is not None and int(args.probe) != hidden_dim:
        print(f"[sae.cli] --probe {args.probe} != hidden_dim {hidden_dim}; using hidden_dim", file=sys.stderr)
    probe_len = hidden_dim

    try:
        probe = _gen_probe_vec(probe_len, seed=int(args.seed))
    except Exception as e:
        print(f"[sae.cli] Failed to generate probe vector: {e}", file=sys.stderr)
        return 4

    try:
        t4 = time.perf_counter()
        top: List[Tuple[int, float]] = project_topk(layer, probe, k=int(args.topk))
        t5 = time.perf_counter()
    except Exception as e:
        print(f"[sae.cli] project_topk failed: {e}", file=sys.stderr)
        return 5

    print(f"[sae.cli] top-{args.topk} computed in {(t5 - t4)*1000:.2f} ms")
    for i, (feat_idx, score) in enumerate(top, 1):
        print(f"{i:02d}. feat_{feat_idx}: {score:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())