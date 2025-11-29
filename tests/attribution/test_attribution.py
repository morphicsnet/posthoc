from __future__ import annotations

import os
import sys
import time
from typing import Dict, List

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    # Prefer canonical import path
    from services.explainer.src.attribution import AttributionConfig, sentence_attribution, token_attribution  # type: ignore
except Exception:
    # Fallback if running from a different CWD
    import importlib  # type: ignore
    import importlib.util  # type: ignore
    spec = importlib.util.spec_from_file_location(
        "attribution",
        os.path.abspath(os.path.join(ROOT, "services", "explainer", "src", "attribution.py")),
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    AttributionConfig = mod.AttributionConfig  # type: ignore
    sentence_attribution = mod.sentence_attribution  # type: ignore
    token_attribution = mod.token_attribution  # type: ignore


def _sample_features(n: int = 50) -> Dict[str, float]:
    # Deterministic simple ramp
    return {f"feat_{i}": float(i + 1) for i in range(max(1, n))}


def test_sentence_attribution_deterministic_and_caps() -> None:
    feats = _sample_features(40)
    tokens = "This is a tiny output sentence.".split()
    cfg = AttributionConfig(
        method="acdc",
        max_samples=256,
        early_stop_delta=0.01,
        max_ms_budget=200,  # tight but sufficient
        random_seed=1234,
        min_edge_weight=0.05,
        per_token_incident_cap=16,
    )
    t0 = time.perf_counter()
    edges1 = sentence_attribution(feats, tokens, cfg)
    t1 = time.perf_counter()
    edges2 = sentence_attribution(feats, tokens, cfg)  # same seed => deterministic
    t2 = time.perf_counter()

    # Determinism: all fields in-order should match
    assert edges1 == edges2, "Sentence attribution should be deterministic under fixed seed"

    # Caps and weights
    assert len(edges1) <= cfg.per_token_incident_cap, "Incident cap must be respected"
    for e in edges1:
        assert isinstance(e.get("id"), str) and e["id"].startswith("att_")
        w = float(e.get("weight", 0.0))
        assert 0.0 <= w <= 1.0, "Edge weights must be within [0,1]"
        assert w >= cfg.min_edge_weight, "Edges should be pruned below min_edge_weight"
        node_ids = e.get("node_ids") or []
        assert isinstance(node_ids, list) and len(node_ids) == 2, "node_ids must connect feature and output token"
        assert node_ids[0].startswith("feat_")
        assert node_ids[1].startswith("token_out_")
        md = e.get("metadata") or {}
        assert md.get("method") in ("acdc", "shapley")
        assert "window" in md

    # Runtime should be reasonable
    assert (t1 - t0) < 1.0 and (t2 - t1) < 1.0, "Method should complete fast for small N"


def test_token_attribution_deterministic_and_caps() -> None:
    feats = _sample_features(60)
    ctx = "Short answer with several tokens to simulate context.".split()
    cfg = AttributionConfig(
        method="shapley",
        max_samples=300,
        early_stop_delta=0.01,
        max_ms_budget=300,
        random_seed=4321,
        min_edge_weight=0.02,
        per_token_incident_cap=10,
    )
    token_idx = max(0, len(ctx) - 1)  # last token

    t0 = time.perf_counter()
    edges1 = token_attribution(feats, token_idx, ctx, cfg)
    t1 = time.perf_counter()
    edges2 = token_attribution(feats, token_idx, ctx, cfg)
    t2 = time.perf_counter()

    # Determinism
    assert edges1 == edges2, "Token attribution should be deterministic under fixed seed"

    # Caps and weights
    assert len(edges1) <= cfg.per_token_incident_cap, "Incident cap must be respected"
    for e in edges1:
        assert isinstance(e.get("id"), str) and e["id"].startswith("att_")
        w = float(e.get("weight", 0.0))
        assert 0.0 <= w <= 1.0, "Edge weights must be within [0,1]"
        assert w >= cfg.min_edge_weight, "Edges should be pruned below min_edge_weight"
        node_ids = e.get("node_ids") or []
        assert isinstance(node_ids, list) and len(node_ids) == 2, "node_ids must connect feature and output token"
        assert node_ids[0].startswith("feat_")
        assert node_ids[1].startswith("token_out_")
        md = e.get("metadata") or {}
        assert md.get("method") in ("acdc", "shapley")
        assert "window" in md

    # Runtime smoke bounds
    assert (t1 - t0) < 1.0 and (t2 - t1) < 1.0, "Method should complete fast for small N"


def test_early_stop_and_budget_smoke() -> None:
    """
    Smoke test: only runs fully when ATTR_TEST_SMOKE=1.
    We assert the elapsed time remains under budget + slack.
    """
    if os.getenv("ATTR_TEST_SMOKE", "0") != "1":
        return

    feats = _sample_features(100)
    tokens = "Budget test sentence tokens for smoke.".split()

    # Sentence budget 50ms -> elapsed <= 150ms slack to avoid flakiness
    cfg_s = AttributionConfig(
        method="acdc",
        max_samples=10000,           # large to rely on early-stop and budget
        early_stop_delta=0.05,       # moderate early stop
        max_ms_budget=50,
        random_seed=7,
        min_edge_weight=0.01,
        per_token_incident_cap=32,
    )
    t0 = time.perf_counter()
    _ = sentence_attribution(feats, tokens, cfg_s)
    t1 = time.perf_counter()
    assert (t1 - t0) * 1000.0 <= (cfg_s.max_ms_budget + 100), "Sentence attribution exceeded budget bound (with slack)"

    # Token budget 80ms -> elapsed <= 180ms slack
    ctx = tokens
    cfg_t = AttributionConfig(
        method="shapley",
        max_samples=10000,
        early_stop_delta=0.05,
        max_ms_budget=80,
        random_seed=13,
        min_edge_weight=0.01,
        per_token_incident_cap=32,
    )
    t2 = time.perf_counter()
    _ = token_attribution(feats, len(ctx) - 1, ctx, cfg_t)
    t3 = time.perf_counter()
    assert (t3 - t2) * 1000.0 <= (cfg_t.max_ms_budget + 100), "Token attribution exceeded budget bound (with slack)"


if __name__ == "__main__":
    # Run all tests when executed directly
    test_sentence_attribution_deterministic_and_caps()
    test_token_attribution_deterministic_and_caps()
    test_early_stop_and_budget_smoke()
    print("OK: attribution tests passed")