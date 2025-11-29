from __future__ import annotations

import math
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque

# Import guards for flexible package layout
try:
    from services.explainer.src.budget import BudgetTimer, select_candidates, rng_from_seed
except Exception:
    try:
        from .budget import BudgetTimer, select_candidates, rng_from_seed  # type: ignore
    except Exception:  # pragma: no cover
        from budget import BudgetTimer, select_candidates, rng_from_seed  # type: ignore
    except Exception:  # pragma: no cover
        from budget import BudgetTimer, select_candidates, rng_from_seed  # type: ignore

# Observability (optional)
try:
    from services.explainer.src import otel as _otel  # type: ignore
except Exception:
    try:
        import otel as _otel  # type: ignore
    except Exception:
        _otel = None  # type: ignore


@dataclass
class AttributionConfig:
    """
    Budgeted attribution configuration.
    - method: "acdc"|"shapley"
    - max_samples: sampling cap (coalitions/subsets)
    - early_stop_delta: stop when rolling avg marginal gain below threshold
    - max_ms_budget: strict wall clock budget
    - random_seed: deterministic seeding; None => non-deterministic
    - min_edge_weight: edges below are pruned
    - per_token_incident_cap: cap incident edges to avoid hairballs
    """
    method: str = "acdc"          # sentence default
    max_samples: int = 512
    early_stop_delta: float = 0.01
    max_ms_budget: int = 900      # sentence default; for token use 3500
    random_seed: Optional[int] = None
    min_edge_weight: float = 0.01
    per_token_incident_cap: int = 256


# ----------------------------
# Internal helpers (pure-stdlib, deterministic with seed)
# ----------------------------

def _norm_activations(features: Dict[str, float]) -> Dict[str, float]:
    # Normalize by max absolute value to bring into [0,1]
    if not features:
        return {}
    m = 0.0
    for v in features.values():
        try:
            m = max(m, abs(float(v)))
        except Exception:
            continue
    m = m if m > 0 else 1.0
    return {k: max(0.0, float(v) / m) for k, v in features.items()}


def _hash_u64(s: str) -> int:
    # Deterministic uint64 from md5
    d = hashlib.md5(s.encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big", signed=False)


def _prand01(key: str, base_seed: int) -> float:
    # Stable pseudo-random in [0,1) derived from key and seed
    u = _hash_u64(f"{base_seed}:{key}")
    # Map to [0,1)
    return (u & ((1 << 53) - 1)) / float(1 << 53)


def _influence01(fid: str, token: str, base_seed: int) -> float:
    # Co-activation proxy for feature-token pair; lightly shape distribution
    x = _prand01(f"{fid}|{token}", base_seed)
    # Emphasize middle band to avoid extreme saturation
    return 0.15 + 0.85 * x  # in (0.15,1.0)


def _pair_synergy01(fa: str, fb: str, token: str, base_seed: int) -> float:
    # Pairwise synergy proxy (symmetric)
    key = "|".join(sorted([fa, fb])) + f"|{token}"
    x = _prand01(key, base_seed ^ 0xA53F_19E7)
    # Slightly centered around 0.5 with modest spread
    return 0.4 + 0.3 * (2.0 * x - 1.0)  # approx ~ (0.1,0.7)


def _subset_score(feats: List[str], act_n: Dict[str, float], tokens: List[str], seed: int) -> float:
    # Proxy value of an intervention subset: combine activation, per-token influence, and pair synergies
    if not feats:
        return 0.0
    # Token-level influence averaged over provided context tokens
    if not tokens:
        tokens = ["_"]  # neutral placeholder
    infs = []
    for f in feats:
        a = act_n.get(f, 0.0)
        # average influence over tokens
        avg_inf = 0.0
        for t in tokens:
            avg_inf += _influence01(f, t, seed)
        avg_inf /= float(len(tokens))
        infs.append(a * avg_inf)
    base = sum(infs) / float(len(infs))  # average contribution per member

    # Pair synergy multiplier (geometric-style mean on [0.1,0.7] -> scaled)
    if len(feats) >= 2:
        pairs: List[float] = []
        tok = tokens[0]  # tie to first token to reduce cost
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                pairs.append(_pair_synergy01(feats[i], feats[j], tok, seed))
        if pairs:
            # Normalize pairs to ~[0.0, 1.0] and multiply
            avg_pair = sum(pairs) / float(len(pairs))
            mult = max(0.0, min(1.0, (avg_pair - 0.1) / 0.6))
            base = base * (0.75 + 0.5 * mult)  # in ~[0.75*base, 1.25*base]
    return float(max(0.0, base))


def _finalize_edges(
    contrib: Dict[str, float],
    target_node_id: str,
    method_name: str,
    window: str,
    min_edge_weight: float,
    cap: int,
) -> List[Dict]:
    if not contrib:
        return []
    # Normalize to max=1.0
    m = max(contrib.values()) if contrib else 0.0
    if m <= 0.0:
        return []
    items = sorted(((k, v / m) for k, v in contrib.items()), key=lambda kv: kv[1], reverse=True)
    # Prune by threshold and cap
    out: List[Dict] = []
    for rank, (fid, w) in enumerate(items, start=1):
        w_clamped = float(max(0.0, min(1.0, w)))
        if w_clamped < float(min_edge_weight):
            continue
        out.append(
            {
                "id": f"att_{method_name}_{rank}",
                "node_ids": [fid, target_node_id],
                "weight": w_clamped,
                "metadata": {"type": "causal_circuit", "method": method_name, "window": window},
            }
        )
        if len(out) >= int(max(1, cap)):
            break
    return out


def apply_degradation(cfg: AttributionConfig, actions: List[str]) -> AttributionConfig:
    """
    Apply graceful degradation actions to AttributionConfig.
    - reduce-samples: halve max_samples (floor 16) and relax early_stop_delta x2 up to 0.1
    - saliency-fallback: set a boolean flag 'force_saliency_fallback' on the config
    The function mutates and returns cfg for convenience.
    """
    try:
        acts = set(actions or [])
    except Exception:
        acts = set()
    try:
        if "reduce-samples" in acts:
            try:
                cfg.max_samples = max(16, int(getattr(cfg, "max_samples", 64)) // 2)
            except Exception:
                pass
            try:
                cfg.early_stop_delta = min(0.1, float(getattr(cfg, "early_stop_delta", 0.01)) * 2.0)
            except Exception:
                pass
        if "saliency-fallback" in acts:
            # Signal to downstream to skip sampling loops in favor of heuristic
            setattr(cfg, "force_saliency_fallback", True)  # attribute is optional at call sites
    except Exception:
        # best-effort only
        pass
    return cfg

# ----------------------------
# Public strategies
# ----------------------------

def sentence_attribution(
    features: Dict[str, float],
    tokens: List[str],
    config: AttributionConfig,
) -> List[Dict]:
    """
    ACDC-style approximation at sentence granularity.
    - Candidate selection: top-N by activation.
    - Sample small subsets (size 1-3) and accumulate distributed proxy contribution.
    - Early stop on marginal gain threshold or time budget.
    - Fallback to activation-weighted if budget hit immediately.
    Return: legacy HIF v1-style incidences with node_ids and metadata.
    """
    t0 = time.perf_counter()
    early_stop = False
    cfg = config or AttributionConfig()
    timer = BudgetTimer(int(max(1, cfg.max_ms_budget or 900)))
    rng = rng_from_seed(cfg.random_seed)
    act_n = _norm_activations(features)
    candidates = [fid for fid, _ in select_candidates(features, max_features=512)]
    if not candidates:
        return []

    target_node_id = "token_out_1"
    window = "sent-1"
    method_name = "acdc"

    # If nearly no budget, fallback immediately
    if timer.time_left_ms() <= 1:
        contrib = {fid: act_n.get(fid, 0.0) for fid in candidates}
        return _finalize_edges(contrib, target_node_id, method_name, window, cfg.min_edge_weight, cfg.per_token_incident_cap)

    contrib: Dict[str, float] = {fid: 0.0 for fid in candidates}
    samples = max(1, int(cfg.max_samples))
    # Rolling window for early stopping
    deltas: deque[float] = deque(maxlen=32)
    last_total = 0.0

    i = 0
    while i < samples and not timer.expired():
        # Random subset size biased towards 1-2
        r = rng.random()
        if r < 0.6:
            k = 1
        elif r < 0.9:
            k = 2
        else:
            k = 3
        k = min(k, len(candidates))
        if k <= 0:
            break
        subset = rng.sample(candidates, k)
        score = _subset_score(subset, act_n, tokens, (cfg.random_seed or 0) + i)
        # Distribute equally (ACDC-like attribution by masking unit)
        per = score / float(k) if k > 0 else 0.0
        for f in subset:
            contrib[f] += per

        # Early stop on low marginal gain
        total = sum(contrib.values())
        delta = abs(total - last_total)
        last_total = total
        deltas.append(delta)
        if len(deltas) >= deltas.maxlen and (sum(deltas) / float(deltas.maxlen)) <= float(cfg.early_stop_delta):
            early_stop = True
            break

        i += 1

    # Fallback if no work was done (e.g., budget exhausted)
    if sum(contrib.values()) <= 0.0:
        contrib = {fid: act_n.get(fid, 0.0) for fid in candidates}

    dt = time.perf_counter() - t0
    try:
        if _otel is not None:
            _otel.attribution_observe(method_name, "sentence", float(dt), early_stop)
    except Exception:
        pass
    return _finalize_edges(contrib, target_node_id, method_name, window, cfg.min_edge_weight, cfg.per_token_incident_cap)


def token_attribution(
    features: Dict[str, float],
    token_idx: int,
    context_tokens: List[str],
    config: AttributionConfig,
) -> List[Dict]:
    """
    Sampled Shapley approximation for a specific output token index.
    - Value function: sum of normalized activations * token-specific influence + mild pair synergy.
    - Random coalitions with early stopping by rolling avg marginal |delta|.
    - Strict wall-clock budget; fallback to activation-weighted if exhausted too early.
    Return: legacy HIF v1-style incidences.
    """
    t0 = time.perf_counter()
    early_stop = False
    cfg = config or AttributionConfig(method="shapley", max_ms_budget=3500)
    timer = BudgetTimer(int(max(1, cfg.max_ms_budget or 3500)))
    rng = rng_from_seed(cfg.random_seed)
    act_n = _norm_activations(features)
    candidates = [fid for fid, _ in select_candidates(features, max_features=512)]
    if not candidates:
        return []

    # Guard token index
    try:
        ti = int(token_idx)
    except Exception:
        ti = 0
    ti = max(0, ti)
    target_tok = context_tokens[ti] if (isinstance(context_tokens, list) and ti < len(context_tokens) and ti >= 0) else f"tok{ti}"
    target_node_id = f"token_out_{ti+1}"
    window = f"tok-{ti}"
    method_name = "shapley"

    # If minimal budget, fallback
    if timer.time_left_ms() <= 1:
        # Greedy activation-weighted heuristic proxied by target token influence
        contrib = {fid: act_n.get(fid, 0.0) * _influence01(fid, target_tok, cfg.random_seed or 0) for fid in candidates}
        return _finalize_edges(contrib, target_node_id, method_name, window, cfg.min_edge_weight, cfg.per_token_incident_cap)

    # Precompute singleton values to accelerate marginal estimates
    single_v: Dict[str, float] = {}
    for fid in candidates:
        single_v[fid] = act_n.get(fid, 0.0) * _influence01(fid, target_tok, (cfg.random_seed or 0) ^ 0x55AA_1234)

    def v_of(S: List[str]) -> float:
        if not S:
            return 0.0
        base = sum(single_v.get(f, 0.0) for f in S)
        # Mild pair synergy on the chosen token
        if len(S) >= 2:
            pairs = []
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    pairs.append(_pair_synergy01(S[i], S[j], target_tok, (cfg.random_seed or 0) ^ 0x91E3_77B5))
            if pairs:
                avgp = sum(pairs) / float(len(pairs))
                mult = max(0.0, min(1.0, (avgp - 0.1) / 0.6))
                base = base * (0.80 + 0.4 * mult)  # [0.8,1.2]x
        return float(max(0.0, base))

    # Shapley sampling: random coalitions around a randomly chosen feature
    max_samples = max(1, int(cfg.max_samples))
    contrib_sum: Dict[str, float] = {fid: 0.0 for fid in candidates}
    contrib_cnt: Dict[str, int] = {fid: 0 for fid in candidates}
    deltas: deque[float] = deque(maxlen=64)

    i = 0
    while i < max_samples and not timer.expired():
        # Pick a random feature i to evaluate marginal for this sample
        fi = rng.choice(candidates)

        # Build a coalition S not containing fi; restrict coalition size to keep computation bounded
        others = [f for f in candidates if f != fi]
        # Sample up to 12 others, then include each with p=0.5
        if others:
            sampled = rng.sample(others, k=min(len(others), 12))
        else:
            sampled = []
        S: List[str] = []
        for f in sampled:
            if rng.random() < 0.5:
                S.append(f)

        # Value difference
        vS = v_of(S)
        vS_i = v_of(S + [fi])
        marg = vS_i - vS
        contrib_sum[fi] += marg
        contrib_cnt[fi] += 1

        deltas.append(abs(marg))
        if len(deltas) >= deltas.maxlen and (sum(deltas) / float(deltas.maxlen)) <= float(cfg.early_stop_delta):
            early_stop = True
            break

        i += 1

    # Aggregate averages; if nothing computed, fallback to activation-weighted influence
    out_contrib: Dict[str, float] = {}
    total_cnt = sum(contrib_cnt.values())
    if total_cnt > 0:
        for fid, csum in contrib_sum.items():
            cnt = max(1, contrib_cnt.get(fid, 0))
            out_contrib[fid] = max(0.0, csum / float(cnt))
    else:
        out_contrib = {fid: single_v.get(fid, 0.0) for fid in candidates}

    # If budget hit too early and contributions are all ~0, fallback to activation-weighted
    if max(out_contrib.values() or [0.0]) <= 0.0:
        out_contrib = {fid: single_v.get(fid, 0.0) for fid in candidates}

    dt = time.perf_counter() - t0
    try:
        if _otel is not None:
            _otel.attribution_observe(method_name, "token", float(dt), early_stop)
    except Exception:
        pass
    return _finalize_edges(out_contrib, target_node_id, method_name, window, cfg.min_edge_weight, cfg.per_token_incident_cap)