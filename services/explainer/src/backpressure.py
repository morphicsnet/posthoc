from __future__ import annotations

"""
Backpressure controller for the Explainer.

Provides:
- BackpressureConfig, BackpressureState
- BackpressureController.evaluate() -> BackpressureState
- Advisor helpers to degrade Attribution, SAE decode, and Hypergraph configs

Metrics:
- backpressure_level{tenant,granularity} (gauge: 0=normal, 1=soft, 2=hard)
- backpressure_actions_total{action} (counter)
- backlog_seconds{granularity} (gauge)
- queue_len_global (gauge)
- queue_len_tenant{tenant} (gauge)

Notes:
- Pure stdlib except optional prometheus_client (already in requirements). Fallbacks are no-op if missing.
- QueueRef is a duck-typed provider from the worker. For tests, a FakeQueueRef can implement:
    * queue_len_tenant(tenant_id) -> int
    * queue_len_global() -> int
    * backlog_seconds(granularity) -> float | None
    * throughput_estimate(granularity) -> float | None     # items/sec
    * running_for_tenant(tenant_id) -> int | None          # in-flight
"""

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

# Optional Prometheus; guard if not present (unit tests still pass)
try:
    from prometheus_client import Counter, Gauge  # type: ignore
    _PROM_AVAILABLE = True
except Exception:  # pragma: no cover
    _PROM_AVAILABLE = False

    class _NoopLabels:
        def labels(self, *a, **k):  # type: ignore
            return self
        def inc(self, *a, **k):  # pragma: no cover
            return None
        def set(self, *a, **k):  # pragma: no cover
            return None

    class Counter:  # type: ignore
        def __init__(self, *a, **k): ...
        def labels(self, *a, **k):  # type: ignore
            return _NoopLabels()

    class Gauge:  # type: ignore
        def __init__(self, *a, **k): ...
        def labels(self, *a, **k):  # type: ignore
            return _NoopLabels()
        def set(self, *a, **k):  # type: ignore
            return None


# ----------------------------
# Config and State
# ----------------------------

@dataclass
class BackpressureConfig:
    max_backlog_seconds_sentence: float = 1.5
    max_backlog_seconds_token: float = 3.0
    max_queue_len_per_tenant: int = 200
    max_queue_len_global: int = 5000
    degrade_ladder: List[str] = field(default_factory=lambda: [
        "token->sentence",
        "reduce-samples",
        "reduce-topk",
        "reduce-layers",
        "saliency-fallback",
        "drop",
    ])
    tenant_minimum_guarantee: int = 2       # concurrent jobs per tenant
    burst_multiplier: float = 1.5           # allowance before hard shedding


@dataclass
class BackpressureState:
    level: str = "normal"  # "normal" | "soft" | "hard"
    actions: List[str] = field(default_factory=list)
    reason: str = ""
    tenant_overrides: Dict[str, List[str]] = field(default_factory=dict)

    def has(self, action: str) -> bool:
        return action in self.actions


# ----------------------------
# Metrics registry (low-cardinality)
# ----------------------------

_BP_LEVEL = Gauge(
    "backpressure_level",
    "Backpressure level (0=normal,1=soft,2=hard) per tenant/granularity",
    ["tenant", "granularity"],
)
_BP_ACTIONS = Counter(
    "backpressure_actions_total",
    "Total backpressure actions taken",
    ["action"],
)
_BP_BACKLOG = Gauge(
    "backlog_seconds",
    "Estimated backlog seconds per granularity",
    ["granularity"],
)
_BP_Q_GLOBAL = Gauge(
    "queue_len_global",
    "Global queue length",
)
_BP_Q_TENANT = Gauge(
    "queue_len_tenant",
    "Per-tenant queue length",
    ["tenant"],
)


def _emit_metrics(tenant: str, granularity: str, level: str, backlog_s: float, q_tenant: int, q_global: int, actions: List[str]) -> None:  # pragma: no cover (smoke)
    try:
        lvl_val = 0 if level == "normal" else (1 if level == "soft" else 2)
        _BP_LEVEL.labels(str(tenant), str(granularity)).set(float(lvl_val))
        _BP_BACKLOG.labels(str(granularity)).set(float(max(0.0, backlog_s)))
        _BP_Q_GLOBAL.set(float(max(0, q_global)))
        _BP_Q_TENANT.labels(str(tenant)).set(float(max(0, q_tenant)))
        for a in actions:
            _BP_ACTIONS.labels(str(a)).inc()
    except Exception:
        # Metrics are best-effort
        pass


# ----------------------------
# Controller
# ----------------------------

class BackpressureController:
    """
    Centralized evaluator. Accepts a queue_ref (duck-typed object providing queue stats).
    metrics_hook: optional callable(tenant, granularity, BackpressureState, extras_dict) -> None
    """

    def __init__(self, queue_ref: Any, metrics_hook: Optional[Callable[[str, str, BackpressureState, Dict[str, Any]], None]] = None, cfg: Optional[BackpressureConfig] = None) -> None:
        self.q = queue_ref
        self.cfg = cfg or BackpressureConfig()
        self.metrics_hook = metrics_hook or (lambda tenant, granularity, bp, extras: _emit_metrics(
            tenant,
            granularity,
            bp.level,
            extras.get("backlog_seconds", 0.0),
            extras.get("queue_len_tenant", 0),
            extras.get("queue_len_global", 0),
            bp.actions,
        ))
        # simple EMA throughput (items/sec) per granularity if queue_ref can't provide
        self._ema_rate: Dict[str, float] = {"sentence": 20.0, "token": 10.0}  # conservative defaults

    # Optional: allow worker to update EMA externally
    def update_throughput(self, granularity: str, per_second: float, alpha: float = 0.2) -> None:
        if not granularity:
            return
        try:
            ps = max(0.1, float(per_second))
            prev = float(self._ema_rate.get(granularity, ps))
            self._ema_rate[granularity] = (1.0 - alpha) * prev + alpha * ps
        except Exception:
            pass

    def _estimate_backlog_seconds(self, granularity: str, q_global: int) -> float:
        # Prefer queue_ref.backlog_seconds(granularity) if present
        try:
            if hasattr(self.q, "backlog_seconds"):
                v = self.q.backlog_seconds(granularity)  # type: ignore
                if v is not None:
                    return float(max(0.0, v))
        except Exception:
            pass
        # Else compute from global length and EMA throughput
        rate = float(self._ema_rate.get(granularity, 1.0))
        try:
            if hasattr(self.q, "throughput_estimate"):
                est = self.q.throughput_estimate(granularity)  # type: ignore
                if est is not None:
                    rate = float(max(0.1, est))
        except Exception:
            pass
        return float(q_global) / float(max(0.1, rate))

    def _thresholds(self, granularity: str) -> Tuple[float, float]:
        if (granularity or "sentence").lower() == "token":
            soft = float(self.cfg.max_backlog_seconds_token)
        else:
            soft = float(self.cfg.max_backlog_seconds_sentence)
        hard = float(max(soft, soft * float(self.cfg.burst_multiplier)))
        return soft, hard

    def _determine_level_and_actions(
        self,
        tenant_id: str,
        granularity: str,
        backlog_s: float,
        q_tenant: int,
        q_global: int,
        running_for_tenant: int,
    ) -> Tuple[str, List[str], str]:
        soft_th, hard_th = self._thresholds(granularity)
        ladder = list(self.cfg.degrade_ladder or [])

        level = "normal"
        reason = ""
        actions: List[str] = []

        # Severity from backlog and queue caps
        soft_trigger = (backlog_s > soft_th) or (q_tenant > int(self.cfg.max_queue_len_per_tenant))
        hard_trigger = (backlog_s > hard_th) or (q_global > int(self.cfg.max_queue_len_global))

        if hard_trigger:
            level = "hard"
            reason = f"hard: backlog_s={backlog_s:.2f} soft={soft_th:.2f} hard={hard_th:.2f} q_tenant={q_tenant} q_global={q_global}"
        elif soft_trigger:
            level = "soft"
            reason = f"soft: backlog_s={backlog_s:.2f} soft={soft_th:.2f} q_tenant={q_tenant} q_global={q_global}"
        else:
            return level, actions, "ok"

        # Build actions in a deterministic order based on ladder and level
        desired: List[str] = []
        for a in ladder:
            if a == "token->sentence":
                if (granularity or "sentence").lower() == "token":
                    # Prefer early downgrade under any overload
                    desired.append(a)
            elif a == "reduce-samples":
                desired.append(a)
            elif a == "reduce-topk":
                desired.append(a)
            elif a == "reduce-layers":
                if level == "hard" or soft_trigger:
                    desired.append(a)
            elif a == "saliency-fallback":
                if level == "hard":
                    desired.append(a)
            elif a == "drop":
                # Consider drop only under very hard overload AND not violating min guarantee
                # Very hard heuristic: backlog 1.5x hard or global queue > 1.2x cap
                very_hard = (backlog_s > (hard_th * 1.5)) or (q_global > int(self.cfg.max_queue_len_global * 1.2))
                if very_hard:
                    desired.append(a)

        # Enforce tenant minimum guarantee (unless global cap at/over max)
        if "drop" in desired:
            allow_drop = (q_global >= int(self.cfg.max_queue_len_global))
            if running_for_tenant < int(self.cfg.tenant_minimum_guarantee) and not allow_drop:
                desired = [x for x in desired if x != "drop"]
                reason += f" | min_guarantee_keep (running={running_for_tenant} < {self.cfg.tenant_minimum_guarantee})"

        actions = desired
        return level, actions, reason

    def evaluate(self, tenant_id: str, granularity: str) -> BackpressureState:
        t = str(tenant_id or "anon")
        g = str(granularity or "sentence").lower()

        # Queue lens and running counts
        try:
            q_tenant = int(getattr(self.q, "queue_len_tenant")(t))  # type: ignore
        except Exception:
            q_tenant = 0
        try:
            q_global = int(getattr(self.q, "queue_len_global")())  # type: ignore
        except Exception:
            q_global = 0
        try:
            running = int(getattr(self.q, "running_for_tenant")(t))  # type: ignore
        except Exception:
            running = 0

        backlog_s = self._estimate_backlog_seconds(g, q_global)
        level, actions, reason = self._determine_level_and_actions(t, g, backlog_s, q_tenant, q_global, running)

        bp = BackpressureState(level=level, actions=actions, reason=reason, tenant_overrides={})
        # Emit metrics via hook
        try:
            self.metrics_hook(t, g, bp, {
                "backlog_seconds": backlog_s,
                "queue_len_tenant": q_tenant,
                "queue_len_global": q_global,
                "running_for_tenant": running,
            })
        except Exception:
            pass
        return bp

    # ----------------------------
    # Advisors (config degraders)
    # ----------------------------

    def advise_attribution(self, config: Any, bp: BackpressureState) -> Any:
        """
        Degrade AttributionConfig based on actions.
        Falls back to in-place best-effort adjustments if attribution.apply_degradation not available.
        """
        try:
            # Prefer helper in attribution module if present
            from services.explainer.src.attribution import apply_degradation as _apply  # type: ignore
        except Exception:
            try:
                from attribution import apply_degradation as _apply  # type: ignore
            except Exception:
                _apply = None  # type: ignore

        if _apply is not None:
            try:
                return _apply(config, list(bp.actions))
            except Exception:
                pass

        # Fallback local logic (pure dataclass-like object with attributes)
        cfg = config
        acts = set(bp.actions)
        try:
            if "reduce-samples" in acts and hasattr(cfg, "max_samples"):
                try:
                    ms = int(getattr(cfg, "max_samples"))
                except Exception:
                    ms = 64
                setattr(cfg, "max_samples", max(16, ms // 2))
            if "reduce-samples" in acts and hasattr(cfg, "early_stop_delta"):
                try:
                    es = float(getattr(cfg, "early_stop_delta"))
                except Exception:
                    es = 0.01
                setattr(cfg, "early_stop_delta", min(0.1, es * 2.0))
            if "saliency-fallback" in acts:
                # Add a generic flag that downstream may interpret
                setattr(cfg, "force_saliency_fallback", True)  # type: ignore[attr-defined]
        except Exception:
            pass
        return cfg

    def advise_sae(self, config: Any, bp: BackpressureState) -> Any:
        """
        Degrade SAEDecodeConfig: lower topk and cache_layers under pressure.
        """
        cfg = config
        acts = set(bp.actions)
        try:
            if "reduce-topk" in acts and hasattr(cfg, "topk"):
                try:
                    cur = int(getattr(cfg, "topk"))
                except Exception:
                    cur = 256
                # hard -> 64, soft -> 128
                target = 64 if bp.level == "hard" else 128
                setattr(cfg, "topk", max(1, min(cur, target)))
            if "reduce-layers" in acts and hasattr(cfg, "cache_layers"):
                try:
                    cl = int(getattr(cfg, "cache_layers"))
                except Exception:
                    cl = 3
                target = 1 if bp.level == "hard" else 2
                setattr(cfg, "cache_layers", max(1, min(cl, target)))
        except Exception:
            pass
        return cfg

    def advise_hypergraph(self, cfg: Any, bp: BackpressureState) -> Any:
        """
        Degrade HypergraphConfig via exported helper if available, else adjust common knobs.
        """
        try:
            from services.explainer.src.hypergraph import degrade_hg_config as _dg  # type: ignore
        except Exception:
            try:
                from hypergraph import degrade_hg_config as _dg  # type: ignore
            except Exception:
                _dg = None  # type: ignore

        if _dg is not None:
            try:
                return _dg(cfg, list(bp.actions), level=bp.level)
            except Exception:
                pass

        # Fallback: reduce per_token_incident_cap and increase min_edge_weight
        try:
            acts = set(bp.actions)
            if "reduce-topk" in acts and hasattr(cfg, "per_token_incident_cap"):
                cur = int(getattr(cfg, "per_token_incident_cap"))
                target = 64 if bp.level == "hard" else 128
                setattr(cfg, "per_token_incident_cap", max(8, min(cur, target)))
            if ("reduce-layers" in acts or "saliency-fallback" in acts) and hasattr(cfg, "min_edge_weight"):
                cur = float(getattr(cfg, "min_edge_weight"))
                target = 0.05 if bp.level == "hard" else 0.02
                setattr(cfg, "min_edge_weight", max(cur, target))
        except Exception:
            pass
        return cfg


__all__ = [
    "BackpressureConfig",
    "BackpressureState",
    "BackpressureController",
]