from __future__ import annotations

import os
import sys
import importlib.util
from typing import Any, Dict, List, Optional, Tuple


def _load_module(mod_name: str, path: str):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    sys.modules[mod_name] = mod
    return mod


# Resolve project root and module paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BP_PATH = os.path.join(ROOT, "services", "explainer", "src", "backpressure.py")
ATTR_PATH = os.path.join(ROOT, "services", "explainer", "src", "attribution.py")
HG_PATH = os.path.join(ROOT, "services", "explainer", "src", "hypergraph.py")

bp_mod = _load_module("explainer_backpressure", BP_PATH)
attr_mod = _load_module("explainer_attribution", ATTR_PATH)
hg_mod = _load_module("explainer_hypergraph", HG_PATH)


# Aliases
BackpressureConfig = bp_mod.BackpressureConfig
BackpressureState = bp_mod.BackpressureState
BackpressureController = bp_mod.BackpressureController
AttributionConfig = attr_mod.AttributionConfig
apply_degradation = getattr(attr_mod, "apply_degradation", None)
HypergraphConfig = hg_mod.HypergraphConfig
degrade_hg_config = getattr(hg_mod, "degrade_hg_config", None)


class FakeQueueRef:
    def __init__(self) -> None:
        self._q_tenant: Dict[str, int] = {}
        self._q_global: int = 0
        self._backlog_seconds: Dict[str, float] = {}
        self._running: Dict[str, int] = {}
        self._throughput: Dict[str, float] = {"sentence": 20.0, "token": 10.0}

    def queue_len_tenant(self, tenant_id: str) -> int:
        return int(self._q_tenant.get(str(tenant_id), 0))

    def queue_len_global(self) -> int:
        return int(self._q_global)

    def backlog_seconds(self, granularity: str) -> Optional[float]:
        # Return per-granularity backlog seconds if set
        g = (granularity or "sentence").lower()
        return float(self._backlog_seconds.get(g)) if g in self._backlog_seconds else None

    def throughput_estimate(self, granularity: str) -> Optional[float]:
        g = (granularity or "sentence").lower()
        return float(self._throughput.get(g, 10.0))

    def running_for_tenant(self, tenant_id: str) -> int:
        return int(self._running.get(str(tenant_id), 0))

    # Mutators for tests
    def set_backlog(self, g: str, seconds: float) -> None:
        self._backlog_seconds[g] = float(seconds)

    def set_qtenant(self, tenant: str, n: int) -> None:
        self._q_tenant[str(tenant)] = int(n)

    def set_qglobal(self, n: int) -> None:
        self._q_global = int(n)

    def set_running(self, tenant: str, n: int) -> None:
        self._running[str(tenant)] = int(n)


def test_evaluate_normal_no_actions() -> None:
    fake = FakeQueueRef()
    cfg = BackpressureConfig()
    ctrl = BackpressureController(fake, metrics_hook=None, cfg=cfg)

    fake.set_qglobal(0)
    fake.set_qtenant("t1", 0)
    fake.set_backlog("sentence", 0.1)

    bp = ctrl.evaluate("t1", "sentence")
    assert isinstance(bp, BackpressureState)
    assert bp.level == "normal"
    assert bp.actions == []


def test_evaluate_soft_overload_actions_include_reduce() -> None:
    fake = FakeQueueRef()
    cfg = BackpressureConfig()
    ctrl = BackpressureController(fake, metrics_hook=None, cfg=cfg)

    # Soft overload for token granularity (exceed soft threshold 3.0)
    fake.set_backlog("token", cfg.max_backlog_seconds_token * 1.1)
    fake.set_qtenant("tA", 10)
    fake.set_qglobal(100)

    bp = ctrl.evaluate("tA", "token")
    acts = set(bp.actions)
    # token->sentence is expected for token requests
    assert "token->sentence" in acts
    # Soft-level degradations should include sample/topk reductions
    assert "reduce-samples" in acts
    assert "reduce-topk" in acts
    # Not necessarily "drop" at soft level
    assert "drop" not in acts


def test_evaluate_hard_overload_and_drop_with_no_min_guarantee_block() -> None:
    fake = FakeQueueRef()
    cfg = BackpressureConfig()
    ctrl = BackpressureController(fake, metrics_hook=None, cfg=cfg)

    # Exceed hard threshold strongly
    fake.set_backlog("token", cfg.max_backlog_seconds_token * 2.0)
    # Push global queue significantly beyond cap to trigger 'very hard'
    fake.set_qglobal(int(cfg.max_queue_len_global * 1.3))
    fake.set_qtenant("tZ", 500)
    # Ensure tenant running meets or exceeds min guarantee -> drop allowed
    fake.set_running("tZ", cfg.tenant_minimum_guarantee)

    bp = ctrl.evaluate("tZ", "token")
    acts = set(bp.actions)
    assert bp.level in ("hard", "soft")
    assert "token->sentence" in acts
    # Under very hard with min guarantee satisfied or global capacity exceeded -> may include drop
    assert "drop" in acts


def test_minimum_guarantee_prevents_drop_when_not_at_global_cap() -> None:
    fake = FakeQueueRef()
    cfg = BackpressureConfig()
    ctrl = BackpressureController(fake, metrics_hook=None, cfg=cfg)

    # Very hard via backlog but keep global below cap so 'drop' depends on min guarantee
    fake.set_backlog("token", cfg.max_backlog_seconds_token * 2.0)
    fake.set_qglobal(int(cfg.max_queue_len_global * 0.5))
    fake.set_qtenant("tG", 300)
    # Running below minimum guarantee => should prevent drop
    fake.set_running("tG", max(0, cfg.tenant_minimum_guarantee - 1))

    bp = ctrl.evaluate("tG", "token")
    acts = set(bp.actions)
    assert "drop" not in acts


def test_attribution_apply_degradation() -> None:
    assert apply_degradation is not None, "apply_degradation should exist in attribution module"
    cfg = AttributionConfig(
        method="shapley",
        max_samples=64,
        early_stop_delta=0.02,
        max_ms_budget=3500,
        random_seed=42,
        min_edge_weight=0.01,
        per_token_incident_cap=256,
    )
    cfg2 = apply_degradation(cfg, ["reduce-samples", "saliency-fallback"])
    assert isinstance(cfg2, AttributionConfig)
    # Halved with floor 16
    assert cfg2.max_samples == 32, f"expected 32, got {cfg2.max_samples}"
    # Doubled with clamp to 0.1
    assert abs(cfg2.early_stop_delta - 0.04) < 1e-9, f"expected 0.04, got {cfg2.early_stop_delta}"
    # Fallback flag present
    assert getattr(cfg2, "force_saliency_fallback", False) is True


def test_hypergraph_degrade_config() -> None:
    assert degrade_hg_config is not None, "degrade_hg_config should exist in hypergraph module"
    # Soft level first
    cfg_soft = HypergraphConfig(min_edge_weight=0.01, per_token_incident_cap=256)
    out_soft = degrade_hg_config(cfg_soft, ["reduce-topk", "reduce-layers"], level="soft")
    assert out_soft.per_token_incident_cap <= 128, "soft reduce-topk should lower per_token_incident_cap to <=128"
    assert out_soft.min_edge_weight >= 0.02, "soft reduce-layers should raise min_edge_weight to >=0.02"

    # Hard level
    cfg_hard = HypergraphConfig(min_edge_weight=0.01, per_token_incident_cap=256)
    out_hard = degrade_hg_config(cfg_hard, ["reduce-topk", "reduce-layers"], level="hard")
    assert out_hard.per_token_incident_cap <= 64, "hard reduce-topk should lower per_token_incident_cap to <=64"
    assert out_hard.min_edge_weight >= 0.05, "hard reduce-layers should raise min_edge_weight to >=0.05"


def _run_all():
    test_evaluate_normal_no_actions()
    test_evaluate_soft_overload_actions_include_reduce()
    test_evaluate_hard_overload_and_drop_with_no_min_guarantee_block()
    test_minimum_guarantee_prevents_drop_when_not_at_global_cap()
    test_attribution_apply_degradation()
    test_hypergraph_degrade_config()
    print("OK: backpressure tests passed")


if __name__ == "__main__":
    _run_all()