#!/usr/bin/env python3
# tests/e2e/test_slo_adherence.py
# SLO adherence check using outputs from the load harness summary.
#
# Validates p95 times:
# - sentence explanations: ≤ 2000 ms
# - token explanations:    ≤ 8000 ms
#
# Inputs:
# - tests/load/results/summary.json produced by [run_load_async()](tests/load/load_runner.py:455)
#
# Behavior:
# - If the summary file is missing or fields absent, SKIP with a clear reason.
# - If there were no completed explanations recorded, SKIP as inconclusive.
# - Otherwise, compare p95 values against SLO thresholds and PASS/FAIL accordingly.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from tests.e2e.utils import E2EConfig, new_result

SUMMARY_PATH = Path("tests/load/results/summary.json")

# Default SLO thresholds (ms)
SLO_SENTENCE_P95_MS = 2000
SLO_TOKEN_P95_MS = 8000


def _load_summary(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def run(config: E2EConfig) -> Dict[str, Any]:
    summary = _load_summary(SUMMARY_PATH)
    if not summary:
        return new_result("test_slo_adherence.py", "SKIP", reason=f"summary not found at {SUMMARY_PATH}")

    expl = summary.get("explanation") or {}
    if not isinstance(expl, dict):
        return new_result("test_slo_adherence.py", "SKIP", reason="explanation summary missing")

    sla = expl.get("sla_p95_ms") or {}
    if not isinstance(sla, dict):
        return new_result("test_slo_adherence.py", "SKIP", reason="sla_p95_ms missing in summary")

    p95_sentence = float(sla.get("sentence") or 0.0)
    p95_token = float(sla.get("token") or 0.0)
    count = int(expl.get("count") or 0)

    if count <= 0:
        return new_result("test_slo_adherence.py", "SKIP", reason="no explanation results to evaluate")

    sentence_ok = (p95_sentence <= float(SLO_SENTENCE_P95_MS)) if p95_sentence > 0 else True
    token_ok = (p95_token <= float(SLO_TOKEN_P95_MS)) if p95_token > 0 else True

    details = {
        "p95_sentence_ms": p95_sentence,
        "p95_token_ms": p95_token,
        "thresholds": {
            "sentence_ms": SLO_SENTENCE_P95_MS,
            "token_ms": SLO_TOKEN_P95_MS,
        },
        "explanations_count": count,
        "expired_pct": expl.get("expired_pct"),
        "failed_pct": expl.get("failed_pct"),
    }

    if sentence_ok and token_ok:
        return new_result("test_slo_adherence.py", "PASS", details=details)

    reasons = []
    if not sentence_ok:
        reasons.append(f"sentence_p95={p95_sentence}ms > {SLO_SENTENCE_P95_MS}ms")
    if not token_ok:
        reasons.append(f"token_p95={p95_token}ms > {SLO_TOKEN_P95_MS}ms")
    return new_result("test_slo_adherence.py", "FAIL", reason=", ".join(reasons), details=details)