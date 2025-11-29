#!/usr/bin/env python3
"""
Microbenchmark planning and autoscaling thresholds (KEDA/MSK).

Inputs:
  --backlog-seconds-slo: "sentence_token", e.g., "1.0,3.0" (defaults 1.0,3.0)
  --observed-service-times-json: JSON from prior runs (default: tools/benchmarks/results/service_times.json)
  --target-rps: chat RPS (default: 500)
  --attach-rate: fraction of chats with explanations (default: 0.3)
  --token-mix: fraction of explanation traffic that is token-level (default: 0.1)
  --gpu-throughput-sentence: explanations/sec/GPU for sentence path (default: 5.0)
  --gpu-throughput-token: explanations/sec/GPU for token path (default: 0.6)

Outputs:
  - tools/benchmarks/results/keda_thresholds.json
  - tools/benchmarks/results/keda_thresholds.md

The backlogSecondsTarget recommended value is derived from the SLO and observed service times,
and the weighted mix across granularity. Min/Max replicas are computed from explanation RPS and
throughput per GPU. Kafka lag thresholds (if MSK is used) are estimated per partition to respect
the backlog-seconds SLO.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Tuple


DEF_OUT_JSON = "tools/benchmarks/results/keda_thresholds.json"
DEF_OUT_MD = "tools/benchmarks/results/keda_thresholds.md"
DEF_SVC_TIMES = "tools/benchmarks/results/service_times.json"


def _ensure_dir_for(path: str) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


def _load_service_times(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Create a sample file if absent for convenience
        sample = {
            "sentence": {"p50": 0.20, "p95": 0.80, "notes": "Sample sentence path timings (s)"},
            "token": {"p50": 2.0, "p95": 5.0, "notes": "Sample token path timings (s)"},
            "queue": {"observed_backlog_seconds": {"mean": 0.4, "p95": 1.2}},
            "kafka": {"enabled": False, "partitions": 12, "notes": "Enable if using MSK; set real partition count"},
            "metadata": {"generated_by": "microbench_plan.py", "note": "sample created automatically"},
        }
        _ensure_dir_for(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sample, f, indent=2, sort_keys=True)
        return sample
    except Exception:
        return {}


def _parse_two_floats(s: str, d1: float, d2: float) -> Tuple[float, float]:
    try:
        parts = [p.strip() for p in (s or "").split(",") if p.strip()]
        if len(parts) == 2:
            a, b = float(parts[0]), float(parts[1])
            return a, b
        if len(parts) == 1:
            a = float(parts[0]); return a, d2
    except Exception:
        pass
    return d1, d2


def _weighted_backlog_target(slo_s: float, slo_t: float, s_p50: float, s_p95: float, t_p50: float, t_p95: float, token_mix: float) -> Tuple[float, float, float]:
    """
    Derive per-granularity backlogSeconds targets, then weight to a single target for Helm values:
      - sentence_target = min(slo_s, max(0.1, min(2*p50, p95)))
      - token_target    = min(slo_t, max(0.5, min(1.5*p50, p95)))
      - combined        = (1-token_mix)*sentence_target + token_mix*token_target
    """
    sentence_target = min(float(slo_s), max(0.1, min(2.0 * float(s_p50), float(s_p95))))
    token_target = min(float(slo_t), max(0.5, min(1.5 * float(t_p50), float(t_p95))))
    combined = (1.0 - token_mix) * sentence_target + token_mix * token_target
    return round(sentence_target, 3), round(token_target, 3), round(combined, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description="Derive KEDA/MSK thresholds from microbenchmark service times and backlog-seconds SLOs.")
    ap.add_argument("--backlog-seconds-slo", default="1.0,3.0", help='Sentence,Token backlog-seconds SLO, e.g. "1.0,3.0" (defaults 1.0,3.0)')
    ap.add_argument("--observed-service-times-json", default=DEF_SVC_TIMES, help="Observed service times JSON path")
    ap.add_argument("--target-rps", type=float, default=500.0, help="Chat RPS (default: 500)")
    ap.add_argument("--attach-rate", type=float, default=0.3, help="Attach rate fraction (default: 0.3)")
    ap.add_argument("--token-mix", type=float, default=0.1, help="Token granularity fraction within explanations (default: 0.1)")
    ap.add_argument("--gpu-throughput-sentence", type=float, default=5.0, help="Sentence explanations/sec/GPU (default: 5.0)")
    ap.add_argument("--gpu-throughput-token", type=float, default=0.6, help="Token explanations/sec/GPU (default: 0.6)")
    ap.add_argument("--output-json", default=DEF_OUT_JSON, help="Output JSON path")
    ap.add_argument("--output-md", default=DEF_OUT_MD, help="Output Markdown path")
    args = ap.parse_args()

    slo_s, slo_t = _parse_two_floats(args.backlog_seconds_slo, 1.0, 3.0)
    svc = _load_service_times(args.observed_service_times_json)

    s_p50 = float(((svc.get("sentence") or {}).get("p50") or 0.2))
    s_p95 = float(((svc.get("sentence") or {}).get("p95") or 0.8))
    t_p50 = float(((svc.get("token") or {}).get("p50") or 2.0))
    t_p95 = float(((svc.get("token") or {}).get("p95") or 5.0))

    token_mix = max(0.0, min(1.0, float(args.token_mix)))
    attach = max(0.0, min(1.0, float(args.attach_rate)))

    # Explanation RPS
    ex_rps = float(args.target_rps) * attach
    s_rps = ex_rps * (1.0 - token_mix)
    t_rps = ex_rps * token_mix

    s_target, t_target, combined = _weighted_backlog_target(slo_s, slo_t, s_p50, s_p95, t_p50, t_p95, token_mix)

    # GPU replicas from throughput
    s_thr = max(1e-9, float(args.gpu_throughput_sentence))
    t_thr = max(1e-9, float(args.gpu_throughput_token))
    s_repl = int(math.ceil(s_rps / s_thr))
    t_repl = int(math.ceil(t_rps / t_thr))
    total_repl = s_repl + t_repl

    # Initial KEDA min/max (tunable)
    min_repl = max(2, int(math.floor(0.8 * total_repl)))
    max_repl = max(min_repl + 1, int(math.ceil(1.5 * total_repl)))

    # Kafka (MSK) lag threshold per partition if enabled
    kafka_cfg = svc.get("kafka") or {}
    kafka_enabled = bool(kafka_cfg.get("enabled", False))
    partitions = int(kafka_cfg.get("partitions", 12))
    lag_per_partition = None
    if kafka_enabled and partitions > 0:
        # Keep backlog seconds under combined target => lag ~= RPS_per_partition * target_seconds
        rps_per_part = ex_rps / float(partitions)
        lag_per_partition = int(max(1, round(rps_per_part * float(combined))))
    else:
        lag_per_partition = None  # no kafka

    out = {
        "inputs": {
            "backlog_seconds_slo": {"sentence": slo_s, "token": slo_t},
            "observed_service_times": {"sentence": {"p50": s_p50, "p95": s_p95}, "token": {"p50": t_p50, "p95": t_p95}},
            "target_rps": float(args.target_rps),
            "attach_rate": attach,
            "token_mix": token_mix,
            "gpu_throughput": {"sentence": float(args.gpu_throughput_sentence), "token": float(args.gpu_throughput_token)},
        },
        "recommended": {
            "keda_backlogSecondsTarget": {
                "sentence": s_target,
                "token": t_target,
                "weighted": combined
            },
            "explainer_replicas": {
                "minReplicaCount": min_repl,
                "maxReplicaCount": max_repl,
                "derivation": {"sentence": s_repl, "token": t_repl, "total": total_repl}
            },
            "kafka": {
                "enabled": kafka_enabled,
                "partitions": partitions,
                "lagThresholdPerPartition": lag_per_partition,
                "comment": "Set only if using MSK and Kafka trigger; else ignore."
            }
        }
    }

    _ensure_dir_for(args.output_json)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    # Markdown summary
    lines = []
    lines.append("# KEDA/MSK autoscaling thresholds (microbenchmark-derived)")
    lines.append("")
    lines.append(f"- Target chat RPS: {float(args.target_rps)}")
    lines.append(f"- Attach rate: {attach}")
    lines.append(f"- Token mix: {token_mix} (sentence={round(1.0-token_mix,3)})")
    lines.append("")
    lines.append("## Backlog seconds targets")
    lines.append("")
    lines.append(f"- Sentence: {s_target} s")
    lines.append(f"- Token:    {t_target} s")
    lines.append(f"- Weighted (set in Helm values): {combined} s")
    lines.append("")
    lines.append("## Explainer replicas (initial)")
    lines.append("")
    lines.append(f"- minReplicaCount: {min_repl}")
    lines.append(f"- maxReplicaCount: {max_repl}")
    lines.append(f"- Derived total (rounded): {total_repl} (sentence={s_repl}, token={t_repl})")
    lines.append("")
    if kafka_enabled and lag_per_partition is not None:
        lines.append("## Kafka (MSK) lag thresholds")
        lines.append("")
        lines.append(f"- Partitions: {partitions}")
        lines.append(f"- Recommended lagThreshold (per partition): {lag_per_partition}")
    else:
        lines.append("## Kafka (MSK)")
        lines.append("")
        lines.append("- Not enabled in observed_service_times.json. If using MSK, enable and set partition count; rerun.")
    lines.append("")
    lines.append("Wire values into Helm:")
    lines.append("- Template: [manifests/helm/hypergraph/templates/keda-scalers.yaml](manifests/helm/hypergraph/templates/keda-scalers.yaml)")
    lines.append("- Values:   [manifests/helm/hypergraph/values.yaml](manifests/helm/hypergraph/values.yaml)")
    lines.append("")
    lines.append("For Kafka KEDA trigger, sample (edit to your env):")
    lines.append("```yaml")
    lines.append("# In ScaledObject.triggers (see keda-scalers.yaml for placement)")
    lines.append("- type: kafka")
    lines.append("  metadata:")
    lines.append("    bootstrapServers: \"<broker-1:9092,broker-2:9092>\"")
    lines.append("    consumerGroup: \"hypergraph-explainer\"")
    lines.append("    topic: \"explanations\"")
    lines.append("    lagThreshold: \"<per-partition-lag>\"  # from this plan")
    lines.append("```")
    _ensure_dir_for(args.output_md)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    try:
        print(json.dumps({"weighted_backlogSecondsTarget": out["recommended"]["keda_backlogSecondsTarget"]["weighted"], "minReplicaCount": min_repl, "maxReplicaCount": max_repl}))
    except Exception:
        pass


if __name__ == "__main__":
    main()
