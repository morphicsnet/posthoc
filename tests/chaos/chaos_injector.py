#!/usr/bin/env python3
# tests/chaos/chaos_injector.py
# Toggle chaos injectors by writing a local control file read by the Explainer worker.
# Supported flags (top-level keys in /tmp/hif/chaos.json):
#  - "drop-s3":            { "enabled": true,  "percent": 10 }
#  - "slow-sae":           { "enabled": true,  "percent": 50, "jitter_ms": 150 }
#  - "fail-attribution":   { "enabled": true,  "percent": 20, "mode": "fallback|fail" }
#  - "rate-limit-spike":   { "enabled": true,  "severity": "soft|hard" }
#
# This does not change public APIs. The worker periodically reads this file best-effort.

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict

DEFAULT_PATH = "/tmp/hif/chaos.json"


def _read(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _atomic_write(path: str, payload: Dict[str, Any]) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".chaos.", dir=d, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"), ensure_ascii=False, indent=None)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass


def _apply_enable(cfg: Dict[str, Any], name: str, args: argparse.Namespace) -> None:
    cur = cfg.get(name)
    if not isinstance(cur, dict):
        cur = {}
    cur["enabled"] = True
    if args.percent is not None:
        try:
            p = max(0.0, min(100.0, float(args.percent)))
        except Exception:
            p = 0.0
        cur["percent"] = p
    if args.jitter_ms is not None:
        try:
            cur["jitter_ms"] = int(args.jitter_ms)
        except Exception:
            pass
    if args.mode:
        cur["mode"] = str(args.mode).lower()
    if args.severity:
        cur["severity"] = str(args.severity).lower()
    cfg[name] = cur


def _apply_disable(cfg: Dict[str, Any], name: str) -> None:
    cur = cfg.get(name)
    if isinstance(cur, dict):
        cur["enabled"] = False
        cfg[name] = cur
    else:
        cfg[name] = {"enabled": False}


def _apply_delete(cfg: Dict[str, Any], name: str) -> None:
    if name in cfg:
        del cfg[name]


def main() -> int:
    parser = argparse.ArgumentParser(description="Chaos injector control writer")
    parser.add_argument("--path", default=os.environ.get("CHAOS_CONTROL_PATH", DEFAULT_PATH),
                        help=f"Path to chaos control JSON (default: {DEFAULT_PATH})")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--enable", metavar="FLAG", help="Enable a chaos flag (drop-s3|slow-sae|fail-attribution|rate-limit-spike)")
    grp.add_argument("--disable", metavar="FLAG", help="Disable a chaos flag")
    grp.add_argument("--delete", metavar="FLAG", help="Remove a chaos flag entry")
    grp.add_argument("--disable-all", action="store_true", help="Disable all known chaos flags")

    parser.add_argument("--percent", type=float, help="Percentage probability (0-100) for drop-s3/slow-sae/fail-attribution")
    parser.add_argument("--jitter-ms", type=int, help="Artificial latency in ms (slow-sae)")
    parser.add_argument("--mode", choices=["fallback", "fail"], help="Mode for fail-attribution")
    parser.add_argument("--severity", choices=["soft", "hard"], help="Severity for rate-limit-spike")

    args = parser.parse_args()

    known = ["drop-s3", "slow-sae", "fail-attribution", "rate-limit-spike"]
    path = args.path
    cfg: Dict[str, Any] = _read(path)

    if args.disable_all:
        for k in known:
            _apply_disable(cfg, k)
    elif args.enable:
        if args.enable not in known:
            print(f"Unknown flag: {args.enable}. Known: {known}", file=sys.stderr)
            return 2
        _apply_enable(cfg, args.enable, args)
    elif args.disable:
        if args.disable not in known:
            print(f"Unknown flag: {args.disable}. Known: {known}", file=sys.stderr)
            return 2
        _apply_disable(cfg, args.disable)
    elif args.delete:
        _apply_delete(cfg, args.delete)

    _atomic_write(path, cfg)
    # Echo resulting config (single line) for convenience
    try:
        print(json.dumps(cfg, separators=(",", ":"), ensure_ascii=False))
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())