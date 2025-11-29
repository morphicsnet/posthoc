#!/usr/bin/env python3
"""
E2E Quickstart script: send a chat request to the Gateway, request explainability, poll for the HIF graph,
write it to disk, and print the top features.

References:
- Python SDK Client [Client](sdks/python/your_company_explainability/your_company_explainability/client.py:143)
- API Reference [docs/api-reference.md](docs/api-reference.md:1)
"""

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Try to import the installed SDK first; fall back to repo-relative path
try:
    from your_company_explainability.client import Client  # type: ignore
except Exception:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root / "sdks" / "python" / "your_company_explainability"))
    from your_company_explainability.client import Client  # type: ignore

BASE_URL = os.environ.get("EXPLAINABILITY_BASE_URL", "http://localhost:8080")
API_TOKEN = os.environ.get("API_TOKEN")  # optional


def top_features_from_hif(hif: Dict[str, Any], top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Aggregate feature importance by summing incidence weights for nodes of type 'sae_feature'.
    Returns top-k (feature_id, score).
    """
    nodes = hif.get("nodes") or []
    incidences = hif.get("incidences") or []
    node_type = {n.get("id"): n.get("type") for n in nodes if isinstance(n, dict)}
    scores: Dict[str, float] = {}
    for inc in incidences:
        if not isinstance(inc, dict):
            continue
        weight = float(inc.get("weight") or 0.0)
        for nid in inc.get("node_ids") or []:
            if node_type.get(nid) == "sae_feature":
                scores[nid] = scores.get(nid, 0.0) + weight
    out = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return out[:top_k]


def main() -> int:
    print(f"[info] Using base_url={BASE_URL}")
    try:
        client = Client(base_url=BASE_URL, api_key=API_TOKEN)
    except Exception as e:
        print(f"[error] Failed to initialize Client: {e}")
        return 1

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain why a model might refuse a harmful instruction."},
            ],
            stream=False,
            explain=True,
            explain_granularity="sentence",  # or "token"
            explain_features="sae-gpt4-2m",
            explain_budget_ms=1500,
            idempotency_key="examples-python-e2e-1",
        )
    except Exception as e:
       print(f"[error] POST /v1/chat/completions failed: {e}")
       return 2

    try:
        hif = res.get_explanation(max_wait_seconds=30, poll_interval=0.5)
    except TimeoutError as te:
        print(f"[timeout] {te}")
        return 3
    except RuntimeError as re:
        print(f"[error] {re}")
        return 4
    finally:
        try:
            client.close()
        except Exception:
            pass

    if not hif:
        print("[warn] No explanation returned within the allotted time.")
        return 5

    # Persist HIF to disk
    out_path = Path.cwd() / "hif_graph.json"
    try:
        out_path.write_text(json.dumps(hif, indent=2), encoding="utf-8")
        print(f"[ok] Wrote HIF graph to: {out_path}")
    except Exception as e:
        print(f"[error] Failed to write HIF: {e}")

    # Show top features by aggregated incidence weights
    top_feats = top_features_from_hif(hif, top_k=5)
    if top_feats:
        print("Top features (by sum of incidence weights):")
        for fid, score in top_feats:
            print(f"  {fid}: {score:.3f}")
    else:
        print("[info] No SAE features present in graph.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())