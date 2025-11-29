# Python Quickstart: Chat + Explainability (HIF v1)

This guide shows how to:
- Send an OpenAI-compatible chat request to the Gateway
- Request explainability via headers
- Poll and fetch the HIF v1 graph
- Inspect top features and write the graph to disk
- Handle errors, timeouts, and optional auth

References:
- Python SDK [Client](sdks/python/your_company_explainability/your_company_explainability/client.py:143), [ChatCompletions.create()](sdks/python/your_company_explainability/your_company_explainability/client.py:97), and [ChatResponse.get_explanation()](sdks/python/your_company_explainability/your_company_explainability/client.py:31)
- API Reference: [docs/api-reference.md](docs/api-reference.md:1)
- HIF Schema Reference (v1): [docs/hif-schema.md](docs/hif-schema.md:1)
- HIF validator CLI: [libs/hif/validator.py](libs/hif/validator.py:1)

Base URL (local): http://localhost:8080

## Install the SDK (editable)

From the repository root:
```bash
pip install -e ./sdks/python/your_company_explainability
```

Alternatively, for a one-off example run without installing, add the package directory to `PYTHONPATH` (not recommended for production):
```bash
export PYTHONPATH="$PWD/sdks/python/your_company_explainability:$PYTHONPATH"
```

## Minimal example

```python
#!/usr/bin/env python3
"""
Minimal Python example: send chat, request explainability, fetch HIF v1,
print top features, and write the graph to disk.

Requires:
  - Gateway running at http://localhost:8080
  - Optional: API token in $API_TOKEN (if your deployment enforces auth)
"""

import os, json, time
from typing import Dict, Any, List, Tuple

# Prefer installed package:
#   from your_company_explainability.client import Client
# If running from repo without install:
import sys
sys.path.append(os.path.join(os.getcwd(), "sdks", "python", "your_company_explainability"))
from your_company_explainability.client import Client  # noqa: E402

BASE_URL = os.environ.get("EXPLAINABILITY_BASE_URL", "http://localhost:8080")
API_TOKEN = os.environ.get("API_TOKEN")  # optional

def top_features_from_hif(hif: Dict[str, Any], top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Aggregate feature importance by summing incidence weights for nodes of type 'sae_feature'.
    Returns top-k (feature_id, score).
    """
    nodes = hif.get("nodes", [])
    incidences = hif.get("incidences", [])
    node_type = {n.get("id"): n.get("type") for n in nodes if isinstance(n, dict)}
    scores: Dict[str, float] = {}
    for inc in incidences:
        if not isinstance(inc, dict):
            continue
        weight = float(inc.get("weight", 0.0) or 0.0)
        ids = inc.get("node_ids") or []
        for nid in ids:
            if node_type.get(nid) == "sae_feature":
                scores[nid] = scores.get(nid, 0.0) + weight
    out = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return out[:top_k]

def main():
    with Client(base_url=BASE_URL, api_key=API_TOKEN) as client:
        # Send a non-streaming request and ask for explainability via headers
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain why a model might refuse a risky or harmful instruction."},
            ],
            stream=False,
            explain=True,
            explain_granularity="sentence",  # or "token"
            explain_features="sae-gpt4-2m",
            explain_budget_ms=1500,
            idempotency_key="example-python-qs-1",
        )

        # Fetch explanation graph with polling and basic failure handling
        try:
            hif = res.get_explanation(max_wait_seconds=30, poll_interval=0.5)
        except TimeoutError as te:
            print(f"[timeout] {te}")
            return
        except RuntimeError as re:
            print(f"[error] {re}")
            return

        if not hif:
            print("No explanation returned within the allotted time.")
            return

        # Persist HIF to disk
        out_path = "hif_graph.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(hif, f, indent=2)
        print(f"Wrote HIF graph to: {out_path}")

        # Show top features by aggregated incidence weights
        top_feats = top_features_from_hif(hif, top_k=5)
        print("Top features (by sum of incidence weights):")
        for fid, score in top_feats:
            print(f"  {fid}: {score:.3f}")

if __name__ == "__main__":
    main()
```

Notes:
- The SDK automatically sets required headers. Explainability is enabled when you pass `explain=True` (maps to `x-explain-mode: hypergraph`) along with granularity/features/budget.
- On success, the chat response includes `explanation_metadata.trace_id`. The helper [ChatResponse.get_explanation()](sdks/python/your_company_explainability/your_company_explainability/client.py:31) polls:
  - GET /v1/traces/{trace_id}/status
  - GET /v1/traces/{trace_id}/graph
  handling 404 (not ready) and 410 (expired).

## Advanced: Manual polling and error handling

If you want full control, poll status first and then fetch the graph:

```python
import httpx

trace_id = "trc_..."  # from explanation_metadata.trace_id
headers = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}

with httpx.Client(timeout=10.0) as http:
    deadline = time.time() + 30.0
    last_state = None
    while time.time() < deadline:
        st = http.get(f"{BASE_URL}/v1/traces/{trace_id}/status", headers=headers)
        if st.status_code == 200:
            state = st.json().get("state")
            if state in ("failed", "canceled", "expired"):
                raise RuntimeError(f"Trace ended with state={state}")
            last_state = state

        gr = http.get(f"{BASE_URL}/v1/traces/{trace_id}/graph", headers=headers)
        if gr.status_code == 200:
            hif = gr.json()
            break
        if gr.status_code == 410:
            raise RuntimeError("Trace expired (410)")
        time.sleep(0.5)
    else:
        raise TimeoutError("Graph not ready within timeout")
```

## Optional auth

If your Gateway enforces auth, export a token and pass it to the SDK:

```bash
export API_TOKEN="sk-..."
```

The SDK adds `Authorization: Bearer ...` when `api_key` is provided.

## Validate HIF with the CLI

Use the validator in [validator.py](libs/hif/validator.py:1):

```bash
python libs/hif/validator.py ./hif_graph.json
```

## Also see

- End-to-end tutorial: [docs/tutorials/e2e.md](docs/tutorials/e2e.md:1)
- Runnable script version in this repo: [examples/python/e2e_quickstart.py](examples/python/e2e_quickstart.py:1)