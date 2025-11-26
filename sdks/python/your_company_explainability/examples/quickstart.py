#!/usr/bin/env python3
"""
Quickstart example for Your Company Explainability SDK.

- Configure base_url and optional API key via env or inline.
- Send a prompt with explainability enabled.
- Poll for a hypergraph and print a small preview.
"""

import os
from your_company_explainability.client import Client

def _extract_graph(hif):
    if not hif:
        return None
    # Accept either envelope with "hypergraph" or raw graph
    graph = hif.get("hypergraph") if isinstance(hif, dict) and "hypergraph" in hif else hif
    return graph

def main():
    base_url = os.environ.get("EXPLAINABILITY_BASE_URL", "http://localhost:8080")
    api_key = os.environ.get("EXPLAINABILITY_API_KEY")

    client = Client(base_url=base_url, api_key=api_key)

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Explain the greenhouse effect in 2 sentences."}],
        stream=False,
        explain=True,
        explain_granularity="sentence",
    )

    hif = res.get_explanation(max_wait_seconds=45)
    if not hif:
        print("No explanation available within timeout.")
        return

    graph = _extract_graph(hif) or {}
    nodes = graph.get("nodes", []) or []
    edges = graph.get("hyperedges", graph.get("edges", graph.get("links", []))) or []

    print(f"Nodes: {len(nodes)}, Edges: {len(edges)}")
    print("Sample nodes:", nodes[:3])
    print("Sample edges:", edges[:3])

if __name__ == "__main__":
    main()
