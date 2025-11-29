from __future__ import annotations

# Minimal assert-based tests for hypergraph constructor/pruner

from typing import List, Dict, Any, Optional

try:
    # Primary import path
    from services.explainer.src.hypergraph import (
        HypergraphConfig,
        prune_and_group,
        build_hif,
    )
except Exception:  # pragma: no cover - fallback if package layout differs
    from hypergraph import (  # type: ignore
        HypergraphConfig,
        prune_and_group,
        build_hif,
    )

try:
    from libs.hif.validator import validate_hif as schema_validate  # type: ignore
except Exception:
    schema_validate = None  # type: ignore


def _token_node() -> Dict[str, Any]:
    return {"id": "token_out_1", "type": "output_token", "label": "Test output", "position": 1}


def test_edge_thresholding_drops_low_weights() -> None:
    nodes = [
        {"id": "f1", "type": "sae_feature", "label": "A"},
        {"id": "f2", "type": "sae_feature", "label": "B"},
        _token_node(),
    ]
    incidences = [
        {"id": "e1", "node_ids": ["f1", "token_out_1"], "weight": 0.009},
        {"id": "e2", "node_ids": ["f2", "token_out_1"], "weight": 0.5},
    ]
    cfg = HypergraphConfig(min_edge_weight=0.01, grouping="none")
    pn, pe, meta = prune_and_group(nodes, incidences, cfg)

    assert len(pe) == 1, f"Expected 1 edge after thresholding, got {len(pe)}"
    kept = pe[0]
    assert kept["id"] == "e2" or kept["weight"] == 0.5
    # Ensure dangling node f1 is removed
    ref_ids = set()
    for e in pe:
        for nid in e["node_ids"]:
            ref_ids.add(nid)
    for n in pn:
        assert n["id"] in ref_ids, f"Dangling node present: {n['id']}"
    assert "f1" not in ref_ids, "Low-weight feature should not be referenced"


def test_per_token_cap_enforced() -> None:
    # Build 30 edges to a single output token; cap at 10 (keep top by weight)
    nodes = [_token_node()]
    incidences: List[Dict[str, Any]] = []
    for i in range(30):
        fid = f"f{i:03d}"
        nodes.append({"id": fid, "type": "sae_feature", "label": f"F{i}"})
        w = 0.001 * (i + 1)  # strictly increasing
        incidences.append({"id": f"e{i:03d}", "node_ids": [fid, "token_out_1"], "weight": w})

    cfg = HypergraphConfig(
        min_edge_weight=0.0,  # keep all by threshold
        per_token_incident_cap=10,
        grouping="none",
        max_incidences=20000,
        max_nodes=5000,
    )
    pn, pe, meta = prune_and_group(nodes, incidences, cfg)
    assert len(pe) == 10, f"Expected 10 edges after per-token cap, got {len(pe)}"

    # Determine expected top-10 feature ids by weight
    top10 = sorted(incidences, key=lambda e: e["weight"], reverse=True)[:10]
    exp_fids = {e["node_ids"][0] for e in top10}
    got_fids = {e["node_ids"][0] for e in pe}
    assert got_fids == exp_fids, f"Top-k mismatch: expected {sorted(exp_fids)}, got {sorted(got_fids)}"


def test_supernode_grouping_and_remap() -> None:
    # Two Biology features share prefix -> cluster into one circuit_supernode; Physics stays separate
    nodes = [
        {"id": "f1", "type": "sae_feature", "label": "Biology: gene"},
        {"id": "f2", "type": "sae_feature", "label": "Biology: cell"},
        {"id": "f3", "type": "sae_feature", "label": "Physics: mass"},
        _token_node(),
    ]
    incidences = [
        {"id": "e1", "node_ids": ["f1", "token_out_1"], "weight": 0.30},
        {"id": "e2", "node_ids": ["f2", "token_out_1"], "weight": 0.40},
        {"id": "e3", "node_ids": ["f3", "token_out_1"], "weight": 0.20},
    ]
    cfg = HypergraphConfig(
        min_edge_weight=0.0,
        per_token_incident_cap=256,
        grouping="supernode",
        supernode_min_group=2,
        supernode_label_delim=":",
        max_incidences=20000,
        max_nodes=5000,
    )
    pn, pe, meta = prune_and_group(nodes, incidences, cfg)

    # Expect a single Biology supernode
    super_nodes = [n for n in pn if n.get("type") == "circuit_supernode"]
    assert len(super_nodes) == 1, f"Expected 1 supernode, found {len(super_nodes)}"
    super_id = super_nodes[0]["id"]
    assert super_id.startswith("super_biology"), f"Unexpected supernode id: {super_id}"

    # Incidences referencing f1/f2 should be merged as a single incidence pointing to the supernode
    # and token_out_1, with weight=max(0.30,0.40)=0.40
    # Along with the Physics edge -> total of 2 edges
    assert len(pe) == 2, f"Expected 2 incidences after grouping/merge, got {len(pe)}"
    # Find edge involving supernode
    sup_edges = [e for e in pe if super_id in e["node_ids"]]
    assert len(sup_edges) == 1, "Expected a single merged incidence for Biology supernode"
    sup_e = sup_edges[0]
    assert abs(float(sup_e["weight"]) - 0.40) < 1e-9, f"Merged weight should be max=0.40, got {sup_e['weight']}"

    # Original f1/f2 should be removed from nodes; f3 remains
    node_ids = {n["id"] for n in pn}
    assert "f1" not in node_ids and "f2" not in node_ids, "Grouped feature nodes should be removed"
    assert "f3" in node_ids, "Ungrouped Physics feature should remain"


def test_global_caps_weight_and_consistency() -> None:
    # Build 50 edges descending by weight; cap incidences at 20; ensure determinism and no dangling nodes.
    nodes = [_token_node()]
    incidences: List[Dict[str, Any]] = []
    for i in range(50):
        fid = f"f{i:03d}"
        nodes.append({"id": fid, "type": "sae_feature", "label": f"F{i}"})
        w = max(0.0, 1.0 - 0.01 * i)  # 1.00, 0.99, ..., 0.51
        incidences.append({"id": f"e{i:03d}", "node_ids": [fid, "token_out_1"], "weight": w})

    cfg = HypergraphConfig(
        min_edge_weight=0.0,
        per_token_incident_cap=1000,
        grouping="none",
        max_incidences=20,
        max_nodes=25,  # enough to keep token + 20 features
    )
    pn, pe, meta = prune_and_group(nodes, incidences, cfg)
    assert len(pe) == 20, f"Expected 20 incidences after global cap, got {len(pe)}"

    # Check top-20 by weight kept
    exp_top20 = sorted(incidences, key=lambda e: e["weight"], reverse=True)[:20]
    exp_ids = {e["id"] for e in exp_top20}
    got_ids = {e["id"] for e in pe}
    # If IDs tie-break differ after normalization, compare by feature ids instead
    if got_ids != exp_ids:
        exp_fids = {e["node_ids"][0] for e in exp_top20}
        got_fids = {e["node_ids"][0] for e in pe}
        assert got_fids == exp_fids, f"Top-20 feature mismatch: expected {sorted(exp_fids)}, got {sorted(got_fids)}"

    # Node set should be exactly token + features used by kept incidences (<= 21 with cap)
    ref_ids = set()
    for e in pe:
        for nid in e["node_ids"]:
            ref_ids.add(nid)
    node_ids = {n["id"] for n in pn}
    assert node_ids == ref_ids, "Node set must match referenced ids (no dangling edges)"
    assert "token_out_1" in node_ids, "Output token must be preserved"


def test_build_hif_and_optional_validation() -> None:
    # Small graph -> build HIF v1 legacy object and validate if validator is available
    nodes = [
        {"id": "f1", "type": "sae_feature", "label": "Biology: gene"},
        _token_node(),
    ]
    incidences = [{"id": "e1", "node_ids": ["f1", "token_out_1"], "weight": 0.75, "metadata": {"type": "causal_circuit", "method": "test"}}]
    cfg = HypergraphConfig()
    meta = {
        "model_name": "gpt-4-turbo",
        "model_hash": "devhash1234",
        "sae_dictionary": "sae-gpt4-2m",
        "granularity": "sentence",
    }
    hif = build_hif(nodes, incidences, meta, cfg)
    assert isinstance(hif, dict)
    assert hif.get("network-type") == "directed"
    assert isinstance(hif.get("nodes"), list)
    assert isinstance(hif.get("incidences"), list)
    limits = hif.get("meta", {}).get("limits", {})
    assert {"min_edge_weight", "max_nodes", "max_incidences"} <= set(limits.keys()), "limits should be populated from cfg"

    # Optional schema validation (jsonschema required)
    if schema_validate is not None:
        try:
            schema_validate(hif)  # accepts v1 legacy or v2 hypergraph
        except Exception as e:
            raise AssertionError(f"Schema validation failed unexpectedly: {e}") from e


def _run_all() -> None:
    # Execute tests sequentially; raise on first failure
    tests = [
        test_edge_thresholding_drops_low_weights,
        test_per_token_cap_enforced,
        test_supernode_grouping_and_remap,
        test_global_caps_weight_and_consistency,
        test_build_hif_and_optional_validation,
    ]
    for t in tests:
        t()
    print("All hypergraph tests passed.")


if __name__ == "__main__":
    _run_all()