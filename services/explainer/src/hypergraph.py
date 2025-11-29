from __future__ import annotations

import time
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Set

# Observability (optional)
try:
    from services.explainer.src import otel as _otel  # type: ignore
except Exception:
    try:
        import otel as _otel  # type: ignore
    except Exception:
        _otel = None  # type: ignore


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class HypergraphConfig:
    """
    Pruning/grouping configuration for HIF v1 legacy graphs (see [`libs/hif/schema.json`](libs/hif/schema.json:1)).

    - min_edge_weight: drop incidences with weight < threshold
    - per_token_incident_cap: per-output-token cap on incidences (keep top by weight)
    - max_nodes / max_incidences: global guardrails
    - grouping: "supernode" enables SAE feature clustering by label prefix; "none" disables
    - supernode_min_group: minimum members required to form a supernode from a prefix group
    - supernode_label_delim: delimiter to split label prefix (e.g., "Biology: ..." -> "Biology")
    """
    min_edge_weight: float = 0.01
    per_token_incident_cap: int = 256
    max_nodes: int = 5000
    max_incidences: int = 20000
    grouping: str = "supernode"  # "supernode" | "none"
    supernode_min_group: int = 3
    supernode_label_delim: str = ":"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

_ALLOWED_NODE_TYPES_V1: Set[str] = {
    "sae_feature",
    "input_token",
    "output_token",
    "circuit_supernode",
}

_ALLOWED_INCIDENCE_META_KEYS: Set[str] = {
    "type",
    "method",
    "window",
    "description",
    "provenance",
}

_ALLOWED_PROVENANCE_KEYS: Set[str] = {
    "capture_layers",
    "sae_version",
    "model_hash",
}


def _slugify_prefix(prefix: str) -> str:
    s = (prefix or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "cluster"
    return s[:80]


def _clamp01(x: Any) -> float:
    try:
        f = float(x)
    except Exception:
        f = 0.0
    if f != f:  # NaN
        f = 0.0
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


def _node_priority(n: Dict[str, Any]) -> Tuple[int, float, str]:
    """
    Sorting priority for node truncation:
    - tokens (input/output) first (0)
    - circuit_supernode (1)
    - sae_feature (2)
    Then by descending activation_strength, then by id for determinism.
    """
    t = str(n.get("type") or "")
    if t in ("input_token", "output_token"):
        pri = 0
    elif t == "circuit_supernode":
        pri = 1
    else:
        pri = 2
    try:
        act = float(n.get("activation_strength", 0.0))
    except Exception:
        act = 0.0
    return (pri, -act, str(n.get("id") or ""))


def _token_last_order(ids: List[str], id2type: Dict[str, str]) -> List[str]:
    # Stable order for node_ids: non-tokens first, tokens last
    def key(nid: str) -> Tuple[int, str]:
        t = id2type.get(nid, "")
        token_flag = 1 if t in ("input_token", "output_token") else 0
        return (token_flag, nid)
    return sorted(ids, key=key)


def _sanitize_incidence_metadata(meta: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(meta, dict):
        return None
    out: Dict[str, Any] = {}
    for k in list(meta.keys()):
        if k not in _ALLOWED_INCIDENCE_META_KEYS:
            continue
        if k == "provenance":
            prov = meta.get("provenance")
            if isinstance(prov, dict):
                out["provenance"] = {pk: pv for pk, pv in prov.items() if pk in _ALLOWED_PROVENANCE_KEYS}
            continue
        out[k] = meta[k]
    return out if out else None


# -----------------------------------------------------------------------------
# Normalization helpers (schema-aligned)
# -----------------------------------------------------------------------------

def _normalize_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, n in enumerate(nodes, start=1):
        nid = str(n.get("id") or f"n{i}")
        t = str(n.get("type") or "").strip()
        if t not in _ALLOWED_NODE_TYPES_V1:
            # Best-effort: treat unknown as SAE feature
            t = "sae_feature"
        node: Dict[str, Any] = {"id": nid, "type": t}

        # label
        label = n.get("label")
        if not isinstance(label, str) or not label.strip():
            if t == "sae_feature":
                label = f"Feature {nid}"
            else:
                label = nid
        node["label"] = str(label)

        # Optional ints
        if isinstance(n.get("layer"), int):
            node["layer"] = int(n["layer"])
        if isinstance(n.get("position"), int):
            node["position"] = int(n["position"])

        # Optional floats
        try:
            if n.get("activation_strength") is not None:
                node["activation_strength"] = float(n["activation_strength"])
        except Exception:
            pass

        # attributes (pass-through, additionalProperties true)
        if isinstance(n.get("attributes"), dict):
            node["attributes"] = dict(n["attributes"])

        out.append(node)
    return out


def _normalize_incidences(incidences: List[Dict[str, Any]], id2type: Dict[str, str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, e in enumerate(incidences, start=1):
        eid = str(e.get("id") or f"e{i}")
        node_ids_raw = e.get("node_ids") or []
        if not isinstance(node_ids_raw, list):
            node_ids_raw = []
        node_ids = [str(x) for x in node_ids_raw if isinstance(x, (str, int))]
        if len(node_ids) < 2:
            # invalid shape; skip
            continue
        w = _clamp01(e.get("weight", 0.0))

        meta = _sanitize_incidence_metadata(e.get("metadata"))

        inc: Dict[str, Any] = {"id": eid, "node_ids": _token_last_order(node_ids, id2type), "weight": w}
        if meta is not None:
            inc["metadata"] = meta
        # attributes allowed, pass-through
        if isinstance(e.get("attributes"), dict):
            inc["attributes"] = dict(e["attributes"])
        out.append(inc)
    return out


def degrade_hg_config(cfg: "HypergraphConfig", actions: List[str], level: str = "soft") -> "HypergraphConfig":
    """
    Apply stronger pruning under backpressure:
    - reduce-topk: lower per_token_incident_cap (soft→128, hard→64)
    - reduce-layers or saliency-fallback: raise min_edge_weight (soft→0.02, hard→0.05)
    Mutates and returns cfg for convenience.
    """
    try:
        acts = set(actions or [])
    except Exception:
        acts = set()
    try:
        if "reduce-topk" in acts and hasattr(cfg, "per_token_incident_cap"):
            cur = int(getattr(cfg, "per_token_incident_cap", 256))
            target = 64 if str(level).lower() == "hard" else 128
            setattr(cfg, "per_token_incident_cap", max(8, min(cur, target)))
        if ("reduce-layers" in acts or "saliency-fallback" in acts) and hasattr(cfg, "min_edge_weight"):
            curw = float(getattr(cfg, "min_edge_weight", 0.01))
            targetw = 0.05 if str(level).lower() == "hard" else 0.02
            setattr(cfg, "min_edge_weight", max(curw, targetw))
    except Exception:
        pass
    return cfg

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def prune_and_group(
    nodes: List[Dict[str, Any]],
    incidences: List[Dict[str, Any]],
    cfg: HypergraphConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Prune and optionally group SAE features into supernodes.

    Steps:
      1) Drop edges with weight < min_edge_weight.
      2) Enforce per-token incident cap for each output_token/input_token id; keep top by weight per token id.
      3) Optional supernode grouping ("supernode"):
         - Group sae_feature nodes by label prefix before cfg.supernode_label_delim.
         - Only form when group size >= supernode_min_group.
         - Create circuit_supernode 'super_{prefix_slug}', remap incidences referencing members to the supernode id.
         - Merge duplicates by taking max weight; tie-break by lexicographic id.
      4) Enforce global caps:
         - Sort incidences by weight desc, id asc; take top cfg.max_incidences.
         - Drop unreferenced nodes.
         - If nodes > cfg.max_nodes, truncate preferring tokens, then supernodes, then highest-activation features.
           Remove incidences that reference any dropped node.

    Returns: (pruned_nodes, pruned_incidences, grouping_metadata)
    grouping_metadata example: { "supernodes": { "super_biology": ["feat_12", "feat_19", ...] } }
    """
    # Normalize minimal node typing map for pruning logic
    nodes_norm = _normalize_nodes(nodes)
    id2type: Dict[str, str] = {str(n["id"]): str(n.get("type") or "") for n in nodes_norm}

    # 1) Threshold edges
    kept_edges_1: List[Dict[str, Any]] = []
    for e in incidences:
        try:
            w = float(e.get("weight", 0.0))
        except Exception:
            w = 0.0
        if w < float(cfg.min_edge_weight):
            continue
        kept_edges_1.append(e)

    # 2) Per-token cap
    # Determine token node ids
    token_ids: Set[str] = {nid for nid, t in id2type.items() if t in ("input_token", "output_token")}

    def _edge_token_id(e: Dict[str, Any]) -> Optional[str]:
        ns = e.get("node_ids") or []
        if not isinstance(ns, list):
            return None
        # Prefer output_token if present
        out_tok = None
        any_tok = None
        for nid in ns:
            sid = str(nid)
            t = id2type.get(sid)
            if t == "output_token":
                out_tok = sid
            if t in ("input_token", "output_token"):
                any_tok = sid
        return out_tok or any_tok

    edges_by_token: Dict[str, List[Dict[str, Any]]] = {}
    other_edges: List[Dict[str, Any]] = []

    for e in kept_edges_1:
        tok = _edge_token_id(e)
        if tok and tok in token_ids:
            edges_by_token.setdefault(tok, []).append(e)
        else:
            other_edges.append(e)

    kept_edges_2: List[Dict[str, Any]] = []
    for tok, lst in edges_by_token.items():
        # Sort by weight desc, id asc for determinism
        lst_sorted = sorted(
            lst,
            key=lambda x: (-_clamp01(x.get("weight", 0.0)), str(x.get("id") or "")),
        )
        cap = max(1, int(cfg.per_token_incident_cap))
        kept_edges_2.extend(lst_sorted[:cap])
    kept_edges_2.extend(other_edges)

    # 3) Optional supernode grouping
    grouping_metadata: Dict[str, Any] = {"supernodes": {}}
    if str(getattr(cfg, "grouping", "supernode")).lower() == "supernode":
        # Build prefix -> member_ids for sae_feature nodes
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for n in nodes_norm:
            if n.get("type") != "sae_feature":
                continue
            label = str(n.get("label") or "")
            if cfg.supernode_label_delim in label:
                prefix = label.split(cfg.supernode_label_delim, 1)[0].strip()
            else:
                # No delimiter - skip grouping
                continue
            if not prefix:
                continue
            groups.setdefault(prefix, []).append(n)

        # Realize supernodes for sufficiently large groups
        member_to_super: Dict[str, str] = {}
        super_nodes: List[Dict[str, Any]] = []
        for prefix, members in groups.items():
            if len(members) < int(cfg.supernode_min_group):
                continue
            slug = _slugify_prefix(prefix)
            super_id = f"super_{slug}"
            # Deduplicate if existing
            if any(sn["id"] == super_id for sn in super_nodes):
                # Ensure uniqueness by suffixing an index
                idx = 2
                base = super_id
                while any(sn["id"] == f"{base}_{idx}" for sn in super_nodes):
                    idx += 1
                super_id = f"{base}_{idx}"
            member_ids = [m["id"] for m in members]
            for mid in member_ids:
                member_to_super[str(mid)] = super_id
            # Track metadata
            grouping_metadata["supernodes"][super_id] = list(member_ids)
            # Build the supernode
            sn = {
                "id": super_id,
                "type": "circuit_supernode",
                "label": f"{prefix} (cluster)",
                "attributes": {
                    "member_count": len(member_ids),
                    "members": member_ids[:200],  # truncated display list
                },
            }
            super_nodes.append(sn)

        if member_to_super:
            # Remove member feature nodes; add supernodes
            filtered_nodes: List[Dict[str, Any]] = []
            for n in nodes_norm:
                nid = str(n["id"])
                if n.get("type") == "sae_feature" and nid in member_to_super:
                    continue
                filtered_nodes.append(n)
            filtered_nodes.extend(super_nodes)
            nodes_norm = filtered_nodes
            id2type = {str(n["id"]): str(n.get("type") or "") for n in nodes_norm}
            token_ids = {nid for nid, t in id2type.items() if t in ("input_token", "output_token")}

            # Remap incidences
            remapped: List[Dict[str, Any]] = []
            for e in kept_edges_2:
                ns = e.get("node_ids") or []
                if not isinstance(ns, list):
                    continue
                new_nodes: List[str] = []
                for nid in ns:
                    sid = str(nid)
                    new_nodes.append(member_to_super.get(sid, sid))
                # Keep original id/weight/meta; normalize later
                e2 = dict(e)
                e2["node_ids"] = new_nodes
                remapped.append(e2)

            # Merge duplicates by key = tuple(token-last order of node_ids)
            merged: Dict[Tuple[str, ...], Dict[str, Any]] = {}
            for e in remapped:
                key_nodes = tuple(_token_last_order([str(x) for x in e.get("node_ids") or []], id2type))
                if len(key_nodes) < 2:
                    continue
                eid = str(e.get("id") or "")
                w = _clamp01(e.get("weight", 0.0))
                prev = merged.get(key_nodes)
                if prev is None:
                    merged[key_nodes] = dict(e)
                else:
                    # Choose max weight; tie-break by lexicographically smaller id
                    prev_w = _clamp01(prev.get("weight", 0.0))
                    if w > prev_w:
                        merged[key_nodes] = dict(e)
                    elif w == prev_w:
                        prev_id = str(prev.get("id") or "")
                        if eid and (not prev_id or eid < prev_id):
                            merged[key_nodes] = dict(e)
            kept_edges_2 = list(merged.values())

    # 4) Global caps on incidences then nodes
    # Normalize id2type again (may have changed)
    id2type = {str(n["id"]): str(n.get("type") or "") for n in nodes_norm}

    # Normalize and sort incidences; take top max_incidences
    incid_norm = _normalize_incidences(kept_edges_2, id2type)
    incid_sorted = sorted(
        incid_norm,
        key=lambda e: (-float(e.get("weight", 0.0)), str(e.get("id") or "")),
    )
    incid_capped = incid_sorted[: max(0, int(cfg.max_incidences))]

    # Drop unreferenced nodes
    ref_ids: Set[str] = set()
    for e in incid_capped:
        for nid in (e.get("node_ids") or []):
            ref_ids.add(str(nid))
    nodes_ref = [n for n in nodes_norm if str(n["id"]) in ref_ids]

    # Node cap: prefer tokens, then supernodes, then high-activation features
    max_nodes = max(0, int(cfg.max_nodes))
    if len(nodes_ref) > max_nodes:
        nodes_sorted = sorted(nodes_ref, key=_node_priority)
        nodes_ref = nodes_sorted[:max_nodes]
        kept_ids = {str(n["id"]) for n in nodes_ref}
        # Remove incidences referencing dropped nodes
        incid_capped = [
            e for e in incid_capped
            if all(str(nid) in kept_ids for nid in (e.get("node_ids") or []))
        ]

    # Final deterministic node order
    nodes_final = sorted(nodes_ref, key=lambda n: (_node_priority(n)[0], str(n.get("id") or "")))
    try:
        if _otel is not None:
            supernode_count = sum(1 for n in nodes_final if n.get("type") == "circuit_supernode")
            _otel.hif_counts(len(nodes_final), len(incid_capped))
            _otel.hif_supernode_count(supernode_count)
            try:
                ratio = 0.0
                try:
                    ratio = 1.0 - (float(len(incid_capped)) / float(len(incidences))) if len(incidences) else 0.0
                except Exception:
                    ratio = 0.0
                _otel.hif_prune_ratio(ratio)
            except Exception:
                pass
    except Exception:
        pass
    return nodes_final, incid_capped, grouping_metadata


def build_hif(
    nodes: List[Dict[str, Any]],
    incidences: List[Dict[str, Any]],
    meta: Dict[str, Any],
    cfg: HypergraphConfig,
) -> Dict[str, Any]:
    """
    Assemble a HIF v1 JSON dict aligned with legacy schema in [`libs/hif/schema.json`](libs/hif/schema.json:185).

    - network-type: "directed"
    - nodes: normalized HIFLegacyNode[]
    - incidences: normalized HIFLegacyIncidence[]
    - meta: sanitized; meta.limits is populated from cfg if missing (only allowed keys)
    """
    # Sanitize meta (HIF v1 meta additionalProperties=false)
    allowed_meta_keys = {"model_name", "model_hash", "sae_dictionary", "granularity", "created_at", "limits", "version"}
    meta_in = meta or {}
    meta_out: Dict[str, Any] = {}

    for k in allowed_meta_keys:
        if k in meta_in:
            meta_out[k] = meta_in[k]

    if "created_at" not in meta_out:
        meta_out["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta_out["version"] = "hif-1"

    # limits
    limits = meta_out.get("limits") or {}
    if not isinstance(limits, dict):
        limits = {}
    limits = {
        "min_edge_weight": float(getattr(cfg, "min_edge_weight", 0.01)),
        "max_nodes": int(getattr(cfg, "max_nodes", 5000)),
        "max_incidences": int(getattr(cfg, "max_incidences", 20000)),
        **{k: limits[k] for k in ("min_edge_weight", "max_nodes", "max_incidences") if k in limits},
    }
    # Reorder keys for consistency
    meta_out["limits"] = {
        "min_edge_weight": float(limits["min_edge_weight"]),
        "max_nodes": int(limits["max_nodes"]),
        "max_incidences": int(limits["max_incidences"]),
    }

    # Normalize nodes and incidences to schema
    nodes_norm = _normalize_nodes(nodes)
    id2type: Dict[str, str] = {str(n["id"]): str(n.get("type") or "") for n in nodes_norm}
    incid_norm = _normalize_incidences(incidences, id2type)

    return {
        "network-type": "directed",
        "nodes": nodes_norm,
        "incidences": incid_norm,
        "meta": meta_out,
    }


def validate_hif(obj: Dict[str, Any]) -> None:
    """
    Optional validation via [`libs/hif/validator.py`](libs/hif/validator.py:1).
    Raises ValueError if validation fails.
    """
    try:
        # Late import to keep stdlib-only dependency for normal operation
        from libs.hif.validator import validate_hif as _validate_any  # type: ignore
    except Exception as e:
        # Validator unavailable; treat as no-op
        return
    try:
        _validate_any(obj)  # back-compat validator accepts v1 legacy or v2 payloads
    except Exception as e:  # jsonschema.ValidationError or other
        raise ValueError(f"HIF validation failed: {e}") from e