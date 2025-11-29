# HIF Schema Reference (v1)

This document describes the HIF v1 legacy structure used by the Async Sidecar. It aligns with the legacy section of the canonical schema in [libs/hif/schema.json](libs/hif/schema.json:1) and is validated by [validator.py](libs/hif/validator.py:1) via [validate_hif()](libs/hif/validator.py:117).

Versioning summary:
- v1 objects MUST declare meta.version = "hif-1".
- The JSON schema also contains newer v2 components for future expansion, but v1 remains supported and frozen for public API v1.

## Top-level shape (v1)

A v1 HIF graph is a single JSON object with the following fields:
- network-type: "directed"
- nodes: array of node objects
- incidences: array of hyperedge/incidence objects
- meta: object with provenance, limits, and version marker

Supported node types:
- sae_feature — internal concept feature (e.g., SAE unit)
- input_token — token from the input text
- output_token — token in the output text
- circuit_supernode — optional aggregation node used for visualization/grouping

## Example (minimal valid v1 graph)

```json
{
  "network-type": "directed",
  "nodes": [
    { "id": "feat_1024", "type": "sae_feature", "label": "Geography: Cities", "layer": 12, "activation_strength": 4.5 },
    { "id": "token_5", "type": "input_token", "label": "Paris", "position": 5 },
    { "id": "token_out_1", "type": "output_token", "label": "Paris is the capital of France.", "position": 1 }
  ],
  "incidences": [
    {
      "id": "e1",
      "node_ids": ["feat_1024", "token_out_1"],
      "weight": 0.85,
      "metadata": {
        "type": "causal_circuit",
        "method": "acdc",
        "window": "sent-1",
        "provenance": { "capture_layers": [6, 12, 24], "sae_version": "sae-gpt4-2m", "model_hash": "abcd1234" }
      }
    }
  ],
  "meta": {
    "model_name": "gpt-4-turbo",
    "model_hash": "abcd1234",
    "sae_dictionary": "sae-gpt4-2m",
    "granularity": "sentence",
    "created_at": "2025-01-01T00:00:00Z",
    "limits": { "min_edge_weight": 0.01, "max_nodes": 5000, "max_incidences": 20000 },
    "version": "hif-1"
  }
}
```

## Field semantics

Nodes:
- id: stable string identifier.
- type: one of sae_feature | input_token | output_token | circuit_supernode.
- label: human-readable text. For tokens, usually the token string. For features, recommended "Namespace: Subgroup" to enable grouping (e.g., "Geography: Cities").
- layer: integer layer index for features when applicable; null/omitted for tokens and supernodes.
- position: integer position for tokens; null/omitted for features and supernodes.
- activation_strength: numeric magnitude for features (e.g., average SAE activation for the window); optional for tokens and supernodes.
- attributes: free-form object for additional, vendor-specific data.

Incidences (hyperedges):
- id: unique string id.
- node_ids: array of node ids incident to the hyperedge. Order may be meaningful for certain methods; the viewer treats all-pairs if tokens/features are mixed.
- weight: numeric attribution score (e.g., Shapley, ACDC). Commonly in [0,1] but not strictly required by the schema.
- metadata: object with:
  - type: string label of incidence type (e.g., "causal_circuit").
  - method: attribution method (e.g., "acdc", "shapley", "saliency").
  - window: token/span window identifier (e.g., "sent-1" or "tok-3..8").
  - description: optional human-readable note.
  - provenance: object with capture_layers: int[], sae_version: string, model_hash: string.
- attributes: free-form object for additional, vendor-specific data.

Meta:
- model_name, model_hash: upstream model identifiers.
- sae_dictionary: identifier of the feature dictionary used.
- granularity: "sentence" or "token". Mirrors x-explain-granularity.
- created_at: RFC3339/ISO-8601 timestamp.
- limits: pruning and display defaults; see below.
- version: MUST be "hif-1" for v1 payloads.

## Limits and pruning defaults

The meta.limits object communicates pruning decisions from the explainer to consumers:
- min_edge_weight: edges below this value were pruned during materialization.
- max_nodes: maximum number of nodes emitted.
- max_incidences: maximum number of incidences emitted.

Viewers should treat these as hints. If absent, reasonable defaults are:
- min_edge_weight: 0.01
- max_nodes: 5000
- max_incidences: 20000

## Validation

Use the provided CLI to validate payloads against the schema.

Install dependency (if needed):
```bash
pip install 'jsonschema>=4.18'
```

Validate a file (JSON or gzipped JSON):
```bash
python libs/hif/validator.py ./my-graph.json
python libs/hif/validator.py ./my-graph.json.gz
```

Programmatic usage:
```python
from libs.hif.validator import validate_hypergraph, validate_explanation_response
validate_hypergraph({"network-type":"directed","nodes":[],"incidences":[],"meta":{"version":"hif-1"}})
validate_explanation_response({
  "id": "chatcmpl-123",
  "status": "completed",
  "hypergraph": {"nodes": [], "hyperedges": []}
})
```

The validator accepts v1 graphs, v2 Hypergraph objects, ExplanationResponse envelopes, and pending-state envelopes; see [validator.py](libs/hif/validator.py:1).

## Forward compatibility and versioning

- The v1 shape (hif-1) is frozen. Additive data should go into attributes fields at node/incidence level to preserve compatibility.
- Future versions (v2) introduce normalized node/edge shapes; the canonical schema in [schema.json](libs/hif/schema.json:1) supports both. API v1 continues to serve HIF v1 for GET /v1/traces/{trace_id}/graph.
- Client libraries should be liberal in what they accept and prefer documented fields; unknown fields MUST be ignored.

## Related SDKs and UI

- Python: [Client](sdks/python/your_company_explainability/your_company_explainability/client.py:143) can fetch and persist HIF for traces.
- TypeScript: [ExplainabilityClient](sdks/typescript/explainability-ui/src/client.ts:19) and [HypergraphViewer](sdks/typescript/explainability-ui/src/HypergraphViewer.tsx:233) can fetch and render graphs. The viewer supports grouping and min-edge-weight thresholds matching meta.limits.