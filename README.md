# Sidecar: Hypergraph Explanation in LLM Interpretability

Sidecar is an architectural pattern for large language model (LLM) interpretability that separates the act of answering a user’s query from the act of explaining the model’s response. The central idea is methodological: treat explanation as a distinct process that can be reasoned about, replicated, and compared, without perturbing the original inference. In Sidecar, a Gateway returns the inference result in the usual OpenAI‑compatible format, and an Explainer concurrently constructs a hypergraph explanation (HIF v1) keyed by a stable trace identifier. This separation clarifies the epistemic status of explanations and facilitates principled experiments.

## Motivation and contribution

Interpretability methods often require auxiliary computation (e.g., attribution, path patching, feature discovery) whose cost and stochasticity can interfere with causal claims about the original model run. Sidecar formalizes a workflow in which:
- inference remains a normal decoding event; and
- explanation is a follow‑up process with its own lifecycle, observables, and artifacts.

The contribution is a reference design that (i) preserves the semantics of mainstream inference APIs; (ii) introduces an explicit handle (trace_id) that ties the inference to a later explanation; and (iii) represents explanations as a structured hypergraph (HIF v1), suitable for analysis and validation.

## Gateway–Explainer collaboration via trace_id

A researcher opts into explanation by adding a single header to an otherwise standard OpenAI‑style POST /v1/chat/completions request. The Gateway returns the model’s answer and an explanation handle, explanation_metadata.trace_id. That handle propagates through the Explainer’s pipeline and is used to poll or stream the explanation’s status; once complete, the explanation is retrieved as a HIF v1 document.

The relationships are explicit in code:
- OpenAI‑compatible handler and opt‑in explainability: [create_chat_completion()](services/gateway/src/app.py:540)
- Explanation lifecycle endpoints: [get_trace_status()](services/gateway/src/app.py:886), [get_trace_graph()](services/gateway/src/app.py:922)
- Event stream for updates: [stream_trace()](services/gateway/src/app.py:965)
- Webhook registration and cancellation: [register_webhook()](services/gateway/src/app.py:1021), [cancel_trace()](services/gateway/src/app.py:1062)

### Minimal example (request → status/stream → explanation)

1) Send a standard chat request and opt into explanation:

```bash
curl -sS -X POST "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "x-explain-mode: hypergraph" \
  -H "x-explain-granularity: sentence" \
  -H "x-explain-features: sae-gpt4-2m" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "system", "content": "You are a careful assistant."},
      {"role": "user", "content": "Explain why a model might refuse a harmful instruction."}
    ],
    "stream": false
  }'
# If access control is enabled (AUTH_MODE=static), include:
#   -H "Authorization: Bearer $TOKEN"
```

The response includes an explanation handle, e.g., `"trace_id": "trc_5e1a9d0c3baf"`.

2) Observe the explanation’s progress:

```bash
# Poll status
curl -sS "http://localhost:8080/v1/traces/trc_5e1a9d0c3baf/status"

# Stream events (Server‑Sent Events: status_update, complete)
curl -N "http://localhost:8080/v1/traces/trc_5e1a9d0c3baf/stream"
```

3) Retrieve the explanation when ready:

```bash
# HIF v1 hypergraph (200 when ready; 404 if not yet available; 410 if expired)
curl -sS --compressed "http://localhost:8080/v1/traces/trc_5e1a9d0c3baf/graph"
```

The returned object is a HIF v1 hypergraph describing interactions between inputs, intermediate features, and outputs.

## Representation: the HIF v1 hypergraph

Explanations are emitted as HIF v1—an interchange format in which nodes represent concepts (e.g., SAE features, tokens) and incidences encode multi‑way interactions with associated weights. The formal shape is defined in the schema and validated by a lightweight tool:
- Overview: [docs/hif-schema.md](docs/hif-schema.md:1)
- Canonical schema: [libs/hif/schema.json](libs/hif/schema.json:1)
- Validation entry point: [validate_hif()](libs/hif/validator.py:117)

This data model is intentionally research‑friendly: it is easy to serialize, adjudicate, and aggregate across runs, facilitating ablation studies, agreement metrics, and cross‑model comparisons.

## Methods and instrumentation

Sidecar includes instrumentation intended to support careful experimental practice:

- Access control and auditing (optional). When enabled, static tokens carry tenant_id and scopes. Failures are explicit (401/403), and audit lines avoid high‑cardinality fields; any short content snippets used for context are scrubbed deterministically (see [rbac.py](services/gateway/src/rbac.py:1), [sanitize_message](libs/sanitize/pii.py:72)).

- Experimental load control. Per‑tenant token buckets express upper bounds as a matter of experimental hygiene; overruns return 429 with Retry‑After (see [rate_limit.py](services/gateway/src/rate_limit.py:180)).

- Observables for replication. The Gateway can expose metrics at /metrics; the Explainer can publish a minimal Prometheus endpoint (port 9090). Tracing export is opt‑in and controlled by environment variables (see [otel.py (gateway)](services/gateway/src/otel.py:330), [otel.py (explainer)](services/explainer/src/otel.py:249)).

- Explanation state and artifact persistence. Status updates are stored with explicit TTL semantics and tenant immutability rules; HIF artifacts can be persisted to object storage or a local path with cautious key construction (see [status_store.py](services/explainer/src/status_store.py:1), [s3_store.py](services/explainer/src/s3_store.py:1)).

- Load‑aware explanation quality. Under heavy load, the Explainer may degrade gracefully (e.g., token → sentence granularity), recording the decision as part of the explanation’s status; this makes trade‑offs visible and testable (see [backpressure.py](services/explainer/src/backpressure.py:1)).

These features are not positioned as “enterprise controls,” but as methodological aids for running careful, repeatable studies in shared environments.

## Relation to interpretability research

Sidecar’s separation of inference and explanation supports a broad set of studies:

- Attribution and agreement. Compare attention‑based and gradient‑based methods on common corpora; compute stability across baselines and seeds using a shared HIF representation.

- Causal analyses. Patch activations along hypothesized paths, intervene on sparse feature activations, and measure downstream effects while preserving the original inference as a reference.

- Feature‑level inquiry. Project activations onto dictionary atoms (e.g., SAE‑derived), label features qualitatively, and test causal roles through targeted ablations.

- Cross‑model and cross‑task comparisons. Repeat explanation protocols over small GPT/BERT‑family checkpoints to assess consistency and transfer properties.

Because explanations are tied to a stable trace_id, it is straightforward to align inference outcomes, explanation artifacts, and analysis notebooks in a single experimental log.

## Proceeding further

- API details and examples: [docs/api-reference.md](docs/api-reference.md:1), OpenAPI at [api/openapi/hypergraph-api.yaml](api/openapi/hypergraph-api.yaml:1)
- HIF format and validation: [docs/hif-schema.md](docs/hif-schema.md:1), [libs/hif/schema.json](libs/hif/schema.json:1), [validate_hif()](libs/hif/validator.py:117)
- Tutorials:
  - Python quickstart: [docs/tutorials/python-quickstart.md](docs/tutorials/python-quickstart.md:1)
  - TypeScript/React quickstart: [docs/tutorials/typescript-quickstart.md](docs/tutorials/typescript-quickstart.md:1)
  - End‑to‑end walkthrough: [docs/tutorials/e2e.md](docs/tutorials/e2e.md:1)

### Notes on access control

If access control is enabled (AUTH_MODE=static), include `-H "Authorization: Bearer $TOKEN"` in HTTP examples where appropriate; see [rbac.py](services/gateway/src/rbac.py:1) for the static token schema and enforcement details.

By treating explanation as its own object of study—traceable, measurable, and reproducible—Sidecar provides a practical scaffold for mechanistic interpretability, attribution research, and causal analysis of LLM behavior.

Implementation notes for the editor:
- Replace the existing README.md content entirely with the above narrative.
- Retain all clickable anchors exactly as written (e.g., [create_chat_completion()](services/gateway/src/app.py:540)).
- Keep localhost examples and the Authorization note as shown (conditional on AUTH_MODE=static).
- Do not modify any other files in this subtask.