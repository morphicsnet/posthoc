# FAQ

Audience: External

## How is HIF different from classic attribution?
HIF represents explanations as hypergraphs, capturing higher-order relationships beyond pairwise attribution. It is model-agnostic and optimized for auditing and downstream analysis.

## What data is stored by default and how can I opt out?
By default, trace metadata and resulting HIF artifacts may be persisted (see EXPLAINER_TABLE). Configure retention policies or disable persistence by omitting DATABASE_URL. Avoid sending sensitive data or tokenize/redact upstream.

## Which model providers are supported?
Gateway is provider-neutral via your configured LLM proxy (LLM_PROXY_URL). Any provider the proxy supports can be used without changing the explainability surface.

## How reliable is verification?
Shadow-model verification is best-effort and configurable. Results depend on the chosen VERIFY_MODEL and parameters; treat them as complementary signals, not ground truth.

## What are attach-rate costs?
Attach-rate is the fraction of chats with `x-explain-mode: hypergraph`. Capacity (and cost) scales with attach-rate, granularity, and token-mix. Use [attach rate analyzer](tools/analysis/attach_rate_analyzer.py:1) and KEDA targets in [values.yaml](manifests/helm/hypergraph/values.yaml:1) to plan.

## Token vs sentence granularity tradeoffs
- sentence: lower cost, faster SLA; coarser attribution windows
- token: higher cost, slower SLA; fine-grained token-level links
Backpressure may downgrade token→sentence under load (see [backpressure.py](services/explainer/src/backpressure.py:69)).

## How do I disable explanation globally for a tenant?
- Omit `traces:write` scope in the token for that tenant to prevent queuing new traces via the Gateway.
- Optionally apply an interceptor policy to drop `x-explain-mode` headers at the edge for that tenant.
