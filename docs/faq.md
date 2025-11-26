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
