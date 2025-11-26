# Python SDK — Your Company Explainability

Audience: External

## Install
Local editable install (and ensure httpx is available):
- pip install -e sdks/python/your_company_explainability
- pip install httpx

Project metadata: [pyproject.toml](sdks/python/your_company_explainability/pyproject.toml)

## Quickstart
```python
from your_company_explainability.client import Client

client = Client(base_url="http://localhost:8080", api_key=None)
res = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain quantum entanglement in 2 sentences."}],
    stream=False,
    explain=True,
    explain_granularity="sentence",
)
hif = res.get_explanation(max_wait_seconds=30)
print("HIF keys:", list(hif.keys()) if hif else None)
```

Runnable example: [sdks/python/your_company_explainability/examples/quickstart.py](sdks/python/your_company_explainability/examples/quickstart.py)

## Concepts
- Chat vs explanation lifecycle: request completes immediately, explanation may arrive asynchronously.
- Polling: use `get_explanation(max_wait_seconds=...)` or follow the trace endpoints.
- Timeouts: choose sensible client timeouts based on workload.

## Advanced
- Custom headers: x-explain-mode, x-explain-granularity, x-explain-features, x-explain-budget.
- Budgets: cap work via x-explain-budget (ms).
- Tracing: set x-trace-id for idempotency and correlation across systems.

## Links
- API docs: [docs/api-reference.md](docs/api-reference.md)
- E2E tutorial: [docs/tutorials/e2e.md](docs/tutorials/e2e.md)
- Client class: [Client](sdks/python/your_company_explainability/your_company_explainability/client.py:1)
