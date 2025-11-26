from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import os
import json

try:
    import httpx
except Exception as e:  # pragma: no cover
    raise RuntimeError("The Python SDK requires httpx. Install with: pip install httpx") from e

# Requests is optional; import lazily in HypergraphClient to avoid hard dependency at import time.
requests = None  # type: ignore


class ChatResponse:
    """
    Wrapper for the chat response that exposes a lazy explanation fetcher.
    """
    def __init__(
        self,
        client: "Client",
        raw: Dict[str, Any],
    ) -> None:
        self._client = client
        self.raw = raw
        self.explanation_metadata: Optional[Dict[str, Any]] = raw.get("explanation_metadata")

    def get_explanation(
        self,
        *,
        poll_interval: float = 0.25,
        max_wait_seconds: float = 30.0,
        raise_on_error: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Polls /v1/traces/{trace_id}/graph until the HIF graph is available or timeout.
        Returns the HIF JSON on success, or None on timeout (unless raise_on_error=True).
        """
        if not self.explanation_metadata:
            return None

        trace_id = self.explanation_metadata.get("trace_id")
        if not trace_id:
            return None

        deadline = time.time() + max_wait_seconds
        last_status: Optional[Dict[str, Any]] = None

        while time.time() < deadline:
            # Check status first (optional)
            st_url = f"{self._client.base_url}/v1/traces/{trace_id}/status"
            try:
                st_resp = self._client._http.get(st_url, headers=self._client._headers(), timeout=self._client.timeout)
                if st_resp.status_code == 200:
                    last_status = st_resp.json()
                    state = last_status.get("state")
                    if state in ("failed", "canceled", "expired"):
                        if raise_on_error:
                            raise RuntimeError(f"Trace {trace_id} ended with state={state}")
                        return None
            except Exception:
                # Ignore transient status failures; proceed to graph fetch
                pass

            # Try the graph
            url = f"{self._client.base_url}/v1/traces/{trace_id}/graph"
            r = self._client._http.get(url, headers=self._client._headers(), timeout=self._client.timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (404, 409):
                # Not ready / conflict, continue polling
                time.sleep(poll_interval)
                continue
            if r.status_code == 410:
                if raise_on_error:
                    raise RuntimeError(f"Trace {trace_id} expired")
                return None

            # Unknown condition; wait a bit and retry
            time.sleep(poll_interval)

        if raise_on_error:
            raise TimeoutError(f"Explanation for trace {trace_id} not ready after {max_wait_seconds}s")
        return None


class ChatCompletions:
    """
    Client for /v1/chat/completions (OpenAI-compatible) with explainability headers.
    """
    def __init__(self, client: "Client") -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        explain: bool = True,
        explain_granularity: str = "sentence",  # or "token"
        explain_features: str = "sae-gpt4-2m",
        explain_budget_ms: Optional[int] = None,
        trace_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> ChatResponse:
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        headers = self._client._headers()
        if explain:
            headers.update({
                "x-explain-mode": "hypergraph",
                "x-explain-granularity": explain_granularity,
                "x-explain-features": explain_features,
            })
            if explain_budget_ms is not None:
                headers["x-explain-budget"] = str(explain_budget_ms)
        if trace_id:
            headers["x-trace-id"] = trace_id
        if idempotency_key:
            headers["x-idempotency-key"] = idempotency_key

        url = f"{self._client.base_url}/v1/chat/completions"
        resp = self._client._http.post(url, headers=headers, json=body, timeout=self._client.timeout)
        resp.raise_for_status()
        return ChatResponse(self._client, resp.json())


class Client:
    """
    Minimal SDK Client

    Usage:
      from your_company_explainability.client import Client
      c = Client(base_url="http://localhost:8080", api_key="sk-...")
      res = c.chat.completions.create(
          model="gpt-4-turbo",
          messages=[{"role": "user", "content": "Explain quantum entanglement."}],
          explain=True,
          explain_granularity="sentence",
      )
      hif = res.get_explanation()
    """
    def __init__(self, *, base_url: str, api_key: Optional[str] = None, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._http = httpx.Client()
        self.chat = type("ChatNamespace", (), {})()
        self.chat.completions = ChatCompletions(self)

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# =========================
# Version 2 SDK extensions
# =========================

class Explanation:
    """
    Explanation model object (version2) that wraps the full explanation envelope
    returned by the gateway and exposes a simple visualization adapter.

    Attributes:
        envelope: The full explanation JSON object as returned by the API.
        hypergraph: The hypergraph sub-document (envelope["hypergraph"]).
    """

    def __init__(self, envelope: Dict[str, Any]) -> None:
        self.envelope: Dict[str, Any] = envelope
        self.hypergraph: Dict[str, Any] = envelope.get("hypergraph", {}) or {}

    def visualize(self, format: str = "force-directed") -> Dict[str, Any]:
        """
        Return a simple force-directed layout friendly structure.

        The returned dict has the form:
            {
              "nodes": [
                 {"id": ..., "type": ..., "label": ...},
                 ...
              ],
              "links": [
                 {"source": ..., "target": ..., "weight": ..., "type": ...},
                 ...
              ]
            }

        This method does not write files or open windows; it just returns an
        in-memory structure that callers can render with their preferred tooling.
        """
        hg = self.hypergraph or {}
        raw_nodes = hg.get("nodes", []) or []
        raw_edges = hg.get("edges") if isinstance(hg, dict) else None
        if not raw_edges:
            raw_edges = hg.get("links", []) or []

        nodes: List[Dict[str, Any]] = []
        for n in raw_nodes:
            nodes.append(
                {
                    "id": n.get("id"),
                    "type": n.get("type"),
                    "label": n.get("text", "") or n.get("label", ""),
                }
            )

        links: List[Dict[str, Any]] = []
        for e in raw_edges:
            src = e.get("source") or e.get("src") or e.get("from")
            tgt = e.get("target") or e.get("dst") or e.get("to")
            if src is None or tgt is None:
                continue
            links.append(
                {
                    "source": src,
                    "target": tgt,
                    "weight": e.get("weight", 0.0),
                    "type": e.get("interaction_type", "SYNERGY"),
                }
            )

        return {"nodes": nodes, "links": links}


class Response:
    """
    Response wrapper for chat completions (version2).

    Wraps the upstream completion JSON and provides lazy explanation retrieval.
    """

    def __init__(self, client: "HypergraphClient", raw: Dict[str, Any], id: str) -> None:
        self.client = client
        self.raw = raw
        self.id = id

    def explain(self, format: str = "hif", timeout_s: int = 30, poll_interval_s: float = 1.0) -> "Explanation":
        """
        Poll GET /v1/chat/completions/{id}/explanation with the given format until completion.

        Behavior:
          - Repeatedly invokes:
                GET {base_url}/v1/chat/completions/{id}/explanation?format={format}
          - If the server replies 202 (pending), the call sleeps for poll_interval_s and retries.
          - If 200, returns an Explanation object.
          - If 404, raises FileNotFoundError with the id.
          - If any other non-2xx status, raises RuntimeError including status and body.
          - If timeout elapses, raises TimeoutError containing id and last known status.

        Args:
            format: Explanation format to request (e.g., "hif").
            timeout_s: Maximum total time to wait before giving up.
            poll_interval_s: Delay between polls when the explanation is pending.

        Returns:
            Explanation: The explanation envelope object.

        Raises:
            TimeoutError: If the explanation remains pending until timeout.
            FileNotFoundError: If the explanation resource is not found (404).
            RuntimeError: For other non-2xx responses.
        """
        deadline = time.time() + float(timeout_s)
        last_status: Optional[str] = None

        url = f"{self.client.base_url}/v1/chat/completions/{self.id}/explanation"
        if format:
            url = f"{url}?format={format}"

        while True:
            status, data, text, _ = self.client._get_json(url)
            if status == 200:
                if not isinstance(data, dict):
                    raise RuntimeError(f"Explanation GET returned 200 but non-JSON body for id={self.id}")
                # Per contract, 200 indicates completion. Return the explanation.
                return Explanation(data)

            if status == 202:
                # Pending; track last reported status if present
                if isinstance(data, dict):
                    last_status = data.get("status") or last_status or "pending"
                if time.time() >= deadline:
                    raise TimeoutError(f"Timeout waiting for explanation for id={self.id}; last_status={last_status}")
                time.sleep(poll_interval_s)
                continue

            if status == 404:
                raise FileNotFoundError(f"Explanation not found for completion id {self.id}")

            # Other non-2xx
            raise RuntimeError(f"GET explanation failed for id={self.id} with status={status}, body={text}")


def _strip_sse_data_prefix(s: str) -> str:
    if s.startswith("data:"):
        return s[5:].lstrip()
    return s


class _HypergraphChatCompletions:
    """
    Accessor for POST /v1/chat/completions for HypergraphClient (version2).

    Supports both stream=False and stream=True. In streaming mode the SDK consumes
    the stream to completion locally without re-emitting events. If aggregation is
    not feasible, it returns a Response with the raw streamed chunks.
    """

    def __init__(self, client: "HypergraphClient") -> None:
        self._client = client

    def create(self, **payload: Any) -> Response:
        """
        Send a chat completion request to the gateway.

        Args:
            **payload: JSON payload forwarded to POST /v1/chat/completions.
                       If payload contains "stream": True, a streaming request is used.

        Returns:
            Response: Wrapper with .id and .explain() available.

        Raises:
            RuntimeError: If the upstream returns a non-2xx status or if no id can be determined.
        """
        url = f"{self._client.base_url}/v1/chat/completions"
        headers = self._client._headers()

        stream = bool(payload.get("stream", False))

        # Non-streaming path: simple POST and JSON decode.
        if not stream:
            if self._client._use_httpx:
                r = self._client._session.post(url, headers=headers, json=payload, timeout=self._client.request_timeout_s)
                status = r.status_code
                text = r.text
                try:
                    raw = r.json()
                except Exception:
                    raw = None
            else:
                r = self._client._session.post(url, headers=headers, json=payload, timeout=self._client.request_timeout_s)
                status = r.status_code
                text = r.text
                try:
                    raw = r.json()
                except Exception:
                    raw = None

            if status < 200 or status >= 300:
                raise RuntimeError(f"POST /v1/chat/completions failed with status={status}, body={text}")

            if not isinstance(raw, dict):
                raise RuntimeError(f"POST /v1/chat/completions returned non-JSON body: {text}")

            id_value = raw.get("id")
            if not id_value or not isinstance(id_value, str):
                raise RuntimeError("POST /v1/chat/completions response must include an 'id' per API contract")

            return Response(self._client, raw=raw, id=id_value)

        # Streaming path: consume to completion locally.
        raw_chunks: List[str] = []
        id_value: Optional[str] = None
        final_json: Optional[Dict[str, Any]] = None

        if self._client._use_httpx:
            # httpx streaming
            with self._client._session.stream(
                "POST", url, headers=headers, json=payload, timeout=self._client.request_timeout_s
            ) as resp:
                status = resp.status_code
                if status < 200 or status >= 300:
                    # Read the body to include in error message
                    text = resp.text
                    raise RuntimeError(f"POST /v1/chat/completions (stream) failed with status={status}, body={text}")
                for line in resp.iter_lines():
                    if line is None:
                        continue
                    s = _strip_sse_data_prefix(line.strip())
                    if not s:
                        continue
                    raw_chunks.append(s)
                    # Best-effort to capture an id from any JSON chunk
                    try:
                        obj = json.loads(s)
                        if isinstance(obj, dict):
                            if final_json is None and "choices" in obj and "id" in obj:
                                # Some servers emit a final aggregated object as the last event.
                                final_json = obj
                            if "id" in obj and isinstance(obj["id"], str):
                                id_value = obj["id"]
                    except Exception:
                        # Ignore non-JSON stream lines
                        pass

                # Try to parse the last non-empty line as final JSON if none set
                if final_json is None and raw_chunks:
                    try:
                        candidate = json.loads(raw_chunks[-1])
                        if isinstance(candidate, dict):
                            final_json = candidate
                    except Exception:
                        pass
        else:
            # requests streaming
            r = self._client._session.post(
                url, headers=headers, json=payload, timeout=self._client.request_timeout_s, stream=True
            )
            status = r.status_code
            if status < 200 or status >= 300:
                text = r.text
                r.close()
                raise RuntimeError(f"POST /v1/chat/completions (stream) failed with status={status}, body={text}")

            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                s = _strip_sse_data_prefix(line.strip())
                if not s:
                    continue
                raw_chunks.append(s)
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict):
                        if final_json is None and "choices" in obj and "id" in obj:
                            final_json = obj
                        if "id" in obj and isinstance(obj["id"], str):
                            id_value = obj["id"]
                except Exception:
                    pass
            r.close()

        raw: Dict[str, Any]
        if isinstance(final_json, dict):
            raw = final_json
            if not id_value and isinstance(final_json.get("id"), str):
                id_value = final_json["id"]
        else:
            raw = {"stream": True, "chunks": raw_chunks}

        if not id_value:
            # Could not locate an id in streamed data. Per contract, we must have one.
            raise RuntimeError("Streaming response must include an 'id' in headers or body per API contract")

        return Response(self._client, raw=raw, id=id_value)


class HypergraphClient:
    """
    Version2 client that supports lazy explanation retrieval and basic visualization.

    - Uses httpx if available, else falls back to requests (soft import).
    - Passes Authorization and X-Provider headers if configured.
    - Exposes chat.completions.create(...) which returns a Response wrapper.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout_s: float = 30.0,
    ) -> None:
        """
        Initialize the HypergraphClient.

        Args:
            provider: Optional upstream provider identifier. Sent via 'X-Provider' header.
            api_key: Optional API key. Sent via 'Authorization: Bearer ...' header.
            base_url: Base URL for the gateway. Defaults to env 'EXPLAINABILITY_BASE_URL' or http://localhost:8000.
            request_timeout_s: Default timeout (seconds) for POST/GET calls (can be overridden per request internally).
        """
        self.provider = provider
        self.api_key = api_key
        self.base_url = (base_url or os.environ.get("EXPLAINABILITY_BASE_URL", "http://localhost:8000")).rstrip("/")
        self.request_timeout_s = float(request_timeout_s)

        # Soft imports: prefer httpx but fall back to requests if available
        self._use_httpx = httpx is not None  # type: ignore[name-defined]
        # Lazy import requests to avoid top-level dependency
        try:
            import requests as _requests  # type: ignore
        except Exception:
            _requests = None
        self._use_requests = _requests is not None

        if not self._use_httpx and not self._use_requests:
            raise RuntimeError("This SDK requires either httpx or requests. Install one of them.")

        # Maintain a single session where possible.
        if self._use_httpx:
            self._session = httpx.Client()  # timeouts passed per call
        else:
            self._session = _requests.Session()  # type: ignore[union-attr]

        # Namespaced accessors
        self.chat = type("ChatNamespace", (), {})()
        self.chat.completions = _HypergraphChatCompletions(self)

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if self.provider:
            h["X-Provider"] = str(self.provider)
        return h

    def _get_json(self, url: str) -> Tuple[int, Optional[Dict[str, Any]], str, Dict[str, Any]]:
        """
        Internal helper to GET JSON from the specified URL using the configured transport.
        Returns (status, json_or_none, text, headers).
        """
        if self._use_httpx:
            r = self._session.get(url, headers=self._headers(), timeout=self.request_timeout_s)
            status = r.status_code
            text = r.text
            headers = dict(r.headers)
            try:
                data = r.json()
            except Exception:
                data = None
            return status, data, text, headers
        else:
            r = self._session.get(url, headers=self._headers(), timeout=self.request_timeout_s)
            status = r.status_code
            text = r.text
            headers = dict(r.headers)
            try:
                data = r.json()
            except Exception:
                data = None
            return status, data, text, headers

    def close(self) -> None:
        try:
            if self._use_httpx and self._session is not None:
                self._session.close()
            elif self._use_requests and self._session is not None:
                self._session.close()
        except Exception:
            pass

    def __enter__(self) -> "HypergraphClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()