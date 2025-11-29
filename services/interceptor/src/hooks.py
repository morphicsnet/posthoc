"""
Hook abstraction and capture pipeline for per-token activation capture.

This file provides:
- CaptureConfig: configuration for the capture pipeline (top-k, windowing, packing).
- CapturePipeline: start_trace, capture_token, flush to build an envelope and write shard windows.
- register_layer_hook: lightweight adapter a model layer can call each token.

Notes:
- No heavy deps; NumPy is optional and only used if present via utils.topk_quantize.
- RedisClientStub is an in-memory stub with a pipelined interface (set/execute).
- Shard batches are written once per window; 'flush' writes any partial window and
  returns a minimal envelope containing shard keys and model metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    # Try package-style absolute import when repository root is on sys.path
    from services.interceptor.src.utils import topk_quantize, pack_shard, shard_key, now_ns, ns_to_ms  # type: ignore  # pylint: disable=import-error
except Exception:
    # Fallback to local module import when running as a script
    from utils import topk_quantize, pack_shard, shard_key, now_ns, ns_to_ms  # type: ignore  # pylint: disable=import-error


@dataclass
class CaptureConfig:
    model_name: str
    model_hash: str
    topk: int = 256
    window_size: int = 16
    compress: str = "json"  # 'json' | 'none'
    namespace: str = "activations"


class _PipelineStub:
    """
    Minimal pipeline stub to simulate Redis pipelined writes.
    """
    def __init__(self, store: Dict[str, bytes]) -> None:
        self._store = store
        self._cmds: List[Tuple[str, str, bytes]] = []

    def set(self, key: str, value: bytes) -> "_PipelineStub":
        self._cmds.append(("set", key, value))
        return self

    def execute(self) -> int:
        n = 0
        for op, k, v in self._cmds:
            if op == "set":
                self._store[k] = v
                n += 1
        self._cmds.clear()
        return n


class RedisClientStub:
    """
    Minimal Redis client stub that provides a pipeline() method and in-memory KV store.
    Replace with a production client that supports .pipeline().set(...).execute().
    """
    def __init__(self) -> None:
        self.store: Dict[str, bytes] = {}

    def pipeline(self) -> _PipelineStub:
        return _PipelineStub(self.store)


Envelope = Dict[str, Any]


class CapturePipeline:
    """
    Capture pipeline that:
    - top-k quantizes per-token activation summaries
    - batches tokens into fixed-size windows
    - packs each window and writes via pipeline to RedisClientStub
    - returns an envelope describing written shard keys and model metadata
    """

    def __init__(self, config: CaptureConfig, redis_client: Optional[RedisClientStub] = None) -> None:
        self.config = config
        self.redis = redis_client or RedisClientStub()
        # Per-trace mutable state
        self._traces: Dict[str, Dict[str, Any]] = {}

    def start_trace(self, trace_id: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize internal buffers for a new trace.
        """
        if trace_id in self._traces:
            # Reset state if re-used
            self._traces.pop(trace_id, None)
        self._traces[trace_id] = {
            "params": params or {},
            "window": [],              # list of token payloads in current window
            "window_idx": 0,           # current window index
            "shard_keys": [],          # keys written so far
            "token_count": 0,
            "pipe": self.redis.pipeline(),
        }

    def _commit_window(self, trace_id: str) -> None:
        st = self._traces[trace_id]
        window: List[Dict[str, Any]] = st["window"]
        if not window:
            return
        widx: int = int(st["window_idx"])
        key = shard_key(trace_id, widx, namespace=self.config.namespace)
        # Minimal shard payload
        shard_payload = {
            "trace_id": trace_id,
            "w": widx,
            "tokens": window,  # [{t: idx, i: [...], v: [...]}]
        }
        packed = pack_shard(shard_payload, compress=self.config.compress)
        st["pipe"].set(key, packed).execute()  # execute per-window to simulate pipelined write
        st["shard_keys"].append(key)
        st["window"] = []
        st["window_idx"] = widx + 1

    def capture_token(self, trace_id: str, token_idx: int, tensor_summary: Union[bytes, memoryview, List[float]]) -> None:
        """
        Capture a single token's activation summary.
        tensor_summary can be a list[float], numpy array (via utils), or bytes/memoryview (JSON array).
        """
        if trace_id not in self._traces:
            raise KeyError(f"Trace not started: {trace_id}")
        st = self._traces[trace_id]

        idxs, vals = topk_quantize(tensor_summary, self.config.topk)
        st["window"].append({"t": int(token_idx), "i": idxs, "v": vals})
        st["token_count"] = int(st["token_count"]) + 1

        if len(st["window"]) >= int(self.config.window_size):
            self._commit_window(trace_id)

    def flush(self, trace_id: str) -> Envelope:
        """
        Write any partial window and return an envelope describing the trace capture.
        Envelope fields:
          - trace_id, model_name, model_hash
          - num_tokens, num_shards
          - shards: list of shard keys written
        """
        if trace_id not in self._traces:
            raise KeyError(f"Trace not started: {trace_id}")
        self._commit_window(trace_id)
        st = self._traces[trace_id]
        env: Envelope = {
            "trace_id": trace_id,
            "model_name": self.config.model_name,
            "model_hash": self.config.model_hash,
            "num_tokens": int(st["token_count"]),
            "num_shards": len(st["shard_keys"]),
            "shards": list(st["shard_keys"]),
        }
        # Optionally clear state to free memory; caller may choose to keep for debugging
        # self._traces.pop(trace_id, None)
        return env


def register_layer_hook(pipeline: CapturePipeline, trace_id: str):
    """
    Create a simple adapter function a model layer can call on each token output.

    Example integration:
        hook = register_layer_hook(pipeline, trace_id)
        for t, acts in enumerate(layer_stream):
            hook(t, acts)

    The hook simply forwards to pipeline.capture_token.
    """
    def _hook(token_idx: int, tensor_summary: Union[bytes, memoryview, List[float]]) -> None:
        pipeline.capture_token(trace_id, token_idx, tensor_summary)
    return _hook