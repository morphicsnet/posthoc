from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Observability (optional)
try:
    from services.explainer.src import otel as _otel  # type: ignore
except Exception:
    try:
        import otel as _otel  # type: ignore
    except Exception:
        _otel = None  # type: ignore
from collections import deque

# Do NOT import torch at module import time. Import inside GPU code paths only.

try:
    # Numpy optional - used for vector conversions only if present.
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

# Loader components (CPU path + dictionary/CSR)
from libs.sae.loader import (
    SAEConfig as _LoaderConfig,
    SAEDictionary,
    SAELayerWeights,
    load_dictionary,
    project_topk,
)

VecLike = Union[List[float], memoryview, bytes]

@dataclass
class SAEDecodeConfig:
    dict_root: str
    dictionary_name: str = "sae-gpt4-2m"
    device: str = "auto"  # "cuda" | "cpu" | "auto"
    dtype: str = "fp16"
    cache_layers: int = 3  # number of hot layers to keep resident
    batch_size: int = 32
    max_wait_ms: int = 8   # batch latency cap
    topk: int = 256
    tile_rows: int = 65536 # features per tile when materializing dense on GPU

class _Pending:
    __slots__ = ("vec", "fut", "topk_override", "cache_layers_override")
    def __init__(self, vec: VecLike, fut: "asyncio.Future[Dict[int, float]]", topk_override: Optional[int] = None, cache_layers_override: Optional[int] = None) -> None:
        self.vec = vec
        self.fut = fut
        self.topk_override = topk_override
        self.cache_layers_override = cache_layers_override

class _LayerQueue:
    __slots__ = ("items", "lock", "timer_task")
    def __init__(self) -> None:
        self.items: deque[_Pending] = deque()
        self.lock = asyncio.Lock()
        self.timer_task: Optional[asyncio.Task] = None

class _LayerRep:
    __slots__ = ("weights", "last_used")
    def __init__(self, weights: SAELayerWeights) -> None:
        self.weights = weights
        self.last_used = time.monotonic()

def _torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())  # pragma: no cover
    except Exception:
        return False

def _decide_device(user: str) -> str:
    u = (user or "auto").lower()
    if u == "cpu":
        return "cpu"
    if u in ("cuda", "gpu"):
        return "cuda" if _torch_cuda_available() else "cpu"
    # auto
    return "cuda" if _torch_cuda_available() else "cpu"

def _to_float32_list(vec: VecLike, expected_len: int) -> List[float]:
    # Try numpy fast paths
    if np is not None:
        if isinstance(vec, np.ndarray):  # type: ignore
            arr = vec.astype(np.float32, copy=False)  # type: ignore
            if arr.ndim != 1 or arr.shape[0] != expected_len:
                raise ValueError(f"dense_vec length {arr.shape[0]} != expected {expected_len}")
            return arr.tolist()  # type: ignore
        if isinstance(vec, (list, tuple)):
            arr = np.asarray(vec, dtype=np.float32)  # type: ignore
            if arr.ndim != 1 or arr.shape[0] != expected_len:
                raise ValueError(f"dense_vec length {arr.shape[0]} != expected {expected_len}")
            return arr.tolist()  # type: ignore
        if isinstance(vec, (memoryview, bytes, bytearray)):
            buf = memoryview(vec) if not isinstance(vec, memoryview) else vec
            if buf.nbytes != expected_len * 4:
                raise ValueError(f"bytes length {buf.nbytes} does not match expected float32 size {expected_len*4}")
            arr = np.frombuffer(buf, dtype=np.float32, count=expected_len)  # type: ignore
            return arr.tolist()  # type: ignore
    # Pure Python
    if isinstance(vec, (list, tuple)):
        if len(vec) != expected_len:
            raise ValueError(f"dense_vec length {len(vec)} != expected {expected_len}")
        return [float(x) for x in vec]
    if isinstance(vec, (memoryview, bytes, bytearray)):
        buf = memoryview(vec) if not isinstance(vec, memoryview) else vec
        if buf.nbytes != expected_len * 4:
            raise ValueError(f"bytes length {buf.nbytes} does not match expected float32 size {expected_len*4}")
        try:
            mv = buf.cast("f")  # type: ignore
            return [float(x) for x in mv]
        except Exception:
            raise ValueError("memoryview casting to float32 failed; provide list[float] input")
    raise TypeError("Unsupported dense_vec type; provide list[float], memoryview('f'), or float32 bytes")

class SAEDecodeService:
    """
    In-process async microservice that performs SAE decode (top-k projection) with batching, hot-layer cache,
    and optional CUDA backend with tiling by layer features.
    """
    def __init__(self, cfg: SAEDecodeConfig) -> None:
        self.cfg = cfg
        self.logger = logging.getLogger("explainer.sae_service")
        # Load dictionary (lazy per-layer)
        loader_cfg = _LoaderConfig(
            root_path=cfg.dict_root,
            dictionary_name=cfg.dictionary_name,
            device=cfg.device,
            dtype=cfg.dtype,
            prefer_sparse=True,
            cache_layers=max(1, int(cfg.cache_layers)),
        )
        self._dict: SAEDictionary = load_dictionary(loader_cfg)
        self._device: str = _decide_device(cfg.device)
        self._queues: Dict[int, _LayerQueue] = {}
        self._rep_cache: Dict[int, _LayerRep] = {}
        self._rep_lru: List[int] = []
        self._running = False
        self._closed = False
        self._last_decode_fallback = False

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("SAEDecodeService already closed")
        self._running = True
        self.logger.info(
            f"SAEDecodeService started device={self._device} dict='{self.cfg.dictionary_name}' "
            f"batch_size={self.cfg.batch_size} max_wait_ms={self.cfg.max_wait_ms} cache_layers={self.cfg.cache_layers} topk={self.cfg.topk}"
        )

    async def stop(self) -> None:
        self._running = False
        # Cancel timers and fail pending gracefully
        for q in list(self._queues.values()):
            try:
                if q.timer_task:
                    q.timer_task.cancel()
            except Exception:
                pass
            async with q.lock:
                while q.items:
                    p = q.items.popleft()
                    if not p.fut.done():
                        p.fut.set_exception(asyncio.CancelledError("SAEDecodeService stopping"))
        self._closed = True
        self.logger.info("SAEDecodeService stopped")

    async def decode(self, layer: int, dense_vec: VecLike, *, override_topk: Optional[int] = None, override_cache_layers: Optional[int] = None) -> Dict[int, float]:
        """
        Enqueue a single decode request and await batched result.
        Returns: mapping feature_index -> score (top-k).

        override_topk: optional per-call k override for backpressure
        override_cache_layers: optional per-call cache_layers override to limit memory
        """
        if not self._running:
            raise RuntimeError("SAEDecodeService not started")
        if not isinstance(layer, int) or layer < 0:
            raise ValueError(f"layer must be non-negative int, got {layer}")
        q = self._queues.get(layer)
        if q is None:
            q = self._queues[layer] = _LayerQueue()
        fut: "asyncio.Future[Dict[int, float]]" = asyncio.get_event_loop().create_future()
        async with q.lock:
            q.items.append(_Pending(dense_vec, fut, override_topk, override_cache_layers))
            if len(q.items) >= max(1, int(self.cfg.batch_size)):
                await self._flush_now(layer, q)
            else:
                # arm timer if not already present
                if q.timer_task is None or q.timer_task.done():
                    q.timer_task = asyncio.create_task(self._flush_after_delay(layer, self.cfg.max_wait_ms / 1000.0))
        return await fut

    async def _flush_after_delay(self, layer: int, delay_s: float) -> None:
        try:
            await asyncio.sleep(max(0.0, float(delay_s)))
            q = self._queues.get(layer)
            if q is None:
                return
            async with q.lock:
                if q.items:
                    await self._flush_now(layer, q)
        except asyncio.CancelledError:
            return
        except Exception as e:
            self.logger.warning(f"_flush_after_delay error: {e}")

    def _get_layer_rep(self, layer: int, effective_cache_layers: Optional[int] = None) -> SAELayerWeights:
        rep = self._rep_cache.get(layer)
        if rep is not None:
            rep.last_used = time.monotonic()
            # bump LRU
            if layer in self._rep_lru:
                self._rep_lru.remove(layer)
            self._rep_lru.append(layer)
            if _otel is not None:
                try:
                    _otel.update_sae_layer_cache_size(len(self._rep_cache))
                except Exception:
                    pass
            return rep.weights
        # Load from dictionary
        lw = self._dict.get_layer(layer)
        self._rep_cache[layer] = _LayerRep(lw)
        self._rep_lru.append(layer)
        # Evict if needed using effective cache layers if provided
        keep = max(1, int(effective_cache_layers if effective_cache_layers is not None else self.cfg.cache_layers))
        while len(self._rep_lru) > keep:
            ev = self._rep_lru.pop(0)
            self._rep_cache.pop(ev, None)
        if _otel is not None:
            try:
                _otel.update_sae_layer_cache_size(len(self._rep_cache))
            except Exception:
                pass
        return lw

    async def _flush_now(self, layer: int, q: _LayerQueue) -> None:
        # Extract up to batch_size
        batch: List[_Pending] = []
        bs = max(1, int(self.cfg.batch_size))
        while q.items and len(batch) < bs:
            batch.append(q.items.popleft())
        # Disarm timer if empty; else start another timer for remaining pending requests
        if q.timer_task:
            try:
                q.timer_task.cancel()
            except Exception:
                pass
            q.timer_task = None
        if q.items:
            # arm a fresh timer for leftovers
            q.timer_task = asyncio.create_task(self._flush_after_delay(layer, self.cfg.max_wait_ms / 1000.0))
        # Process
        try:
            t0 = time.perf_counter()
            # Resolve effective overrides from batch (choose the most restrictive)
            eff_topk = int(self.cfg.topk)
            eff_cache_layers: Optional[int] = None
            for p in batch:
                if getattr(p, "topk_override", None) is not None:
                    try:
                        eff_topk = min(eff_topk, int(p.topk_override))  # type: ignore[arg-type]
                    except Exception:
                        pass
                if getattr(p, "cache_layers_override", None) is not None:
                    try:
                        val = int(p.cache_layers_override)  # type: ignore[arg-type]
                        eff_cache_layers = val if eff_cache_layers is None else min(eff_cache_layers, val)
                    except Exception:
                        pass

            weights = self._get_layer_rep(layer, effective_cache_layers=eff_cache_layers)
            rows, hidden_dim = weights.shape
            # Validate inputs minimally
            vecs: List[List[float]] = [_to_float32_list(p.vec, expected_len=hidden_dim) for p in batch]
            if self._device == "cuda":
                results = await self._decode_batch_gpu(weights, vecs, k=int(eff_topk), tile_rows=int(self.cfg.tile_rows))
            else:
                results = self._decode_batch_cpu(weights, vecs, k=int(eff_topk))
            # Deliver
            # record metrics
            try:
                if _otel is not None:
                    elapsed = float(time.perf_counter() - t0)
                    fb = bool(getattr(self, "_last_decode_fallback", False))
                    try:
                        self._last_decode_fallback = False
                    except Exception:
                        pass
                    _otel.sae_decode_observe(self._device, layer, len(batch), elapsed, fallback=fb)
            except Exception:
                pass
            for p, res in zip(batch, results):
                if not p.fut.done():
                    p.fut.set_result(res)
        except Exception as e:
            self.logger.exception(f"_flush_now error on layer {layer}: {e}")
            for p in batch:
                if not p.fut.done():
                    p.fut.set_exception(e)

    def _decode_batch_cpu(self, weights: SAELayerWeights, vecs: List[List[float]], k: int) -> List[Dict[int, float]]:
        out: List[Dict[int, float]] = []
        for v in vecs:
            top = project_topk(weights, v, k=k)
            out.append({int(i): float(s) for (i, s) in top})
        return out

    async def _decode_batch_gpu(self, weights: SAELayerWeights, vecs: List[List[float]], k: int, tile_rows: int) -> List[Dict[int, float]]:
        """
        Import-guarded CUDA path with tiling. Falls back to CPU if torch unavailable or any CUDA error occurs.
        Strategy per tile [f0:f1):
          - Materialize dense chunk D of shape (hidden_dim, tile_cols) on device
          - Compute S = V @ D  where V is (B, hidden_dim)
          - Take per-row top-k and merge across tiles
        """
        try:
            import torch  # type: ignore
        except Exception:
            # Fallback to CPU
            try:
                if self._device == "cuda":
                    self._last_decode_fallback = True
            except Exception:
                pass
            return self._decode_batch_cpu(weights, vecs, k)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            try:
                if self._device == "cuda":
                    self._last_decode_fallback = True
            except Exception:
                pass
            return self._decode_batch_cpu(weights, vecs, k)
        torch_dtype = torch.float16 if (self.cfg.dtype == "fp16") else torch.float32
        B = len(vecs)
        rows, H = weights.shape
        # Build batch tensor
        batch = torch.tensor(vecs, dtype=torch.float32, device=device)  # accumulate in fp32 for stability
        # Initialize accumulators
        g_vals: Optional[torch.Tensor] = None  # (B, m)
        g_idx: Optional[torch.Tensor] = None   # (B, m) global indices
        # Bias handling
        has_bias = weights.bias is not None
        # numpy views for faster slicing
        data = weights.data
        indices = weights.indices
        indptr = weights.indptr
        bias = weights.bias
        # Tile loop with OOM backoff
        tr = max(1024, int(tile_rows))
        start = 0
        while start < rows:
            end = min(rows, start + tr)
            tile_cols = end - start
            # Build dense chunk H x tile_cols
            try:
                D = torch.zeros((H, tile_cols), dtype=torch_dtype, device=device)
                # Fill columns from CSR rows
                for r in range(start, end):
                    s = int(indptr[r])
                    e = int(indptr[r + 1])
                    if e <= s:
                        continue
                    col = r - start
                    # Slice
                    idx = indices[s:e]
                    vals = data[s:e]
                    # Convert to torch
                    idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
                    vals_t = torch.as_tensor(vals, dtype=torch_dtype, device=device)
                    D.index_put_((idx_t, torch.full_like(idx_t, col)), vals_t, accumulate=False)  # set column
                # Matmul and bias
                S = batch @ D  # (B, tile_cols)
                if has_bias:
                    try:
                        b_np = bias[start:end]  # type: ignore
                        b_t = torch.as_tensor(b_np, dtype=S.dtype, device=device)
                        S = S + b_t
                    except Exception:
                        pass
                # Top-k for this tile
                k_loc = min(k, tile_cols)
                vals, idxs = torch.topk(S, k=k_loc, dim=1)
                idxs = idxs + start  # global indices
                # Merge with global
                if g_vals is None:
                    g_vals = vals
                    g_idx = idxs
                else:
                    g_vals = torch.cat([g_vals, vals], dim=1)
                    g_idx = torch.cat([g_idx, idxs], dim=1)
                    # prune to k
                    vals2, ord2 = torch.topk(g_vals, k=min(k, g_vals.shape[1]), dim=1)
                    idx2 = torch.gather(g_idx, 1, ord2)
                    g_vals, g_idx = vals2, idx2
                start = end
            except RuntimeError as rte:
                # Likely OOM -> reduce tile size
                if "CUDA" in str(rte).upper() and tr > 1024:
                    tr = max(1024, tr // 2)
                    self.logger.warning(f"CUDA OOM during tiling; reducing tile_rows to {tr}")
                    torch.cuda.empty_cache()
                    continue
                # Other CUDA error -> fallback CPU
                self.logger.warning(f"CUDA decode failed, falling back to CPU: {rte}")
                return self._decode_batch_cpu(weights, vecs, k)
        # Convert results
        assert g_vals is not None and g_idx is not None
        results: List[Dict[int, float]] = []
        g_vals = g_vals.to(dtype=torch.float32)
        g_idx = g_idx.to(dtype=torch.long)
        vals_cpu = g_vals.detach().cpu().tolist()
        idx_cpu = g_idx.detach().cpu().tolist()
        for b in range(B):
            pairs = sorted(zip(idx_cpu[b], vals_cpu[b]), key=lambda x: x[1], reverse=True)[:k]
            results.append({int(i): float(v) for (i, v) in pairs})
        return results

# ----------------------------
# Minimal async demo when run directly
# ----------------------------
async def _demo() -> None:
    logging.basicConfig(level=logging.INFO)
    import os
    dict_root = os.getenv("DICT_ROOT", "")
    if not dict_root:
        print("Set DICT_ROOT to run demo against a real dictionary. Exiting.")
        return
    layer = int(os.getenv("SAE_LAYER", "12"))
    topk = int(os.getenv("SAE_TOPK", "64"))
    bs = int(os.getenv("SAE_BATCH_SIZE", "32"))
    N = int(os.getenv("DEMO_N", "100"))
    cfg = SAEDecodeConfig(
        dict_root=dict_root,
        dictionary_name=os.getenv("DICT_NAME", "sae-gpt4-2m"),
        device=os.getenv("SAE_DEVICE", "auto"),
        dtype="fp16",
        cache_layers=int(os.getenv("SAE_CACHE_LAYERS", "3")),
        batch_size=bs,
        max_wait_ms=int(os.getenv("SAE_MAX_WAIT_MS", "8")),
        topk=topk,
        tile_rows=int(os.getenv("SAE_TILE_ROWS", "65536")),
    )
    svc = SAEDecodeService(cfg)
    await svc.start()
    # Probe hidden_dim from layer
    lw = svc._get_layer_rep(layer)
    H = lw.shape[1]
    vecs: List[List[float]] = []
    try:
        # numpy RNG for reproducibility if available
        if np is not None:
            rng = np.random.default_rng(42)  # type: ignore
            for _ in range(N):
                vecs.append(rng.standard_normal(H, dtype=np.float32).tolist())  # type: ignore
        else:
            import random
            r = random.Random(42)
            for _ in range(N):
                vecs.append([r.uniform(-1.0, 1.0) for _ in range(H)])
    except Exception:
        pass
    t0 = time.perf_counter()
    # Fire off tasks
    async def one(v):
        return await svc.decode(layer, v)
    tasks = [asyncio.create_task(one(v)) for v in vecs]
    res = await asyncio.gather(*tasks)
    t1 = time.perf_counter()
    print(f"Demo decoded N={N} avg_k={topk} device={svc._device} elapsed_ms={(t1-t0)*1000:.2f} qps={N/((t1-t0)+1e-9):.1f}")
    # Show one
    print(f"Example result keys={list(res[0].keys())[:8]}")
    await svc.stop()

if __name__ == "__main__":
    asyncio.run(_demo())