from __future__ import annotations

import json
import os
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Optional numeric backends (lazy usage)
try:  # numpy is optional
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


# ----------------------------
# Config and data containers
# ----------------------------

@dataclass
class SAEConfig:
    """
    SAE loader configuration.

    Fields:
      - root_path: Root directory that contains one or more SAE dictionaries
      - dictionary_name: The dictionary directory name to load (e.g., "sae-gpt4-2m")
      - device: "auto" | "cpu" | "cuda"
      - dtype: "fp16" (default) for data storage on load. Internal accumulations use float32.
      - prefer_sparse: Keep CSR format for CPU path. GPU residency is a future enhancement.
      - cache_layers: Number of layers to keep resident (LRU) at once
    """
    root_path: Union[str, Path]
    dictionary_name: str
    device: str = "auto"
    dtype: str = "fp16"
    prefer_sparse: bool = True
    cache_layers: int = 2


class SAELayerWeights:
    """
    Holds a single layer's encoder weights using CSR format:

      W_enc: features x hidden_dim
      - data: float16 values of non-zeros
      - indices: int32 column indices for non-zeros
      - indptr: int32 row pointer array of length (features + 1)
      - shape: tuple[int, int] = (features, hidden_dim)
      - bias: optional float16 vector of length features

    Notes:
      - This class is device-agnostic for now (CPU baseline). GPU residency
        can be layered in without changing call sites.
    """

    def __init__(
        self,
        *,
        data: Any,
        indices: Any,
        indptr: Any,
        shape: Tuple[int, int],
        bias: Optional[Any] = None,
        device: str = "cpu",
    ) -> None:
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = (int(shape[0]), int(shape[1]))
        self.bias = bias
        self.device = device  # "cpu" or "cuda" (future)

        self._validate_shapes()

    def _validate_shapes(self) -> None:
        rows, cols = self.shape
        if rows <= 0 or cols <= 0:
            raise ValueError(f"Invalid shape for encoder CSR: {self.shape}")
        # Validate arrays roughly
        try:
            nnz = int(len(self.data))
            if nnz < 0:
                raise ValueError
        except Exception:
            raise ValueError("Invalid CSR: data has no length")

        try:
            if len(self.indices) != nnz:
                raise ValueError("CSR indices length must equal data length")
        except Exception:
            raise ValueError("Invalid CSR: indices has no length")

        try:
            if len(self.indptr) != (rows + 1):
                raise ValueError("CSR indptr length must be rows+1")
        except Exception:
            raise ValueError("Invalid CSR: indptr has no length")

        if self.bias is not None:
            try:
                if len(self.bias) != rows:
                    raise ValueError("Bias length must equal number of features (rows)")
            except Exception:
                raise ValueError("Invalid bias: has no length")

    @property
    def nnz(self) -> int:
        try:
            return int(len(self.data))
        except Exception:
            return 0


class SAEDictionary:
    """
    Dictionary handle that lazily loads layer CSR matrices with LRU caching.

    Access:
      - get_layer(layer_idx: int) -> SAELayerWeights
      - meta: Dict[str, Any]
      - cfg: SAEConfig
    """

    def __init__(self, root_dir: Path, meta: Dict[str, Any], cfg: SAEConfig) -> None:
        self.root_dir = root_dir
        self.meta = meta
        self.cfg = cfg
        self._cache: "Dict[int, SAELayerWeights]" = {}
        self._lru: "List[int]" = []  # keys ordered least-recent to most-recent
        self._gpu_enabled = _decide_gpu(cfg.device)

    def _touch_lru(self, key: int) -> None:
        if key in self._lru:
            self._lru.remove(key)
        self._lru.append(key)
        # Evict if capacity exceeded
        while self.cfg.cache_layers >= 0 and len(self._lru) > self.cfg.cache_layers:
            evict_key = self._lru.pop(0)
            self._cache.pop(evict_key, None)

    def get_layer(self, layer_idx: int) -> SAELayerWeights:
        if not isinstance(layer_idx, int) or layer_idx < 0:
            raise ValueError(f"Layer index must be a non-negative int, got: {layer_idx}")

        if layer_idx in self._cache:
            self._touch_lru(layer_idx)
            return self._cache[layer_idx]

        layer = _load_layer_weights(self.root_dir, layer_idx, dtype=self.cfg.dtype)
        # Device residency strategy (placeholder: keep CSR on CPU)
        layer.device = "cuda" if self._gpu_enabled else "cpu"

        self._cache[layer_idx] = layer
        self._touch_lru(layer_idx)
        return layer


# ----------------------------
# Public Loader API
# ----------------------------

def load_dictionary(cfg: SAEConfig) -> SAEDictionary:
    """
    Load dictionary metadata and return a SAEDictionary with lazy layer loading.
    """
    root_dir = _resolve_dictionary_root(cfg.root_path, cfg.dictionary_name)
    meta_path = root_dir / "meta.json"
    if not meta_path.exists():
        raise ValueError(f"Missing meta.json at: {meta_path}")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Failed to parse meta.json: {e}")

    _validate_meta(meta, required_fields=["version", "model_hash", "feature_count", "dtype", "format"])

    # Optional: layer_ranges sanity (either dict or list)
    if "layer_ranges" in meta and not isinstance(meta["layer_ranges"], (list, dict)):
        raise ValueError("meta.json: 'layer_ranges' must be a list or dict if present")

    # dtype compatibility
    if cfg.dtype not in ("fp16",):
        raise ValueError(f"Unsupported dtype requested: {cfg.dtype}; only 'fp16' supported")

    return SAEDictionary(root_dir=root_dir, meta=meta, cfg=cfg)


# ----------------------------
# CPU baseline projection API
# ----------------------------

def project_topk(
    layer: SAELayerWeights,
    dense_vec: Union[List[float], Sequence[float], memoryview, bytes],
    k: int = 256,
) -> List[Tuple[int, float]]:
    """
    Compute approximate top-k feature activations for a single layer using CPU only.

    Assumptions:
      - layer uses CSR with rows = features and columns = hidden_dim.
      - dense_vec length must equal hidden_dim.
      - Returns list of (feature_index, score) sorted by descending score.

    Implementation:
      - If numpy is available, use vectorized gather-dot per CSR row.
      - Otherwise, pure-Python accumulation with a fixed-size min-heap for top-k.
    """
    rows, hidden_dim = layer.shape
    vec = _as_float32_vector(dense_vec, expected_len=hidden_dim)

    # Min-heap of (score, idx) to keep only top-k
    heap: List[Tuple[float, int]] = []
    push = heapq.heappush
    replace = heapq.heapreplace

    use_numpy = np is not None and isinstance(vec, np.ndarray)

    if use_numpy:
        # Ensure vec is float32 contiguous
        vec_np: "np.ndarray" = vec.astype(np.float32, copy=False)  # type: ignore
        data_np: "np.ndarray" = _ensure_numpy(layer.data, dtype=np.float16)  # type: ignore
        idx_np: "np.ndarray" = _ensure_numpy(layer.indices, dtype=np.int32)  # type: ignore
        ptr_np: "np.ndarray" = _ensure_numpy(layer.indptr, dtype=np.int32)  # type: ignore
        bias_np: Optional["np.ndarray"] = None
        if layer.bias is not None:
            bias_np = _ensure_numpy(layer.bias, dtype=np.float16)  # type: ignore

        for i in range(rows):
            s = int(ptr_np[i])
            e = int(ptr_np[i + 1])
            if e <= s:
                score = float(bias_np[i].astype(np.float32)) if bias_np is not None else 0.0  # type: ignore
            else:
                cols = idx_np[s:e].astype(np.int64, copy=False)
                vals = data_np[s:e].astype(np.float32, copy=False)
                # dot(vec[cols], vals)
                score = float((vec_np[cols] * vals).sum())
                if bias_np is not None:
                    score += float(bias_np[i].astype(np.float32))  # type: ignore
            if len(heap) < k:
                push(heap, (score, i))
            else:
                if score > heap[0][0]:
                    replace(heap, (score, i))
    else:
        # Pure Python path
        # Extract buffers as Python-friendly containers
        indices = _to_py_list(layer.indices)
        indptr = _to_py_list(layer.indptr)
        data = _to_py_list(layer.data)
        bias = _to_py_list(layer.bias) if layer.bias is not None else None

        for i in range(rows):
            s = int(indptr[i])
            e = int(indptr[i + 1])
            acc = 0.0
            # Loop over non-zeros in row i
            for p in range(s, e):
                j = int(indices[p])
                if 0 <= j < hidden_dim:
                    acc += float(data[p]) * float(vec[j])
            if bias is not None:
                acc += float(bias[i])
            if len(heap) < k:
                push(heap, (acc, i))
            else:
                if acc > heap[0][0]:
                    replace(heap, (acc, i))

    # Convert to sorted (idx, score) descending
    heap.sort(key=lambda x: x[0], reverse=True)
    return [(idx, float(score)) for (score, idx) in ((s, i) for (s, i) in heap)]


# ----------------------------
# Helpers: I/O and conversions
# ----------------------------

def _resolve_dictionary_root(root_path: Union[str, Path], dictionary_name: str) -> Path:
    root = Path(root_path)
    if not root.exists():
        raise ValueError(f"root_path does not exist: {root}")
    # Allow either: root/dictionary_name OR root itself named as dictionary_name
    cand = root / dictionary_name
    if cand.exists():
        return cand
    # If the given root is already at the dictionary
    if (root / "meta.json").exists():
        # sanity: name check
        return root
    raise ValueError(f"Dictionary '{dictionary_name}' not found under: {root}")

def _validate_meta(meta: Dict[str, Any], required_fields: List[str]) -> None:
    for f in required_fields:
        if f not in meta:
            raise ValueError(f"meta.json missing required field: '{f}'")
    # Minimal format check
    fmt = str(meta.get("format", "")).lower()
    if fmt not in ("csr.npz", "csr_npz", "npz"):
        # Allow unknown-but-documented formats; we currently support npz-based CSR only
        pass

def _decide_gpu(device: str) -> bool:
    device = (device or "auto").lower()
    if device == "cpu":
        return False
    if device == "cuda" or device == "gpu" or device == "auto":
        try:
            import torch  # type: ignore
            return bool(torch.cuda.is_available())  # pragma: no cover
        except Exception:
            return False
    return False

def _load_layer_weights(root_dir: Path, layer_idx: int, dtype: str = "fp16") -> SAELayerWeights:
    layer_dir = root_dir / f"layer_{layer_idx}"
    if not layer_dir.exists():
        raise ValueError(f"Layer directory missing: {layer_dir}")

    enc_npz = layer_dir / "encoder.npz"
    if not enc_npz.exists():
        raise ValueError(f"Encoder NPZ missing: {enc_npz}")

    if np is None:
        raise RuntimeError(
            f"numpy is required to load NPZ artifacts but is not available. Failed on: {enc_npz}"
        )

    with np.load(str(enc_npz), allow_pickle=False) as z:  # type: ignore
        # Expect scipy-like CSR keys
        if not all(k in z for k in ("data", "indices", "indptr", "shape")):
            raise ValueError(
                f"encoder.npz must contain keys data/indices/indptr/shape; got keys={list(z.keys())}"
            )
        data = z["data"]
        indices = z["indices"]
        indptr = z["indptr"]
        shape_arr = z["shape"]

        # Normalize dtypes and shapes
        data = data.astype(np.float16, copy=False) if dtype == "fp16" else data.astype(np.float32, copy=False)
        indices = indices.astype(np.int32, copy=False)
        indptr = indptr.astype(np.int32, copy=False)

        # shape may be stored as array([rows, cols])
        if hasattr(shape_arr, "shape"):
            if shape_arr.size != 2:
                raise ValueError(f"Bad shape field in encoder.npz; expected size 2, got {shape_arr}")
            rows, cols = int(shape_arr[0]), int(shape_arr[1])
        else:
            # Fallback for tuple saved
            rows, cols = int(shape_arr[0]), int(shape_arr[1])

    # Optional bias
    bias_path = layer_dir / "bias.npy"
    bias = None
    if bias_path.exists():
        try:
            bias = np.load(str(bias_path), allow_pickle=False)  # type: ignore
            bias = bias.astype(np.float16, copy=False) if dtype == "fp16" else bias.astype(np.float32, copy=False)
            if bias.shape[0] != rows:
                raise ValueError(f"bias length {bias.shape[0]} must match features {rows}")
        except Exception as e:  # pragma: no cover
            raise ValueError(f"Failed to load bias.npy: {e}")

    return SAELayerWeights(
        data=data,
        indices=indices,
        indptr=indptr,
        shape=(rows, cols),
        bias=bias,
        device="cpu",
    )

def _ensure_numpy(arr: Any, dtype: Any) -> "np.ndarray":
    if np is None:
        raise RuntimeError("numpy required for this path")
    if isinstance(arr, np.ndarray):
        return arr.astype(dtype, copy=False)  # type: ignore
    # Convert common sequences
    return np.asarray(arr, dtype=dtype)  # type: ignore

def _to_py_list(arr: Any) -> List[Any]:
    if arr is None:
        return []
    try:
        if np is not None and isinstance(arr, np.ndarray):  # type: ignore
            return arr.tolist()  # type: ignore
    except Exception:
        pass
    try:
        return list(arr)  # type: ignore
    except Exception:
        # As a last resort, attempt to iterate
        return [x for x in arr]  # type: ignore

def _as_float32_vector(dense_vec: Union[List[float], Sequence[float], memoryview, bytes], expected_len: int):
    """
    Convert the input vector to a float32 vector (numpy array if available, otherwise Python list).
    Accepts: list-like floats, memoryview of 'f', or bytes of length expected_len*4.
    """
    # numpy fast path
    if np is not None:
        if isinstance(dense_vec, np.ndarray):  # type: ignore
            vec_np: "np.ndarray" = dense_vec.astype(np.float32, copy=False)  # type: ignore
            if vec_np.ndim != 1 or vec_np.shape[0] != expected_len:
                raise ValueError(f"dense_vec length {vec_np.shape[0]} != expected {expected_len}")
            return vec_np
        if isinstance(dense_vec, (list, tuple)):
            vec_np = np.asarray(dense_vec, dtype=np.float32)  # type: ignore
            if vec_np.ndim != 1 or vec_np.shape[0] != expected_len:
                raise ValueError(f"dense_vec length {vec_np.shape[0]} != expected {expected_len}")
            return vec_np
        if isinstance(dense_vec, (memoryview, bytes)):
            buf = memoryview(dense_vec) if isinstance(dense_vec, (bytes, bytearray)) else dense_vec  # type: ignore
            # Interpret as float32
            if buf.nbytes != expected_len * 4:
                raise ValueError(f"bytes length {buf.nbytes} does not match expected float32 size {expected_len*4}")
            vec_np = np.frombuffer(buf, dtype=np.float32, count=expected_len)  # type: ignore
            return vec_np

    # Pure Python fallback
    if isinstance(dense_vec, (list, tuple)):
        if len(dense_vec) != expected_len:
            raise ValueError(f"dense_vec length {len(dense_vec)} != expected {expected_len}")
        # Already list of floats
        return [float(x) for x in dense_vec]

    if isinstance(dense_vec, (memoryview, bytes, bytearray)):
        buf = memoryview(dense_vec) if not isinstance(dense_vec, memoryview) else dense_vec
        if buf.nbytes != expected_len * 4:
            raise ValueError(f"bytes length {buf.nbytes} does not match expected float32 size {expected_len*4}")
        # Interpret as little-endian 32-bit floats
        # memoryview with format 'f' if available
        try:
            mv = buf.cast("f")  # type: ignore
            return [float(x) for x in mv]
        except Exception:
            # Manual unpack (slow path); avoid struct to keep deps minimal in tight loops
            raise ValueError("memoryview casting to float32 failed; provide list[float] input for CPU fallback")

    raise TypeError("Unsupported dense_vec type; provide list[float], memoryview('f'), or float32 bytes")