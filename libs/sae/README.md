# SAE Dictionary Loader (production-ready skeleton)

This library provides a minimal, production-ready loader and CPU projection utilities for large Sparse Autoencoder (SAE) dictionaries. It focuses on:
- Lightweight runtime: optional NumPy, no torch required for baseline CPU path
- Memory-aware sparse formats (CSR), half precision payloads
- Pluggable device strategy: CPU baseline now, GPU residency/pinning hooks preserved for later
- Stable on-disk artifact layout with semver-style versioning

Core implementation lives in:
- libs/sae/loader.py
- libs/sae/cli.py (smoke CLI)


## 1) On-disk artifact layout and versioning

A dictionary is a directory named by featureset (e.g., `sae-gpt4-2m`):

```
sae-gpt4-2m/
  meta.json                      # manifest with required fields
  layer_0/
    encoder.npz                  # CSR: data/indices/indptr/shape (float16/int32/int32/[rows,cols])
    bias.npy                     # optional float16 bias vector of length = feature_count for the layer
  layer_1/
    encoder.npz
  ...
  layer_12/
    encoder.npz
    bias.npy
```

Required `meta.json` fields:
- `version` (string, semver-like, e.g. `"1.0.0"`)
- `model_hash` (string or hex digest for the base model)
- `feature_count` (integer; total features across layers or per-layer summary)
- `dtype` (string; recommended `"fp16"`)
- `format` (string; e.g. `"csr.npz"` to indicate format for `encoder.npz`)
- Optional: `layer_ranges` (list or dict) for index ranges per layer

Example `meta.json`:

```json
{
  "version": "1.0.0",
  "model_hash": "sha256:deadbeef...",
  "feature_count": 2000000,
  "dtype": "fp16",
  "format": "csr.npz",
  "layer_ranges": {
    "0": [0, 16384],
    "12": [16384, 32768]
  }
}
```

Accepted formats (current):
- CSR in `.npz` per layer with SciPy-like arrays:
  - `data` (float16)
  - `indices` (int32)
  - `indptr` (int32)
  - `shape` (array of `[rows, cols]` = `[features, hidden_dim]`)
- Optional bias vector `bias.npy` (float16) per layer

Notes:
- A single consolidated npz per layer (carrying those keys) is also acceptable.
- Room is reserved for future `safetensors` / `.pt` (torch) artifacts without changing call sites.


## 2) Loader API (no heavy deps required)

Public API is provided by libs/sae/loader.py:

- `SAEConfig`: Configuration for loading and caching
- `SAELayerWeights`: Container for a single layer’s CSR encoder and optional bias
- `SAEDictionary`: Lazy loader with LRU caching of hot layers
- `load_dictionary(cfg: SAEConfig) -> SAEDictionary`
- `project_topk(layer: SAELayerWeights, dense_vec, k=256) -> list[(feature_idx, score)]` (CPU baseline)

Key behaviors:
- CSR is interpreted as `W_enc` shaped `[features, hidden_dim]`
- Projection computes: `score_i = sum_j W_enc[i,j] * x[j] (+ bias[i] if present)`, returning top-k features by score
- Data is stored as `fp16` for `data`/`bias`; compute accumulations use float32 internally
- Indices are int32

Device strategy:
- `device="auto"` attempts CUDA detection (import-guarded, no hard torch dependency)
- Baseline path keeps CSR on CPU; GPU residency and dense tiling for active layers are left for a future fast-path
- LRU cache size is configurable via `SAEConfig.cache_layers`


## 3) CPU-only fast path: project_topk

`project_topk` accepts:
- a `SAELayerWeights` instance
- a dense activation vector of length `hidden_dim`
- optional `k` (top-k features; default 256)

Implementation details:
- If NumPy is present, per-row gather-dot over CSR (vectorized per row) with float32 accumulation
- Else, pure-Python loop with a fixed-size min-heap to keep top-k results
- Returns a list of `(feature_index, score)` sorted by descending `score`
- ReLU/normalization are intentionally not applied here; callers can clamp/filter after


## 4) CLI smoke test

A simple CLI is provided to test a single layer load and projection.

Usage:
```
python -m libs.sae.cli --root /path/to/sae-gpt4-2m --dict sae-gpt4-2m --layer 12 --topk 10
```

Behavior:
- Loads `meta.json` and the given layer’s `encoder.npz`
- Generates a random dense probe vector of length `hidden_dim` from the encoder’s `shape`
- Prints top-k features with scores
- Works without torch; will prefer NumPy if available (faster), otherwise pure Python fallback


## 5) Optional integration in Explainer worker

The worker can optionally use this loader path when enabled via env:

- `USE_SAE_LOADER=1` to enable
- `DICT_ROOT=/absolute/or/relative/path/containing/dictionary`
- Dictionary name currently expected: `"sae-gpt4-2m"`

If enabled and artifacts are present, the worker will:
- Load config `SAEConfig(root_path=DICT_ROOT, dictionary_name="sae-gpt4-2m")`
- Fetch a representative layer (e.g., layer 12)
- Create a synthetic dense vector (deterministic seed from trace id in current stub)
- Call `project_topk(..., k=256)` and map indices to `feat_{idx}` with the returned scores

If anything fails (import guard, missing numpy for `.npz`, missing files), the worker gracefully falls back to the existing stub behavior.


## 6) Performance and batching tips

- Use half-precision `fp16` payloads for `data`/`bias` to reduce disk and memory footprint
- Keep CSR row-major by feature; projection is row-wise sparse dot with an input vector
- Batch tokens by:
  - Reusing the same layer load across multiple vectors via `SAEDictionary` caching
  - Potential future dense-tiling on GPU: transform CSR to compact dense tiles for active rows and do GEMM
- Pin hot layers with `SAEConfig.cache_layers` size; memory-balanced eviction via LRU
- For very large `k`, consider a two-stage selection (coarse heap, then refine)


## 7) Error handling and validation

- `meta.json` is validated for required fields and basic typing
- Each layer checks:
  - presence of `encoder.npz`
  - CSR keys `data/indices/indptr/shape`
  - shape consistency
  - `bias.npy` length matches features if present
- On failure, clear and informative `ValueError` or `RuntimeError` are raised by the loader
- The worker’s adapter keeps default behavior as fallback


## 8) Example: Python usage

```python
from libs.sae.loader import SAEConfig, load_dictionary, project_topk  # optional NumPy; no torch required

cfg = SAEConfig(
    root_path="/models/sae",
    dictionary_name="sae-gpt4-2m",
    device="auto",
    dtype="fp16",
    prefer_sparse=True,
    cache_layers=2,
)
dct = load_dictionary(cfg)
layer12 = dct.get_layer(12)

# probe vector length must equal hidden_dim
_, hidden_dim = layer12.shape
dense = [0.0] * hidden_dim
dense[0] = 1.0

top = project_topk(layer12, dense, k=10)
for idx, score in top:
    print(f"feat_{idx}: {score:.6f}")
```

## 9) Future enhancements

- Accept `safetensors` and `.pt` variants without changing public API
- CUDA fast path: keep LRU hot layers as dense tiles on GPU and batch projections
- Optional Cupy/cuSPARSE adapters (import-guarded; never required at build time)


## 10) Run commands

CLI:
```
python -m libs.sae.cli --root /path/to/sae-gpt4-2m --dict sae-gpt4-2m --layer 12 --topk 10
```

Enable optional path in worker:
```
export USE_SAE_LOADER=1
export DICT_ROOT=/path/to/sae-dictionaries
# run explainer worker as usual (defaults remain unchanged if not set)
```

Notes:
- Real SAE weights are not included in this repo and must be provided externally.
- NumPy is optional but required to load `.npz` artifacts. If NumPy is not available, loading will raise a clear error; the worker integration will catch it and fall back to stubs.