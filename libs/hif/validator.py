"""
HIF v2 Schema Validator

Public API:
- validate_hypergraph(obj: dict) -> None
- validate_explanation_response(obj: dict) -> None

Both raise jsonschema.ValidationError on invalid inputs and return None on success.

Example:
    from libs.hif.validator import validate_hypergraph, validate_explanation_response
    validate_hypergraph({"nodes": [], "hyperedges": []})
    validate_explanation_response({
        "id": "chatcmpl-123",
        "status": "completed",
        "hypergraph": {"nodes": [], "hyperedges": []}
    })

TODO(version2): Keep this aligned with OpenAPI components in [`api/openapi/hypergraph-api.yaml`](api/openapi/hypergraph-api.yaml:1). Source of truth is this schema; OpenAPI should mirror it.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None  # type: ignore

# TODO(dep): ensure 'jsonschema>=4.18' is listed in libs/requirements.txt
try:
    import jsonschema
    from jsonschema import Draft202012Validator, RefResolver
except Exception:  # pragma: no cover
    print(
        "[validator] Missing dependency 'jsonschema'. "
        "Install with: pip install 'jsonschema>=4.18'",
        file=sys.stderr,
    )
    raise


SCHEMA_PATH = Path(__file__).parent / "schema.json"


def _loads(data: bytes) -> Any:
    if orjson is not None:
        try:
            return orjson.loads(data)  # type: ignore[attr-defined]
        except Exception:
            pass
    return json.loads(data.decode("utf-8"))


def _read_file(path: Path) -> bytes:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return f.read()
    return path.read_bytes()


def load_schema(path: Path | None = None) -> Dict[str, Any]:
    """
    Load the HIF JSON Schema (Draft 2020-12).
    """
    p = path or SCHEMA_PATH
    raw = _read_file(p)
    return _loads(raw)


# Load schema and build resolver at import time
SCHEMA: Dict[str, Any] = load_schema()
# TODO(version2): Keep this aligned with OpenAPI components in [`api/openapi/hypergraph-api.yaml`](api/openapi/hypergraph-api.yaml:1). Source of truth is this schema; OpenAPI should mirror it.
RESOLVER: RefResolver = RefResolver.from_schema(SCHEMA)


def _validate_at(pointer: str, obj: Dict[str, Any]) -> None:
    """
    Validate obj against a subschema addressed by a JSON Pointer within the loaded schema.
    """
    _, subschema = RESOLVER.resolve(pointer)
    Draft202012Validator(subschema, resolver=RESOLVER).validate(obj)


def validate_hypergraph(obj: Dict[str, Any]) -> None:
    """
    Validate a pure Hypergraph payload (nodes, hyperedges).
    Raises jsonschema.ValidationError on invalid input; returns None on success.
    """
    _validate_at("#/$defs/Hypergraph", obj)


def validate_explanation_response(obj: Dict[str, Any]) -> None:
    """
    Validate the full ExplanationResponse envelope (id, status, hypergraph, optional provider/model/created_at).
    Raises jsonschema.ValidationError on invalid input; returns None on success.
    """
    _validate_at("#/$defs/ExplanationResponse", obj)


def is_valid(schema_pointer: str, obj: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Helper that returns (True, None) if valid else (False, error_message).
    """
    try:
        _validate_at(schema_pointer, obj)
        return True, None
    except jsonschema.ValidationError as ve:  # pragma: no cover
        return False, ve.message


# Back-compat validator for any HIF payload (v1 legacy or v2 objects)
def validate_hif(obj: Dict[str, Any], schema: Dict[str, Any] | None = None) -> None:
    """
    Backward-compatible validator that accepts either:
    - v2 Hypergraph
    - v2 ExplanationResponse
    - v2 ExplanationPending
    - v1 legacy HIF graph
    """
    sch = schema or SCHEMA
    Draft202012Validator(sch, resolver=RESOLVER).validate(obj)


def _main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python libs/hif/validator.py /path/to/payload.json[.gz]", file=sys.stderr)
        return 2

    in_path = Path(argv[1])
    if not in_path.exists():
        print(f"[validator] File not found: {in_path}", file=sys.stderr)
        return 2

    try:
        obj = _loads(_read_file(in_path))
    except Exception as e:
        print(f"[validator] Failed to read/parse input: {e}", file=sys.stderr)
        return 2

    try:
        validate_hif(obj, SCHEMA)
    except jsonschema.ValidationError as ve:
        print("[validator] INVALID")
        print(f"  path: {'/'.join(map(str, ve.path))}")
        print(f"  message: {ve.message}")
        if ve.schema_path:
            print(f"  schema_path: {'/'.join(map(str, ve.schema_path))}")
        try:
            snippet = obj
            try:
                s = json.dumps(snippet, indent=2)[:2000]
            except Exception:
                s = str(snippet)[:2000]
            print("  instance:", s)
        except Exception:
            pass
        return 1
    except Exception as e:
        print(f"[validator] Validation failed unexpectedly: {e}", file=sys.stderr)
        return 2

    print("[validator] VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))