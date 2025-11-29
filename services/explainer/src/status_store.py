from __future__ import annotations

"""
Status/Index Store abstraction with local JSON persistence and a DynamoDB-like stub.

Environment:
- STATUS_BACKEND: "json" | "ddb" (default: "json"). The gateway may choose to not enable any backend
  unless this is set; the worker will default to json if available.
- STATUS_JSON_PATH: path to local JSON file (default: /tmp/hif/status.json)
- DDB_TABLE: optional table name for the DynamoDB stub (no real AWS calls)
- TRACE_TTL_DAYS: default 7 (used by callers to compute expires_at)

No AWS dependencies. boto3 is optional and only used to log client availability when using ddb stub.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional, TypedDict, Any
from time import time


class TraceStatusItem(TypedDict, total=False):
    trace_id: str
    state: str
    progress: float
    stage: str
    s3_key: Optional[str]
    error: Optional[str]
    granularity: Optional[str]
    featureset: Optional[str]
    created_at: float
    updated_at: float
    expires_at: Optional[float]


class StatusStore:
    def put_status(self, item: TraceStatusItem) -> None:
        raise NotImplementedError

    def get_status(self, trace_id: str) -> Optional[TraceStatusItem]:
        raise NotImplementedError

    def update_fields(self, trace_id: str, **fields: Any) -> None:
        raise NotImplementedError

    def delete(self, trace_id: str) -> None:
        raise NotImplementedError


class LocalJSONStatusStore(StatusStore):
    """
    Simple local JSON file backend with atomic writes.

    File format:
    {
      "<trace_id>": { ... TraceStatusItem ... },
      ...
    }
    """

    def __init__(self, file_path: str = "/tmp/hif/status.json") -> None:
        self.file_path = file_path
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        # Initialize file if missing
        if not os.path.exists(self.file_path):
            self._atomic_write({})

    def _read(self) -> Dict[str, TraceStatusItem]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # Ensure types are dicts
                return {str(k): dict(v) for k, v in data.items() if isinstance(v, dict)}
            return {}
        except FileNotFoundError:
            return {}
        except Exception:
            return {}

    def _atomic_write(self, data: Dict[str, TraceStatusItem]) -> None:
        # Write to temp file then rename to avoid partial writes
        d = os.path.dirname(self.file_path) or "."
        fd, tmp_path = tempfile.mkstemp(prefix=".status.", dir=d, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.file_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass

    def put_status(self, item: TraceStatusItem) -> None:
        if not isinstance(item, dict) or "trace_id" not in item:  # type: ignore[reportUnknownArgumentType]
            return
        now = time()
        item.setdefault("created_at", now)
        item["updated_at"] = now
        data = self._read()
        tid = str(item["trace_id"])
        existing = data.get(tid)
        # SECURITY: tenant_id is immutable once set to prevent cross-tenant hijack
        try:
            new_tenant = item.get("tenant_id")  # type: ignore[attr-defined]
            old_tenant = existing.get("tenant_id") if isinstance(existing, dict) else None  # type: ignore[attr-defined]
            if old_tenant is not None and new_tenant is not None and str(old_tenant) != str(new_tenant):
                # audit warning to stderr; ignore attempted tenant change
                try:
                    import sys
                    print(f'{{"ts":{int(now)},"event":"status.tenant_immutable_violation","trace_id":"{tid}","old":"{old_tenant}","new":"{new_tenant}"}}', file=sys.stderr)
                except Exception:
                    pass
                item["tenant_id"] = old_tenant  # type: ignore[index]
        except Exception:
            pass
        data[tid] = item  # type: ignore[index]
        self._atomic_write(data)

    def get_status(self, trace_id: str) -> Optional[TraceStatusItem]:
        data = self._read()
        item = data.get(str(trace_id))
        if not isinstance(item, dict):
            return None
        return item

    def update_fields(self, trace_id: str, **fields: Any) -> None:
        data = self._read()
        tid = str(trace_id)
        current = data.get(tid, {"trace_id": tid, "state": "unknown", "progress": 0.0})
        if not isinstance(current, dict):
            current = {"trace_id": tid}
        # SECURITY: disallow tenant_id mutation once set
        try:
            if "tenant_id" in fields and "tenant_id" in current:
                if str(fields["tenant_id"]) != str(current["tenant_id"]):
                    try:
                        import sys
                        print(f'{{"ts":{int(time())},"event":"status.tenant_immutable_violation","trace_id":"{tid}","old":"{current.get("tenant_id")}","new":"{fields.get("tenant_id")}"}}', file=sys.stderr)
                    except Exception:
                        pass
                    fields.pop("tenant_id", None)
        except Exception:
            pass
        current.update(fields)
        current["updated_at"] = time()
        data[tid] = current  # type: ignore[index]
        self._atomic_write(data)

    def delete(self, trace_id: str) -> None:
        data = self._read()
        if str(trace_id) in data:
            del data[str(trace_id)]
            self._atomic_write(data)


class DDBStatusStore(StatusStore):
    """
    No-op DynamoDB-like placeholder. Logs intended operations.
    Does not perform any AWS calls.

    If boto3 is importable and DDB_TABLE is set, this class will note client availability
    in logs but still not call AWS.
    """

    def __init__(self, table_name: Optional[str]) -> None:
        self.table_name = table_name or os.getenv("DDB_TABLE") or "unset"
        self._boto_available = False
        try:
            import boto3  # type: ignore

            _ = boto3.__version__  # only to silence lint
            self._boto_available = True
        except Exception:
            self._boto_available = False

    def _log(self, action: str, **kwargs: Any) -> None:
        # Minimal stderr logging to avoid adding logging config dependencies
        msg = f"[DDBStatusStore:{'boto' if self._boto_available else 'noboto'} table={self.table_name}] {action} {kwargs}"
        try:
            import sys

            print(msg, file=sys.stderr)
        except Exception:
            pass

    def put_status(self, item: TraceStatusItem) -> None:
        self._log("put_status", item=item)

    def get_status(self, trace_id: str) -> Optional[TraceStatusItem]:
        self._log("get_status", trace_id=trace_id)
        # No real storage; always None to force caller fallbacks
        return None

    def update_fields(self, trace_id: str, **fields: Any) -> None:
        self._log("update_fields", trace_id=trace_id, fields=fields)

    def delete(self, trace_id: str) -> None:
        self._log("delete", trace_id=trace_id)


def get_status_store_from_env() -> StatusStore:
    """
    Factory: returns a StatusStore per env.

    STATUS_BACKEND=json|ddb (default json)
    STATUS_JSON_PATH=/tmp/hif/status.json (default)
    DDB_TABLE optional (stub only)
    """
    backend = (os.getenv("STATUS_BACKEND", "json") or "json").strip().lower()
    if backend == "ddb":
        return DDBStatusStore(table_name=os.getenv("DDB_TABLE"))
    # default: json
    path = os.getenv("STATUS_JSON_PATH", "/tmp/hif/status.json")
    return LocalJSONStatusStore(file_path=path)


__all__ = [
    "TraceStatusItem",
    "StatusStore",
    "LocalJSONStatusStore",
    "DDBStatusStore",
    "get_status_store_from_env",
]