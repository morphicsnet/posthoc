from __future__ import annotations

"""
S3 persistence for HIF artifacts with optional local fallback.

Key layout (canonical):
  {prefix}/{yyyy}/{mm}/{dd}/{trace_id}/{granularity}-{sae_version}-{model_hash}.hif.json.gz

Defaults and env:
- S3_BUCKET (required for S3 mode; if absent, local fallback is used)
- S3_PREFIX (default "traces")
- S3_KMS_KEY_ID (optional; enables SSE-KMS when provided)

Lifecycle recommendations (documentation only; configure in your S3 lifecycle policies):
- Retain recent artifacts for 7 days (TTL=7).
- Transition to Intelligent-Tiering after 24 hours.
- Transition to Glacier (or Deep Archive) after 30 days.
"""

import gzip
import json
import os
import pathlib
import sys
from datetime import datetime
from typing import Any, Dict, Optional


def _safe(val: Optional[str]) -> str:
    """
    Path-safe sanitizer for dynamic key components.
    Keeps alnum, dot, underscore, and hyphen. Everything else becomes '-'.
    Empty or None becomes 'unknown'.
    """
    s = (val or "unknown").strip()
    if not s:
        s = "unknown"
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    return "".join(ch if ch in allowed else "-" for ch in s)[:128]


class S3Store:
    """
    Uploads gzip-compressed JSON to S3 (if configured) or to a local fallback.

    __init__(kms_key_id: Optional[str], bucket: Optional[str], prefix: str = "traces")
    put_json_gz(trace_id: str, payload: dict, meta: dict) -> str

    - Builds a key per UTC date and meta:
      {prefix}/{yyyy}/{mm}/{dd}/{trace_id}/{granularity}-{sae_version}-{model_hash}.hif.json.gz
      where:
        granularity  := meta.get("granularity") or "unknown"
        sae_version  := meta.get("sae_dictionary") or meta.get("sae_version") or meta.get("version") or "unknown"
        model_hash   := meta.get("model_hash") or "unknown"
    - Returns:
        s3://bucket/key (S3 mode) OR file:///tmp/hif/<key> (local fallback)
    - S3 object metadata:
        ContentType=application/json, ContentEncoding=gzip
        SSE-KMS if kms_key_id is provided.
    """

    def __init__(self, kms_key_id: Optional[str] = None, bucket: Optional[str] = None, prefix: str = "traces") -> None:
        self.kms_key_id: Optional[str] = (kms_key_id or os.getenv("S3_KMS_KEY_ID") or None)
        self.bucket: Optional[str] = bucket or os.getenv("S3_BUCKET")
        self.prefix: str = os.getenv("S3_PREFIX", prefix) or "traces"

        self._s3 = None
        if self.bucket:
            try:
                import boto3  # type: ignore

                self._s3 = boto3.client("s3")
            except Exception as e:
                print(f"[S3Store] boto3 unavailable or init failed: {e}", file=sys.stderr)
                self._s3 = None
                # Force local fallback if boto3 is not available
                self.bucket = None

        # Local fallback root
        self._local_root = pathlib.Path("/tmp/hif")
        self._local_root.mkdir(parents=True, exist_ok=True)

    def _build_key(self, trace_id: str, meta: Dict[str, Any]) -> str:
        now = datetime.utcnow()
        yyyy = f"{now.year:04d}"
        mm = f"{now.month:02d}"
        dd = f"{now.day:02d}"
        # SECURITY: slugify dynamic components (see docs/security/SECURITY.md)
        granularity = _safe(meta.get("granularity"))
        sae_version = _safe(meta.get("sae_dictionary") or meta.get("sae_version") or meta.get("version"))
        model_hash = _safe(meta.get("model_hash"))
        tid = _safe(trace_id)
        prefix = (self.prefix or "traces").strip("/")
        filename = f"{granularity}-{sae_version}-{model_hash}.hif.json.gz"
        key = f"{prefix}/{yyyy}/{mm}/{dd}/{tid}/{filename}"
        # SECURITY: refuse path traversal and collapse accidental doubles
        parts = key.split("/")
        if any(p in (".", "..") for p in parts):
            raise ValueError("unsafe s3 key path traversal detected")
        while "//" in key:
            key = key.replace("//", "/")
        return key

    def put_json_gz(self, trace_id: str, payload: Dict[str, Any], meta: Dict[str, Any]) -> str:
        """
        Serialize payload to JSON, gzip it, and persist.
        Returns a URI string:
          - s3://bucket/key when S3 is configured
          - file:///tmp/hif/<key> when falling back to local
        """
        key = self._build_key(trace_id, meta)
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        gz_bytes = gzip.compress(data, compresslevel=5)

        # S3 path
        if self.bucket and self._s3:
            args = {
                "Bucket": self.bucket,
                "Key": key,
                "Body": gz_bytes,
                "ContentType": "application/json",
                "ContentEncoding": "gzip",
            }
            if self.kms_key_id:
                # SSE-KMS
                args["ServerSideEncryption"] = "aws:kms"
                args["SSEKMSKeyId"] = self.kms_key_id
            try:
                self._s3.put_object(**args)  # type: ignore[arg-type]
                return f"s3://{self.bucket}/{key}"
            except Exception as e:
                print(f"[S3Store] S3 put_object failed, falling back to local: {e}", file=sys.stderr)

        # Local fallback
        path = self._local_root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gz_bytes)
        return f"file://{path}"

__all__ = ["S3Store"]