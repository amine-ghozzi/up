"""Worker runtime helpers: lazy pipeline, document fetch, DB persistence, result serialization.

DB writes use a **fresh async engine per call with NullPool** — loop-safe across the
``asyncio.run()`` calls a sync Celery task makes (avoids "attached to a different loop").
"""

from __future__ import annotations

import json
import logging
import tempfile
import uuid
from pathlib import Path

from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from api.models import Document, Job, JobStatus
from api.settings import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()

_pipeline = None  # loaded once per worker process


def get_pipeline():
    """Lazy-load ``FinAlzePipeline`` once per prefork child (heavy import deferred here)."""
    global _pipeline
    if _pipeline is None:
        from pipeline import FinAlzePipeline  # noqa: PLC0415 — intentional deferred import

        logger.info("Loading FinAlzePipeline (once per worker process)…")
        _pipeline = FinAlzePipeline()
    return _pipeline


def _uid(value: str) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _engine():
    return create_async_engine(str(_settings.database_url), poolclass=NullPool)


def _json_safe(obj):
    """Coerce to JSON-serializable (handles Decimal/numpy/datetime via ``default=str``)."""
    return json.loads(json.dumps(obj, default=str))


def serialize_result(result) -> dict:
    """``ExtractionResult`` (dataclass, no ``to_dict``) → JSON-safe dict for the Job row."""
    canonical = [
        ct.to_dict() for ct in (getattr(result, "canonical_tables", None) or [])
        if hasattr(ct, "to_dict")
    ]
    return _json_safe(
        {
            "text": result.text,
            "tables": result.tables,
            "qcs_score": result.qcs_score,
            "tier_used": result.tier_used,
            "confidence_details": result.confidence_details,
            "metadata": result.metadata,
            "canonical_tables": canonical,
        }
    )


def fetch_document(storage_key: str, filename: str) -> Path:
    """Return a local path to the source file — from MinIO/S3 if configured, else a local path."""
    if _settings.s3_endpoint_url and _settings.s3_access_key:
        from minio import Minio  # noqa: PLC0415

        endpoint = _settings.s3_endpoint_url.split("://", 1)[-1]
        client = Minio(
            endpoint,
            access_key=_settings.s3_access_key,
            secret_key=_settings.s3_secret_key,
            secure=_settings.s3_secure,
        )
        fd, tmp = tempfile.mkstemp(suffix=Path(filename).suffix)
        Path(tmp).unlink(missing_ok=True)  # fget_object writes the file
        import os

        os.close(fd)
        client.fget_object(_settings.s3_bucket, storage_key, tmp)
        return Path(tmp)
    # Dev fallback: storage_key is a local filesystem path.
    return Path(storage_key)


# --- persistence -----------------------------------------------------------


async def load_job_context(job_id: str) -> dict | None:
    """Mark the job running and return the document context (or None if missing)."""
    eng = _engine()
    sm = async_sessionmaker(eng, expire_on_commit=False)
    try:
        async with sm() as s:
            job = await s.get(Job, _uid(job_id))
            if job is None:
                return None
            doc = await s.get(Document, job.document_id)
            job.status = JobStatus.running
            await s.commit()
            return {
                "document_id": str(doc.id),
                "storage_key": doc.storage_key,
                "filename": doc.filename,
                "standard": doc.accounting_standard,
            }
    finally:
        await eng.dispose()


async def save_success(job_id: str, result: dict, tier_used, qcs, hitl: bool) -> None:
    eng = _engine()
    sm = async_sessionmaker(eng, expire_on_commit=False)
    try:
        async with sm() as s:
            await s.execute(
                update(Job).where(Job.id == _uid(job_id)).values(
                    status=JobStatus.done, result=result, tier_used=tier_used,
                    qcs_score=qcs, hitl_required=hitl, error=None,
                )
            )
            await s.commit()
    finally:
        await eng.dispose()


async def save_failure(job_id: str, error) -> None:
    eng = _engine()
    sm = async_sessionmaker(eng, expire_on_commit=False)
    try:
        async with sm() as s:
            await s.execute(
                update(Job).where(Job.id == _uid(job_id)).values(
                    status=JobStatus.failed, error=str(error)[:4000],
                )
            )
            await s.commit()
    finally:
        await eng.dispose()
