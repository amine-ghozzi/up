"""Documents — submit (upload → store → enqueue), list, get, result.

Submit is scoped `documents:submit`; reads are `documents:read`. All queries are tenant-scoped
by `principal.org_id`. Idempotency-Key dedupes partner retries. Uploads are streamed with a size
cap and an extension allowlist (fuller AV/sniffing hardening is a documented follow-up).
"""

from __future__ import annotations

import io
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api import storage
from api.celery_app import enqueue_extraction
from api.db import get_async_session
from api.deps import Pagination, pagination
from api.ratelimit import submit_rate_limit
from api.models import DocState, Document, Job, JobStatus
from api.schemas import (
    DocumentDetail,
    DocumentRead,
    JobSummary,
    PageResult,
    SubmitResponse,
)
from api.security import Principal, require_scopes
from api.settings import get_settings

router = APIRouter(prefix="/documents", tags=["documents"])
_settings = get_settings()

_ALLOWED_EXT = {".pdf", ".docx", ".jpg", ".jpeg", ".png", ".tiff", ".bmp"}


async def _read_capped(file: UploadFile, max_bytes: int) -> bytes:
    buf = io.BytesIO()
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Upload exceeds {_settings.max_upload_mb} MB",
            )
        buf.write(chunk)
    return buf.getvalue()


def _require_org(principal: Principal) -> uuid.UUID:
    if not principal.org_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="No org context for this principal")
    return uuid.UUID(principal.org_id)


async def _owned_document(session: AsyncSession, document_id: uuid.UUID, principal: Principal) -> Document:
    doc = await session.get(Document, document_id)
    if doc is None or (principal.org_id and str(doc.org_id) != principal.org_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=SubmitResponse)
async def submit_document(
    file: UploadFile = File(...),
    accounting_standard: str | None = Form(default=None),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    principal: Principal = Depends(require_scopes("documents:submit")),
    session: AsyncSession = Depends(get_async_session),
    _rl: None = Depends(submit_rate_limit),
) -> SubmitResponse:
    org_id = _require_org(principal)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_EXT:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=f"Unsupported file type: {ext or '(none)'}"
        )

    if idempotency_key:
        existing = (
            await session.execute(
                select(Document).where(
                    Document.org_id == org_id, Document.idempotency_key == idempotency_key
                )
            )
        ).scalars().first()
        if existing is not None:
            job = (
                await session.execute(
                    select(Job).where(Job.document_id == existing.id).order_by(Job.created_at.desc())
                )
            ).scalars().first()
            return SubmitResponse(job_id=job.id, document_id=existing.id, status=job.status.value)

    data = await _read_capped(file, _settings.max_upload_bytes)
    standard = accounting_standard or _settings.default_accounting_standard
    storage_key = storage.store_upload(str(org_id), file.filename or "upload", data, file.content_type)

    doc = Document(
        org_id=org_id, filename=file.filename or "upload", storage_key=storage_key,
        content_type=file.content_type, size_bytes=len(data), accounting_standard=standard,
        idempotency_key=idempotency_key,
    )
    session.add(doc)
    await session.flush()
    job = Job(document_id=doc.id, org_id=org_id, status=JobStatus.queued)
    session.add(job)
    await session.flush()

    job.celery_task_id = enqueue_extraction(str(job.id))
    await session.commit()
    return SubmitResponse(job_id=job.id, document_id=doc.id, status=job.status.value)


@router.get("", response_model=PageResult)
async def list_documents(
    state: str | None = None,
    principal: Principal = Depends(require_scopes("documents:read")),
    session: AsyncSession = Depends(get_async_session),
    pg: Pagination = Depends(pagination),
) -> PageResult:
    stmt = select(Document)
    if principal.org_id:
        stmt = stmt.where(Document.org_id == uuid.UUID(principal.org_id))
    if state:
        stmt = stmt.where(Document.state == DocState(state))
    total = (await session.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one()
    rows = (
        await session.execute(
            stmt.order_by(Document.created_at.desc()).limit(pg.limit).offset(pg.offset)
        )
    ).scalars().all()
    return PageResult(
        items=[DocumentRead.model_validate(r) for r in rows],
        limit=pg.limit, offset=pg.offset, total=total,
    )


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: uuid.UUID,
    principal: Principal = Depends(require_scopes("documents:read")),
    session: AsyncSession = Depends(get_async_session),
) -> DocumentDetail:
    doc = await _owned_document(session, document_id, principal)
    job = (
        await session.execute(
            select(Job).where(Job.document_id == doc.id).order_by(Job.created_at.desc())
        )
    ).scalars().first()
    detail = DocumentDetail.model_validate(doc)
    if job is not None:
        detail.latest_job = JobSummary.model_validate(job)
    return detail


@router.get("/{document_id}/result")
async def get_result(
    document_id: uuid.UUID,
    principal: Principal = Depends(require_scopes("documents:read")),
    session: AsyncSession = Depends(get_async_session),
) -> dict:
    doc = await _owned_document(session, document_id, principal)
    job = (
        await session.execute(
            select(Job).where(Job.document_id == doc.id, Job.status == JobStatus.done)
            .order_by(Job.created_at.desc())
        )
    ).scalars().first()
    if job is None or job.result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No completed result for this document")
    return job.result
