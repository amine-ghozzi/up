"""HITL review workflow — queue, validation view, cell corrections (→ feedback loop),
state machine (Draft→Validated→Archived/Rejected) with optimistic-lock 409, and audit trail.

Reads need `documents:read`; mutations need `hitl:write`. Corrections are recorded as
`Correction` rows and, when a canonical mapping is supplied, also seed an **approved**
`VariationCandidate` (closing the feedback loop into the DB). Every mutation writes an `AuditEvent`.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm.exc import StaleDataError
from sqlalchemy.ext.asyncio import AsyncSession

from api.db import get_async_session
from api.models import (
    AuditEvent,
    Correction,
    DocState,
    Document,
    Job,
    JobStatus,
    Validation,
    VariationCandidate,
    VariationStatus,
)
from api.schemas import (
    AssignRequest,
    AuditEventRead,
    CellsPatch,
    CellsPatchResult,
    ReviewQueueItem,
    TransitionRequest,
    ValidationView,
)
from api.security import Principal, require_scopes

router = APIRouter(tags=["review"])

_ALLOWED_TRANSITIONS: dict[DocState, set[DocState]] = {
    DocState.draft: {DocState.validated, DocState.rejected},
    DocState.validated: {DocState.archived},
}


# --- helpers --------------------------------------------------------------


def _actor_user_id(principal: Principal) -> uuid.UUID | None:
    return uuid.UUID(principal.id) if principal.kind == "user" else None


async def _owned_document(session: AsyncSession, document_id: uuid.UUID, principal: Principal) -> Document:
    doc = await session.get(Document, document_id)
    if doc is None or (principal.org_id and str(doc.org_id) != principal.org_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc


async def _latest_done_job(session: AsyncSession, doc: Document) -> Job | None:
    return (
        await session.execute(
            select(Job).where(Job.document_id == doc.id, Job.status == JobStatus.done)
            .order_by(Job.created_at.desc())
        )
    ).scalars().first()


async def _get_or_create_validation(session: AsyncSession, doc: Document) -> Validation:
    v = (
        await session.execute(
            select(Validation).where(Validation.document_id == doc.id)
            .order_by(Validation.created_at.desc())
        )
    ).scalars().first()
    if v is not None:
        return v
    job = await _latest_done_job(session, doc)
    report = ((job.result or {}).get("metadata") or {}).get("validation_report") if job else None
    v = Validation(document_id=doc.id, org_id=doc.org_id, state=doc.state, report=report)
    session.add(v)
    await session.flush()
    return v


def _derive_flags(job_result: dict | None):
    conflicts: list = []
    flagged: list = []
    for ti, ct in enumerate((job_result or {}).get("canonical_tables") or []):
        conflicts.extend(ct.get("conflicts") or [])
        for ri, row in enumerate(ct.get("rows") or []):
            for col, cell in (row.get("cells") or {}).items():
                if cell.get("flag") and cell["flag"] != "green":
                    flagged.append({
                        "table_index": ti, "row_index": ri, "column_name": col,
                        "flag": cell["flag"], "raw_value": cell.get("raw_value"),
                    })
    return conflicts, flagged


async def _audit(session, principal, action, document_id, detail=None) -> None:
    session.add(AuditEvent(
        org_id=uuid.UUID(principal.org_id) if principal.org_id else None,
        actor_id=uuid.UUID(principal.id), actor_kind=principal.kind,
        action=action, target_type="document", target_id=str(document_id), detail=detail,
    ))


# --- endpoints ------------------------------------------------------------


@router.get("/review/queue", response_model=list[ReviewQueueItem])
async def review_queue(
    principal: Principal = Depends(require_scopes("documents:read")),
    session: AsyncSession = Depends(get_async_session),
) -> list[ReviewQueueItem]:
    stmt = select(Document).where(Document.state == DocState.draft)
    if principal.org_id:
        stmt = stmt.where(Document.org_id == uuid.UUID(principal.org_id))
    docs = (await session.execute(stmt.order_by(Document.created_at.desc()))).scalars().all()
    items: list[ReviewQueueItem] = []
    for doc in docs:
        job = (
            await session.execute(
                select(Job).where(Job.document_id == doc.id).order_by(Job.created_at.desc())
            )
        ).scalars().first()
        item = ReviewQueueItem.model_validate(doc)
        if job is not None:
            item.hitl_required = job.hitl_required
            item.qcs_score = job.qcs_score
        items.append(item)
    return items


@router.get("/documents/{document_id}/validation", response_model=ValidationView)
async def get_validation(
    document_id: uuid.UUID,
    principal: Principal = Depends(require_scopes("documents:read")),
    session: AsyncSession = Depends(get_async_session),
) -> ValidationView:
    doc = await _owned_document(session, document_id, principal)
    v = await _get_or_create_validation(session, doc)
    job = await _latest_done_job(session, doc)
    conflicts, flagged = _derive_flags(job.result if job else None)
    await session.commit()
    return ValidationView(
        document_id=doc.id, state=v.state.value, version_id=v.version_id,
        assignee_id=v.assignee_id, report=v.report, conflicts=conflicts, flagged_cells=flagged,
    )


@router.post("/documents/{document_id}/assign", response_model=ValidationView)
async def assign(
    document_id: uuid.UUID,
    body: AssignRequest,
    principal: Principal = Depends(require_scopes("hitl:write")),
    session: AsyncSession = Depends(get_async_session),
) -> ValidationView:
    doc = await _owned_document(session, document_id, principal)
    v = await _get_or_create_validation(session, doc)
    v.assignee_id = body.assignee_id or _actor_user_id(principal)
    await _audit(session, principal, "review.assign", doc.id, {"assignee_id": str(v.assignee_id)})
    await session.commit()
    return ValidationView(document_id=doc.id, state=v.state.value, version_id=v.version_id,
                          assignee_id=v.assignee_id, report=v.report)


@router.patch("/documents/{document_id}/cells", response_model=CellsPatchResult)
async def patch_cells(
    document_id: uuid.UUID,
    body: CellsPatch,
    if_match: int | None = Header(default=None, alias="If-Match"),
    principal: Principal = Depends(require_scopes("hitl:write")),
    session: AsyncSession = Depends(get_async_session),
) -> CellsPatchResult:
    doc = await _owned_document(session, document_id, principal)
    v = await _get_or_create_validation(session, doc)
    if if_match is not None and if_match != v.version_id:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=f"Stale validation version (current {v.version_id})")

    var_count = 0
    for corr in body.corrections:
        session.add(Correction(
            document_id=doc.id, org_id=doc.org_id, user_id=_actor_user_id(principal),
            table_index=corr.table_index, row_index=corr.row_index, column_name=corr.column_name,
            raw_text=corr.raw_text, old_value=corr.old_value, new_value=corr.new_value,
            corrected_canonical=corr.corrected_canonical,
        ))
        # Feedback loop: a human-confirmed label mapping → approved variation candidate.
        if corr.corrected_canonical and corr.raw_text:
            session.add(VariationCandidate(
                org_id=doc.org_id, raw_text=corr.raw_text, matched_to=corr.corrected_canonical,
                match_type="hitl_correction", confidence=1.0, status=VariationStatus.approved,
                document_id=doc.id,
            ))
            var_count += 1

    # Cells are append-only Correction rows; the Validation row isn't mutated here, so its
    # version_id (managed by version_id_col, bumped on transition) stays stable.
    await _audit(session, principal, "review.cells", doc.id, {"count": len(body.corrections)})
    await session.commit()
    return CellsPatchResult(applied=len(body.corrections), variation_candidates=var_count,
                            validation_version=v.version_id)


@router.post("/documents/{document_id}/transition", response_model=ValidationView)
async def transition(
    document_id: uuid.UUID,
    body: TransitionRequest,
    if_match: int | None = Header(default=None, alias="If-Match"),
    principal: Principal = Depends(require_scopes("hitl:write")),
    session: AsyncSession = Depends(get_async_session),
) -> ValidationView:
    doc = await _owned_document(session, document_id, principal)
    v = await _get_or_create_validation(session, doc)

    try:
        target = DocState(body.to_state)
    except ValueError:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Unknown state {body.to_state!r}")
    if target not in _ALLOWED_TRANSITIONS.get(doc.state, set()):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"Illegal transition {doc.state.value} → {target.value}",
        )
    if if_match is not None and if_match != v.version_id:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=f"Stale validation version (current {v.version_id})")

    v.state = target  # dirty → ORM version_id_col bumps + guards via WHERE version
    doc.state = target
    await _audit(session, principal, "review.transition", doc.id,
                 {"to": target.value, "reason": body.reason})
    try:
        await session.commit()
    except StaleDataError:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Concurrent modification; reload and retry")
    return ValidationView(document_id=doc.id, state=v.state.value, version_id=v.version_id,
                          assignee_id=v.assignee_id, report=v.report)


@router.get("/documents/{document_id}/audit", response_model=list[AuditEventRead])
async def get_audit(
    document_id: uuid.UUID,
    principal: Principal = Depends(require_scopes("documents:read")),
    session: AsyncSession = Depends(get_async_session),
) -> list[AuditEventRead]:
    doc = await _owned_document(session, document_id, principal)
    events = (
        await session.execute(
            select(AuditEvent).where(
                AuditEvent.target_type == "document", AuditEvent.target_id == str(doc.id)
            ).order_by(AuditEvent.created_at.asc())
        )
    ).scalars().all()
    return [AuditEventRead.model_validate(e) for e in events]
