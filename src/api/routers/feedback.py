"""Feedback loop — variation promotion, rule-flag stats, quality KPIs, ground-truth export.

This is what makes corrections *compound* and powers the foundation-first accuracy gate.
Reads/writes need `feedback:write`.
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db import get_async_session
from api.deps import Pagination, pagination
from api.models import (
    Correction,
    Document,
    Job,
    JobStatus,
    VariationCandidate,
    VariationStatus,
)
from api.schemas import (
    QualityKPIs,
    RuleFlagStats,
    VariationCandidateRead,
)
from api.security import Principal, require_scopes

router = APIRouter(prefix="/feedback", tags=["feedback"])


def _org(principal: Principal) -> uuid.UUID | None:
    return uuid.UUID(principal.org_id) if principal.org_id else None


@router.get("/variations", response_model=list[VariationCandidateRead])
async def list_variations(
    status_filter: str | None = Query(default=None, alias="status"),
    principal: Principal = Depends(require_scopes("feedback:write")),
    session: AsyncSession = Depends(get_async_session),
    pg: Pagination = Depends(pagination),
) -> list[VariationCandidateRead]:
    stmt = select(VariationCandidate)
    org = _org(principal)
    if org:
        stmt = stmt.where(VariationCandidate.org_id == org)
    if status_filter:
        stmt = stmt.where(VariationCandidate.status == VariationStatus(status_filter))
    rows = (
        await session.execute(
            stmt.order_by(VariationCandidate.created_at.desc()).limit(pg.limit).offset(pg.offset)
        )
    ).scalars().all()
    return [VariationCandidateRead.model_validate(r) for r in rows]


async def _get_owned_candidate(session, cand_id, principal) -> VariationCandidate:
    cand = await session.get(VariationCandidate, cand_id)
    if cand is None or (principal.org_id and cand.org_id and str(cand.org_id) != principal.org_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Variation candidate not found")
    return cand


@router.post("/variations/{cand_id}/approve", response_model=VariationCandidateRead)
async def approve_variation(
    cand_id: uuid.UUID,
    principal: Principal = Depends(require_scopes("feedback:write")),
    session: AsyncSession = Depends(get_async_session),
) -> VariationCandidateRead:
    cand = await _get_owned_candidate(session, cand_id, principal)
    cand.status = VariationStatus.promoted
    # Bridge into the pipeline's hot-reload store so future extractions benefit (best-effort).
    try:
        from accounting.variation_logger import VariationLogger  # noqa: PLC0415

        if cand.matched_to:
            VariationLogger().log_hitl_correction(cand.raw_text, cand.matched_to)
    except Exception:  # noqa: BLE001 — logger is best-effort
        pass
    await session.commit()
    return VariationCandidateRead.model_validate(cand)


@router.post("/variations/{cand_id}/reject", response_model=VariationCandidateRead)
async def reject_variation(
    cand_id: uuid.UUID,
    principal: Principal = Depends(require_scopes("feedback:write")),
    session: AsyncSession = Depends(get_async_session),
) -> VariationCandidateRead:
    cand = await _get_owned_candidate(session, cand_id, principal)
    cand.status = VariationStatus.rejected
    await session.commit()
    return VariationCandidateRead.model_validate(cand)


@router.get("/rules/flags", response_model=RuleFlagStats)
async def rule_flags(
    principal: Principal = Depends(require_scopes("feedback:write")),
    session: AsyncSession = Depends(get_async_session),
) -> RuleFlagStats:
    stmt = select(Job).where(Job.status == JobStatus.done)
    org = _org(principal)
    if org:
        stmt = stmt.where(Job.org_id == org)
    jobs = (await session.execute(stmt)).scalars().all()
    rates: list[float] = []
    crit = 0
    for j in jobs:
        rep = ((j.result or {}).get("metadata") or {}).get("validation_report") or {}
        total, passed = rep.get("total_checks"), rep.get("passed_checks")
        if total:
            rates.append((passed or 0) / total)
        if rep.get("critical_failures"):
            crit += 1
    return RuleFlagStats(
        jobs_evaluated=len(jobs),
        avg_pass_rate=(sum(rates) / len(rates)) if rates else None,
        critical_failure_rate=(crit / len(jobs)) if jobs else 0.0,
    )


@router.get("/quality", response_model=QualityKPIs)
async def quality(
    principal: Principal = Depends(require_scopes("feedback:write")),
    session: AsyncSession = Depends(get_async_session),
) -> QualityKPIs:
    org = _org(principal)
    jstmt = select(Job)
    if org:
        jstmt = jstmt.where(Job.org_id == org)
    jobs = (await session.execute(jstmt)).scalars().all()
    done = [j for j in jobs if j.status == JobStatus.done]
    hitl = [j for j in done if j.hitl_required]
    qcs = [j.qcs_score for j in done if j.qcs_score is not None]

    def _count(model):
        s = select(func.count()).select_from(model)
        if org:
            s = s.where(model.org_id == org)
        return s

    corrections_total = (await session.execute(_count(Correction))).scalar_one()
    docs_total = (await session.execute(_count(Document))).scalar_one()
    return QualityKPIs(
        jobs_total=len(jobs), jobs_done=len(done),
        automation_rate=((len(done) - len(hitl)) / len(done)) if done else 0.0,
        hitl_rate=(len(hitl) / len(done)) if done else 0.0,
        avg_qcs=(sum(qcs) / len(qcs)) if qcs else None,
        corrections_total=corrections_total,
        correction_rate=(corrections_total / docs_total) if docs_total else 0.0,
    )


@router.get("/ground-truth")
async def ground_truth(
    principal: Principal = Depends(require_scopes("feedback:write")),
    session: AsyncSession = Depends(get_async_session),
) -> Response:
    """Export HITL corrections as a JSONL labeled dataset (golden-set / calibration source)."""
    stmt = select(Correction)
    org = _org(principal)
    if org:
        stmt = stmt.where(Correction.org_id == org)
    rows = (await session.execute(stmt.order_by(Correction.created_at.asc()))).scalars().all()
    lines = [
        json.dumps(
            {
                "document_id": str(c.document_id),
                "table_index": c.table_index,
                "row_index": c.row_index,
                "column_name": c.column_name,
                "raw_text": c.raw_text,
                "corrected_value": c.new_value,
                "corrected_canonical": c.corrected_canonical,
            },
            ensure_ascii=False,
        )
        for c in rows
    ]
    return Response("\n".join(lines), media_type="application/x-ndjson")
