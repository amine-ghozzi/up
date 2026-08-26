"""Pydantic API DTOs (pure — no torch). fastapi-users user schemas + partner API-key schemas.

Extraction-result DTOs (Phase 4) serialize ``ExtractionResult`` via ``dataclasses.asdict()`` +
per-table ``CanonicalTable.to_dict()`` and live alongside these.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi_users import schemas
from pydantic import BaseModel, ConfigDict


# --- Users (extend fastapi-users base schemas) ---------------------------
class UserRead(schemas.BaseUser[uuid.UUID]):
    role: str = "analyst"
    org_id: uuid.UUID | None = None
    full_name: str | None = None


class UserCreate(schemas.BaseUserCreate):
    role: str = "analyst"
    org_id: uuid.UUID | None = None
    full_name: str | None = None


class UserUpdate(schemas.BaseUserUpdate):
    role: str | None = None
    full_name: str | None = None


# --- Partner API keys -----------------------------------------------------
class ApiKeyCreate(BaseModel):
    name: str
    scopes: list[str] = []
    expires_at: datetime | None = None


class ApiKeyRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    name: str
    prefix: str
    scopes: list[str]
    active: bool
    expires_at: datetime | None = None
    created_at: datetime


class ApiKeyCreated(ApiKeyRead):
    """Returned only at mint time — carries the one-time plaintext key."""
    api_key: str


# --- Documents & jobs -----------------------------------------------------
class SubmitResponse(BaseModel):
    job_id: uuid.UUID
    document_id: uuid.UUID
    status: str


class JobSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    status: str
    tier_used: int | None = None
    qcs_score: float | None = None
    hitl_required: bool = False
    error: str | None = None


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    filename: str
    accounting_standard: str
    state: str
    entity: str | None = None
    period: str | None = None
    created_at: datetime


class DocumentDetail(DocumentRead):
    latest_job: JobSummary | None = None


class PageResult(BaseModel):
    items: list[DocumentRead]
    limit: int
    offset: int
    total: int


# --- HITL review / validation --------------------------------------------
class ReviewQueueItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    filename: str
    state: str
    accounting_standard: str
    created_at: datetime
    hitl_required: bool = False
    qcs_score: float | None = None


class ValidationView(BaseModel):
    document_id: uuid.UUID
    state: str
    version_id: int
    assignee_id: uuid.UUID | None = None
    report: dict | None = None
    conflicts: list = []
    flagged_cells: list = []


class CellCorrection(BaseModel):
    table_index: int | None = None
    row_index: int | None = None
    column_name: str | None = None
    raw_text: str | None = None
    old_value: str | None = None
    new_value: str | None = None
    corrected_canonical: str | None = None


class CellsPatch(BaseModel):
    corrections: list[CellCorrection]


class CellsPatchResult(BaseModel):
    applied: int
    variation_candidates: int
    validation_version: int


class TransitionRequest(BaseModel):
    to_state: str            # validated | archived | rejected
    reason: str | None = None


class AssignRequest(BaseModel):
    assignee_id: uuid.UUID | None = None  # None → claim to self


class AuditEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    action: str
    actor_kind: str
    actor_id: uuid.UUID | None = None
    detail: dict | None = None
    created_at: datetime


# --- Feedback loop --------------------------------------------------------
class VariationCandidateRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    raw_text: str
    matched_to: str | None = None
    confidence: float | None = None
    match_type: str | None = None
    status: str
    seen_count: int
    created_at: datetime


class QualityKPIs(BaseModel):
    jobs_total: int
    jobs_done: int
    automation_rate: float
    hitl_rate: float
    avg_qcs: float | None = None
    corrections_total: int
    correction_rate: float


class RuleFlagStats(BaseModel):
    jobs_evaluated: int
    avg_pass_rate: float | None = None
    critical_failure_rate: float


# --- Config ---------------------------------------------------------------
class OrgConfigRead(BaseModel):
    org_id: uuid.UUID
    config: dict


class OrgConfigUpdate(BaseModel):
    config: dict


class StandardInfo(BaseModel):
    code: str
    name: str


class NomenclatureTerm(BaseModel):
    canonical_term: str
    statement_type: str
    account_code: str | None = None


# --- Webhooks -------------------------------------------------------------
class WebhookCreate(BaseModel):
    url: str
    events: list[str] = []


class WebhookRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    url: str
    events: list[str]
    active: bool
    created_at: datetime


class WebhookCreated(WebhookRead):
    secret: str  # shown once at creation


class WebhookDeliveryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    endpoint_id: uuid.UUID
    event: str
    status_code: int | None = None
    attempts: int
    delivered: bool
    created_at: datetime
