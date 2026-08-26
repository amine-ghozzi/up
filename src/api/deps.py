"""Shared FastAPI dependencies: pagination and tenant scoping."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Query

from api.security import Principal


@dataclass
class Pagination:
    limit: int
    offset: int


def pagination(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Pagination:
    return Pagination(limit=limit, offset=offset)


def tenant_org_id(principal: Principal) -> str | None:
    """Org to scope queries by. Partners are always scoped; users without an org (admins)
    see across orgs (None → no filter)."""
    if principal.kind == "partner":
        return principal.org_id
    return principal.org_id  # may be None for cross-org admins
