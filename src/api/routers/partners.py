"""Partner admin — create partner orgs and mint/revoke hashed API keys (scope: admin).

The plaintext key is returned **once** at mint; only its Argon2 hash is stored.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.db import get_async_session
from api.models import ApiKey, Org
from api.schemas import ApiKeyCreate, ApiKeyCreated
from api.security import Principal, generate_api_key, require_scopes

router = APIRouter(prefix="/partners", tags=["partners"])


class PartnerCreate(BaseModel):
    name: str


class PartnerRead(BaseModel):
    id: uuid.UUID
    name: str


@router.post("", status_code=status.HTTP_201_CREATED, response_model=PartnerRead)
async def create_partner(
    body: PartnerCreate,
    _: Principal = Depends(require_scopes("admin")),
    session: AsyncSession = Depends(get_async_session),
) -> PartnerRead:
    org = Org(name=body.name)
    session.add(org)
    await session.commit()
    await session.refresh(org)
    return PartnerRead(id=org.id, name=org.name)


@router.post(
    "/{org_id}/api-keys", status_code=status.HTTP_201_CREATED, response_model=ApiKeyCreated
)
async def mint_api_key(
    org_id: uuid.UUID,
    body: ApiKeyCreate,
    _: Principal = Depends(require_scopes("admin")),
    session: AsyncSession = Depends(get_async_session),
) -> ApiKeyCreated:
    org = await session.get(Org, org_id)
    if org is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Partner org not found")
    full, prefix, key_hash = generate_api_key()
    key = ApiKey(
        org_id=org_id, name=body.name, prefix=prefix, key_hash=key_hash,
        scopes=body.scopes, expires_at=body.expires_at,
    )
    session.add(key)
    await session.commit()
    await session.refresh(key)
    return ApiKeyCreated(
        id=key.id, name=key.name, prefix=key.prefix, scopes=key.scopes,
        active=key.active, expires_at=key.expires_at, created_at=key.created_at, api_key=full,
    )


@router.delete("/{org_id}/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    org_id: uuid.UUID,
    key_id: uuid.UUID,
    _: Principal = Depends(require_scopes("admin")),
    session: AsyncSession = Depends(get_async_session),
) -> None:
    key = await session.get(ApiKey, key_id)
    if key is None or key.org_id != org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="API key not found")
    key.active = False
    await session.commit()
