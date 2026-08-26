import json
import asyncio
from typing import Optional

from api.db import get_async_session
from api.models import AuditEvent


async def async_log_rejection(
    filename: str,
    classification: dict,
    threshold: float,
    org_id: Optional[str] = None,
    actor_id: Optional[str] = None,
    actor_kind: str = "system",
    request_id: Optional[str] = None,
) -> None:
    """Persist an AuditEvent row asynchronously into the DB.

    This function is safe to call from async contexts (create_task) or
    from sync contexts via `asyncio.run` as a fallback.
    """
    # Support get_async_session being either an async context manager or a
    # coroutine that returns one (tests may patch it as a coroutine).
    session_cm = get_async_session()
    if asyncio.iscoroutine(session_cm):
        session_cm = await session_cm

    async with session_cm as session:
        event = AuditEvent(
            org_id=org_id,
            actor_id=actor_id,
            actor_kind=actor_kind,
            action="document_rejected",
            target_type="document",
            target_id=filename,
            detail={"classification": classification, "threshold": threshold},
            request_id=request_id,
        )
        session.add(event)
        await session.commit()
