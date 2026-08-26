import asyncio
import pytest

import pipeline_audit_db


class FakeSession:
    def __init__(self):
        self.added = []
        self.committed = False

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.committed = True


class AsyncSessionCtx:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_async_log_rejection_uses_session(monkeypatch):
    fake = FakeSession()

    async def get_session_mock():
        return AsyncSessionCtx(fake)

    # Monkeypatch the imported get_async_session used in pipeline_audit_db
    monkeypatch.setattr(pipeline_audit_db, "get_async_session", get_session_mock)

    # Run the async writer
    asyncio.run(pipeline_audit_db.async_log_rejection("file.pdf", {"category": "Bilan"}, 0.7))

    # Verify that the session recorded an added AuditEvent and committed
    assert fake.added, "No object was added to the session"
    assert fake.committed is True
