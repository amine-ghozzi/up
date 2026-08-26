import os
import json
import asyncio
from pathlib import Path

import pytest

from pipeline import FinAlzePipeline


def test_classification_gate_rejects_and_writes_jsonl(tmp_path, monkeypatch):
    # Arrange: pipeline will see the doc as 'Autre' with low confidence
    p = FinAlzePipeline()

    def fake_classify(path):
        return {"category": "Autre", "confidence": 0.1, "reject_reason": "not_financial"}

    monkeypatch.setattr(p, "_classify_document_strict", lambda path: fake_classify(path))

    # Force DB persistence to fail so fallback to JSONL is used
    def raise_async(*args, **kwargs):
        raise RuntimeError("db unavailable")

    import pipeline_audit_db
    monkeypatch.setattr(pipeline_audit_db, "async_log_rejection", raise_async)

    # Redirect JSONL audit path to temp
    import pipeline_audit
    pipeline_audit.AUDIT_PATH = tmp_path / "audit_events.jsonl"

    # Create a dummy file for processing
    doc = tmp_path / "doc.pdf"
    doc.write_bytes(b"%PDF-1.4\n%EOF")

    # Act
    res = p.process_document(doc)

    # Assert: pipeline returned a rejected ExtractionResult
    assert res.metadata.get("pipeline_status") == "REJECTED"
    assert res.confidence_details.get("classification") is not None

    # Assert: fallback JSONL created and contains an event
    assert pipeline_audit.AUDIT_PATH.exists()
    data = pipeline_audit.AUDIT_PATH.read_text(encoding="utf-8").strip()
    assert data
    event = json.loads(data.splitlines()[-1])
    assert event.get("event") == "document_rejected"
