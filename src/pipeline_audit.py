import json
from datetime import datetime
from pathlib import Path

AUDIT_PATH = Path("output") / "audit_events.jsonl"


def _ensure_parent():
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_rejection(filename: str, classification: dict, threshold: float) -> None:
    """Append a rejection audit event to a local JSONL file.

    Kept synchronous to be usable from the synchronous pipeline orchestrator.
    """
    _ensure_parent()
    event = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event": "document_rejected",
        "filename": filename,
        "classification": classification,
        "threshold": threshold,
    }
    with AUDIT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def read_audit_events(limit: int = 100) -> list:
    """Read recent audit events (for debug / UI)."""
    if not AUDIT_PATH.exists():
        return []
    events = []
    with AUDIT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events[-limit:]
