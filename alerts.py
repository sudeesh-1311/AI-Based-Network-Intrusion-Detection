"""Alert and investigation-record primitives for human review.

Every record describes a model/rule signal, not verified malicious activity.
Functions create plain dictionaries suitable for CSV/JSON export; they do not
send notifications, execute content, or infer an incident as proven.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any

VALID_RISK = {"low", "medium", "high", "critical", "unscored"}
VALID_STATUS = {"open", "reviewing", "resolved", "dismissed"}


def create_alert(record: dict[str, Any], *, prediction=None, confidence=None,
                 risk_score=None, risk_label="unscored", rule_matches=None,
                 source="model", alert_id=None, created_at=None) -> dict[str, Any]:
    """Create a normalized alert with an opaque ID, timestamp and review caveat."""
    if not isinstance(record, dict):
        raise TypeError("record must be a dictionary")
    risk = str(risk_label).lower()
    if risk not in VALID_RISK:
        raise ValueError(f"risk_label must be one of {sorted(VALID_RISK)}")
    return {"alert_id": str(alert_id or uuid4()),
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
            "source": str(source), "prediction": prediction,
            "confidence": _number_or_none(confidence), "risk_score": _number_or_none(risk_score),
            "risk_label": risk, "record": deepcopy(record),
            "rule_matches": deepcopy(rule_matches or []),
            "status": "open", "analyst_notes": "",
            "assessment_note": "Automated triage signal only; this alert is not proof of intrusion."}


def create_investigation_record(alert: dict[str, Any], *, analyst="", notes="", status="open") -> dict[str, Any]:
    """Create a separate human investigation record based on an alert."""
    if not isinstance(alert, dict) or not alert.get("alert_id"):
        raise ValueError("alert must be a mapping with an alert_id")
    status = str(status).lower()
    if status not in VALID_STATUS:
        raise ValueError(f"status must be one of {sorted(VALID_STATUS)}")
    return {"investigation_id": str(uuid4()), "alert_id": str(alert["alert_id"]),
            "created_at": datetime.now(timezone.utc).isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(),
            "analyst": str(analyst), "status": status, "notes": str(notes),
            "evidence": [], "alert_snapshot": deepcopy(alert),
            "assessment_note": "Human review record; automated signals do not establish an intrusion."}


def update_investigation(record: dict[str, Any], *, status=None, notes=None, evidence=None) -> dict[str, Any]:
    """Return an updated copy of an investigation record; evidence is plain text/data."""
    if not isinstance(record, dict) or not record.get("investigation_id"):
        raise ValueError("record must be an investigation record")
    out = deepcopy(record)
    if status is not None:
        status = str(status).lower()
        if status not in VALID_STATUS:
            raise ValueError(f"status must be one of {sorted(VALID_STATUS)}")
        out["status"] = status
    if notes is not None:
        out["notes"] = str(notes)
    if evidence is not None:
        if not isinstance(evidence, (list, tuple)):
            evidence = [evidence]
        out.setdefault("evidence", []).extend(deepcopy(list(evidence)))
    out["updated_at"] = datetime.now(timezone.utc).isoformat()
    return out


def _number_or_none(value):
    if value is None:
        return None
    try:
        result = float(value)
        return result if result == result and abs(result) != float("inf") else None
    except (TypeError, ValueError):
        return None


build_alert = create_alert
