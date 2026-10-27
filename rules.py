"""Configurable, non-executable threshold rules for tabular records.

Rules are data, not Python expressions: only a small allow-list of comparison
operators is accepted. Rule matches are investigative signals, never proof of
intrusion. No feature names or domain thresholds are assumed by default.
"""
from __future__ import annotations

import operator
from typing import Any, Iterable, Mapping

import pandas as pd

OPS = {"gt": operator.gt, ">": operator.gt, "gte": operator.ge, ">=": operator.ge,
       "lt": operator.lt, "<": operator.lt, "lte": operator.le, "<=": operator.le,
       "eq": operator.eq, "==": operator.eq, "ne": operator.ne, "!=": operator.ne}


def _normalize_rule(rule: Mapping[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(rule, Mapping):
        raise TypeError(f"Rule at index {index} must be a mapping.")
    field = rule.get("field", rule.get("column"))
    op = str(rule.get("operator", rule.get("op", ""))).strip().lower()
    if not field:
        raise ValueError(f"Rule at index {index} is missing a field.")
    if op not in OPS:
        raise ValueError(f"Unsupported rule operator {op!r}; allowed: {', '.join(sorted(OPS))}.")
    if "value" not in rule:
        raise ValueError(f"Rule at index {index} is missing a comparison value.")
    severity = str(rule.get("severity", "medium")).lower()
    if severity not in {"low", "medium", "high", "critical"}:
        raise ValueError(f"Invalid severity {severity!r} in rule {index}.")
    return {"name": str(rule.get("name", f"Rule {index + 1}")), "field": str(field),
            "operator": op, "value": rule["value"], "severity": severity,
            "description": str(rule.get("description", ""))}


def evaluate_rules(records: pd.DataFrame | Mapping[str, Any], rules: Iterable[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    """Evaluate each rule against each row; returns one list of match objects per row.

    Missing columns and null values do not match. Invalid rule configuration is
    rejected up front, rather than silently ignored.
    """
    if isinstance(records, Mapping):
        frame = pd.DataFrame([dict(records)])
    elif isinstance(records, pd.DataFrame):
        frame = records
    else:
        raise TypeError("records must be a DataFrame or mapping")
    normalized = [_normalize_rule(r, i) for i, r in enumerate(rules)]
    output: list[list[dict[str, Any]]] = [[] for _ in range(len(frame))]
    for rule in normalized:
        if rule["field"] not in frame.columns:
            continue
        values = frame[rule["field"]]
        try:
            matched = OPS[rule["operator"]](values, rule["value"]).fillna(False)
            if not isinstance(matched, pd.Series):
                matched = pd.Series(bool(matched), index=frame.index)
        except (TypeError, ValueError):
            continue
        for pos, is_match in enumerate(matched.to_numpy(dtype=bool)):
            if is_match:
                output[pos].append({k: rule[k] for k in ("name", "field", "operator", "value", "severity", "description")})
    return output


class RuleEngine:
    """Small reusable wrapper around a caller-supplied rule list."""
    def __init__(self, rules: Iterable[Mapping[str, Any]] = ()):
        self.rules = [_normalize_rule(r, i) for i, r in enumerate(rules)]

    def evaluate(self, records: pd.DataFrame | Mapping[str, Any]) -> list[list[dict[str, Any]]]:
        return evaluate_rules(records, self.rules)
