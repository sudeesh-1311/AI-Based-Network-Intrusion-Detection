"""Portable exports for predictions, alerts, investigations, and evaluations.

JSON/CSV exports contain ordinary tabular/report data. ``save_local_model`` is
only for a pipeline just trained locally with this package: it checks for the
expected sklearn Pipeline shape and never loads an uploaded artifact. Joblib
artifacts are executable serialization formats and must remain trusted/local;
there is intentionally no model-loading helper here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def export_json(data: Any, path: str | Path) -> Path:
    """Write JSON to a caller-chosen local path and return the path."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default, allow_nan=False), encoding="utf-8")
    return destination


def _records_frame(records) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, dict):
        records = [records]
    frame = pd.DataFrame(list(records))
    for col in frame.columns:
        if frame[col].map(lambda v: isinstance(v, (dict, list, tuple))).any():
            frame[col] = frame[col].map(lambda v: json.dumps(v, ensure_ascii=False, default=_json_default) if isinstance(v, (dict, list, tuple)) else v)
    return frame


def export_csv(records, path: str | Path) -> Path:
    """Export dataframe or list-of-dict records as UTF-8 CSV."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _records_frame(records).to_csv(destination, index=False, encoding="utf-8")
    return destination


def export_predictions(predictions, path: str | Path) -> Path:
    """Export predictions as CSV when extension is .csv, otherwise JSON."""
    if Path(path).suffix.lower() == ".csv":
        return export_csv(predictions, path)
    if isinstance(predictions, pd.DataFrame):
        predictions = predictions.to_dict(orient="records")
    return export_json(predictions, path)


def export_alerts(alerts, path: str | Path) -> Path:
    """Export alert records as CSV or JSON according to the path suffix."""
    return export_csv(alerts, path) if Path(path).suffix.lower() == ".csv" else export_json(alerts, path)


def export_investigations(records, path: str | Path) -> Path:
    """Export investigation records as CSV or JSON according to suffix."""
    return export_csv(records, path) if Path(path).suffix.lower() == ".csv" else export_json(records, path)


def export_evaluation(evaluation: dict[str, Any], path: str | Path) -> Path:
    """Export metrics/report/matrix in JSON format."""
    return export_json(evaluation, path)


def save_local_model(pipeline, path: str | Path) -> Path:
    """Persist a locally trained sklearn preprocessing+classifier Pipeline.

    Only use with a pipeline produced by ``src.train_model.fit_model`` in the
    current trusted application. Loading joblib files from users is unsafe;
    this project intentionally offers no artifact deserialization function.
    """
    from sklearn.pipeline import Pipeline
    if not isinstance(pipeline, Pipeline) or not {"preprocessor", "classifier"}.issubset(pipeline.named_steps):
        raise TypeError("Expected a locally trained Pipeline with preprocessor and classifier steps.")
    if not hasattr(pipeline.named_steps["classifier"], "classes_"):
        raise ValueError("Pipeline classifier is not fitted; train it before saving.")
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to save a model (normally installed with scikit-learn).") from exc
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination)
    return destination
