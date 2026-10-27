"""Lightweight, auditable explanations for common sklearn classifiers.

The helper exposes model-derived linear contribution proxies when available and
otherwise reports global tree feature importance. These are descriptive model
signals, not causal explanations or forensic proof.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def explain_prediction(pipeline, record, *, top_n: int = 5) -> dict:
    """Return feature-level explanation signals for one record.

    Linear coefficients are multiplied by the transformed feature value for a
    simple class-specific contribution proxy. Tree estimators expose global
    importance only, clearly labeled as such. Unsupported models return a
    useful limitation rather than fabricated explanations.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    frame = record.to_frame().T if isinstance(record, pd.Series) else pd.DataFrame([record]) if isinstance(record, dict) else record
    if not isinstance(frame, pd.DataFrame) or len(frame) != 1:
        raise ValueError("record must describe exactly one row")
    try:
        prep = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        transformed = prep.transform(frame)
        vals = np.asarray(transformed.toarray() if hasattr(transformed, "toarray") else transformed).reshape(-1)
        names = list(prep.get_feature_names_out())
    except (AttributeError, KeyError, ValueError, TypeError) as exc:
        return {"method": "unavailable", "features": [], "note": f"Could not inspect pipeline: {exc}"}
    if hasattr(classifier, "coef_"):
        coefs = np.asarray(classifier.coef_)
        contribution = coefs[0] * vals if coefs.ndim == 2 and len(coefs) else np.zeros_like(vals)
        order = np.argsort(np.abs(contribution))[::-1][:top_n]
        features = [{"feature": str(names[i]), "value": float(vals[i]),
                     "coefficient": float(coefs[0, i]), "contribution_proxy": float(contribution[i])}
                    for i in order if i < len(names)]
        return {"method": "linear_contribution_proxy", "features": features,
                "note": "Contribution proxy is model-specific, not causal or proof of malicious activity."}
    if hasattr(classifier, "feature_importances_"):
        importance = np.asarray(classifier.feature_importances_).reshape(-1)
        order = np.argsort(importance)[::-1][:top_n]
        features = [{"feature": str(names[i]), "global_importance": float(importance[i]),
                     "value": float(vals[i]) if i < len(vals) else None}
                    for i in order if i < len(names)]
        return {"method": "global_tree_feature_importance", "features": features,
                "note": "Global importance is not a per-record causal explanation or proof."}
    return {"method": "unsupported", "features": [],
            "note": f"{type(classifier).__name__} does not expose supported explanation attributes."}
