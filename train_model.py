"""Reproducible classification training with train-only preprocessing.

``fit_model(train_df, target_col, model_name='random_forest', test_size=.25,
random_state=42, params=None)`` returns a dictionary containing the fitted
Pipeline, held-out ``X_test``/``y_test``, predictions, metrics, classification
report, config, and transformed feature names when supported. The preprocessor
is fitted only inside the Pipeline after splitting. No model files are loaded.
"""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .evaluation import evaluate_classification
from .preprocessing import feature_frame, split_dataset


def _estimator(model_name: str, random_state: int, params: dict[str, Any] | None):
    name = str(model_name).strip().lower().replace("-", "_").replace(" ", "_")
    choices = {
        "random_forest": RandomForestClassifier,
        "decision_tree": DecisionTreeClassifier,
        "extra_trees": ExtraTreesClassifier,
        "logistic_regression": LogisticRegression,
    }
    if name not in choices:
        raise ValueError(f"Unknown model_name {model_name!r}; choose one of {', '.join(choices)}.")
    kwargs = dict(params or {})
    if not isinstance(params, (dict, type(None))):
        raise TypeError("params must be a dictionary of estimator parameters.")
    kwargs.setdefault("random_state", random_state)
    if name == "random_forest":
        kwargs.setdefault("n_estimators", 100)
        kwargs.setdefault("n_jobs", -1)
    elif name == "extra_trees":
        kwargs.setdefault("n_estimators", 100)
        kwargs.setdefault("n_jobs", -1)
    elif name == "logistic_regression":
        kwargs.setdefault("max_iter", 1000)
    try:
        return name, choices[name](**kwargs)
    except TypeError as exc:
        raise ValueError(f"Invalid parameters for {name}: {exc}") from exc


def fit_model(
    train_df, target_col: str, model_name: str = "random_forest", test_size: float = 0.25,
    random_state: int = 42, params: dict[str, Any] | None = None, positive_label=None,
) -> dict[str, Any]:
    """Split, fit, and evaluate a classifier; return all reusable test artifacts."""
    normalized_name, classifier = _estimator(model_name, random_state, params)
    X_train, X_test, y_train, y_test, preprocessor = split_dataset(
        train_df, target_col, test_size=test_size, random_state=random_state,
        scale_numeric=normalized_name == "logistic_regression",
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(X_test)
        except (AttributeError, ValueError, NotImplementedError):
            probabilities = None
    evaluation = evaluate_classification(y_test, predictions, probabilities=probabilities,
                                         labels=getattr(pipeline, "classes_", None),
                                         positive_label=positive_label)
    try:
        feature_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    except (AttributeError, ValueError, TypeError):
        feature_names = None
    original_X, original_y = feature_frame(train_df, target_col)
    if original_X.duplicated(keep=False).any():
        split_strategy = "GroupShuffleSplit; identical feature rows kept together"
    elif original_y.value_counts().min() >= 2:
        split_strategy = "stratified random split"
    else:
        split_strategy = "random split; stratification not feasible"
    config = {"model_name": normalized_name, "target_col": target_col,
              "test_size": test_size, "random_state": random_state,
              "params": dict(params or {}), "positive_label": evaluation["metrics"].get("positive_class"),
              "split_strategy": split_strategy,
              "scale_numeric": normalized_name == "logistic_regression",
              "feature_columns": list(X_train.columns)}
    return {"pipeline": pipeline, "X_test": X_test, "y_test": y_test,
            "predictions": predictions, "metrics": evaluation["metrics"],
            "report": evaluation["classification_report"],
            "classification_report": evaluation["classification_report"],
            "confusion_matrix": evaluation["confusion_matrix"],
            "labels": evaluation["labels"], "config": config,
            "feature_names": feature_names}
