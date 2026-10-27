"""Tests use only small generated examples; they are not a real traffic dataset."""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_csv, validate_dataframe
from src.preprocessing import feature_frame, split_dataset
from src.rules import RuleEngine, evaluate_rules
from src.train_model import fit_model
from src.evaluation import evaluate_classification
from src.predict import predict_records
from src.explainability import explain_prediction
from src.alerts import create_alert, create_investigation_record, update_investigation
from src.reporting import export_alerts, export_csv, export_evaluation, export_json, export_predictions, save_local_model


@pytest.fixture
def synthetic_flows():
    """An intentionally tiny fabricated dataset for unit tests, not real telemetry."""
    rng = np.random.default_rng(23)
    n = 80
    burst = rng.integers(0, 2, n)
    return pd.DataFrame({
        "bytes": burst * 700 + rng.normal(100, 15, n),
        "protocol": np.where(np.arange(n) % 2, "tcp", "udp"),
        "attack_name": np.where(burst == 1, "fabricated_pattern", "normal"),
        "Label": np.where(burst == 1, "attack", "benign"),
    })


def test_csv_ingestion_normalizes_and_validates():
    frame = load_csv(io.StringIO("\ufeff bytes ,kind\n1,tcp\n2,udp\n"))
    assert list(frame.columns) == ["bytes", "kind"]
    assert len(frame) == 2
    with pytest.raises(ValueError, match="empty"):
        load_csv(io.StringIO(""))
    with pytest.raises(ValueError, match="unique"):
        validate_dataframe(pd.DataFrame([[1, 2]], columns=[" x", "x "]))
    with pytest.raises(ValueError, match="at least"):
        validate_dataframe(pd.DataFrame(columns=["x"]))


def test_split_excludes_obvious_label_name_and_returns_unfitted_transformer(synthetic_flows):
    x_train, x_test, y_train, y_test, prep = split_dataset(synthetic_flows, "Label", random_state=12)
    assert "attack_name" not in x_train and "Label" not in x_train
    assert len(x_train) + len(x_test) == len(synthetic_flows)
    assert not hasattr(prep, "transformers_")  # fitted only in estimator Pipeline
    x, y = feature_frame(synthetic_flows, "Label")
    assert len(y) == len(synthetic_flows) and list(x.columns) == ["bytes", "protocol"]
    with pytest.raises(ValueError, match="missing"):
        split_dataset(synthetic_flows.assign(Label=lambda d: d.Label.mask(d.index == 0)), "Label")
    inf_frame = synthetic_flows.drop(columns=["attack_name"]).copy()
    inf_frame.loc[0, "bytes"] = np.inf
    x_train, x_test, *_ = split_dataset(inf_frame, "Label", random_state=12)
    assert not np.isinf(pd.concat([x_train["bytes"], x_test["bytes"]]).dropna()).any()
    repeated = pd.DataFrame({"flow_key": np.repeat(np.arange(20), 2), "Label": np.repeat(np.arange(20) % 2, 2)})
    x_train, x_test, *_ = split_dataset(repeated, "Label", random_state=12)
    assert set(x_train["flow_key"]).isdisjoint(set(x_test["flow_key"]))


def test_rule_engine_is_configurable_and_non_executable():
    records = pd.DataFrame({"bytes": [10, 120, None], "protocol": ["udp", "tcp", "tcp"]})
    rules = [{"name": "large", "field": "bytes", "operator": ">=", "value": 100, "severity": "high"}]
    matches = RuleEngine(rules).evaluate(records)
    assert len(matches) == 3
    assert matches[1][0]["name"] == "large"
    assert matches[0] == [] and matches[2] == []
    assert evaluate_rules({"bytes": 120}, rules)[0]
    with pytest.raises(ValueError, match="operator"):
        RuleEngine([{"field": "bytes", "operator": "__import__", "value": 0}])


def test_training_prediction_and_explanation_are_reproducible(synthetic_flows):
    kwargs = {"model_name": "random_forest", "test_size": .25, "random_state": 42,
              "params": {"n_estimators": 12, "max_depth": 4, "n_jobs": 1}}
    result1 = fit_model(synthetic_flows, "Label", **kwargs)
    result2 = fit_model(synthetic_flows, "Label", **kwargs)
    assert set(("pipeline", "X_test", "y_test", "predictions", "metrics", "report", "config", "feature_names")) <= set(result1)
    assert np.array_equal(result1["predictions"], result2["predictions"])
    assert result1["config"]["random_state"] == 42
    assert "attack_name" not in result1["config"]["feature_columns"]
    out = predict_records(result1["pipeline"], result1["X_test"].head(3), positive_label="attack")
    assert {"prediction", "confidence", "risk_label", "risk_note"} <= set(out)
    assert out["risk_note"].str.contains("not proof").all()
    explanation = explain_prediction(result1["pipeline"], result1["X_test"].iloc[0])
    assert explanation["features"]
    assert "not" in explanation["note"]


def test_evaluation_binary_multiclass_and_metrics():
    binary = evaluate_classification([0, 0, 1, 1], [0, 1, 1, 1], [[.9, .1], [.7, .3], [.2, .8], [.1, .9]], labels=[0, 1])
    assert binary["confusion_matrix"] == [[1, 1], [0, 2]]
    assert binary["metrics"]["positive_class"] == 1
    assert binary["metrics"]["false_positive_rate"] == pytest.approx(.5)
    assert 0 <= binary["metrics"]["roc_auc"] <= 1
    assert "average_precision" in binary["metrics"]
    attack_first = evaluate_classification(
        ["attack", "attack", "benign", "benign"],
        ["attack", "benign", "attack", "benign"],
        [[.9, .1], [.6, .4], [.7, .3], [.1, .9]],
        labels=["attack", "benign"],
    )
    assert attack_first["metrics"]["positive_class"] == "attack"
    assert attack_first["metrics"]["false_positive_rate"] == pytest.approx(.5)
    multiclass = evaluate_classification(["a", "b", "c"], ["a", "b", "c"], labels=["a", "b", "c"])
    assert "roc_auc_ovr_macro" not in multiclass["metrics"]
    assert multiclass["metrics"]["accuracy"] == 1
    with pytest.raises(ValueError):
        evaluate_classification([], [])


def test_alert_investigation_and_exports(tmp_path, synthetic_flows):
    alert = create_alert({"flow": 7}, prediction="attack", confidence=.92,
                         risk_score=.92, risk_label="high", alert_id="test-alert")
    inv = create_investigation_record(alert, analyst="student", notes="Review context")
    updated = update_investigation(inv, status="reviewing", evidence={"source": "test fixture"})
    assert updated["status"] == "reviewing" and len(updated["evidence"]) == 1
    assert "not proof" in alert["assessment_note"]
    assert inv["status"] == "open"  # updates are immutable copies
    p1 = export_alerts([alert], tmp_path / "alerts.json")
    p2 = export_csv([updated], tmp_path / "investigations.csv")
    p3 = export_predictions(pd.DataFrame({"prediction": ["benign"]}), tmp_path / "predictions.json")
    p4 = export_evaluation({"metrics": {"accuracy": np.float64(.9)}}, tmp_path / "metrics.json")
    assert json.loads(p1.read_text())[0]["alert_id"] == "test-alert"
    assert "investigation_id" in p2.read_text().splitlines()[0]
    assert json.loads(p3.read_text())[0]["prediction"] == "benign"
    assert json.loads(p4.read_text())["metrics"]["accuracy"] == pytest.approx(.9)
    assert export_json({"ok": True}, tmp_path / "nested" / "x.json").exists()


def test_save_only_fitted_local_pipeline(synthetic_flows, tmp_path):
    result = fit_model(synthetic_flows, "Label", model_name="decision_tree", random_state=3)
    path = save_local_model(result["pipeline"], tmp_path / "local-model.joblib")
    assert path.is_file()
    with pytest.raises(TypeError):
        save_local_model(object(), tmp_path / "bad.joblib")
