"""End-to-end software smoke test using fabricated records only (not network telemetry)."""
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alerts import create_alert
from src.evaluation import evaluate_classification
from src.predict import predict_records
from src.reporting import export_alerts, export_evaluation, export_predictions
from src.rules import evaluate_rules
from src.train_model import fit_model


def main() -> None:
    rng = np.random.default_rng(42)
    n = 120
    synthetic_target = rng.integers(0, 2, size=n)
    frame = pd.DataFrame({
        "flow_duration_ms": np.maximum(0, rng.normal(900 + 1700 * synthetic_target, 300, size=n)),
        "bytes_per_flow": np.maximum(0, rng.normal(400 + 1100 * synthetic_target, 220, size=n)),
        "protocol": np.where(np.arange(n) % 3 == 0, "udp", "tcp"),
        "Label": np.where(synthetic_target == 1, "attack", "normal"),
    })
    print("SMOKE TEST DATA ONLY: generated synthetic rows, not real network telemetry.")

    trained = fit_model(
        frame, "Label", model_name="random_forest", test_size=0.25,
        random_state=42, params={"n_estimators": 30, "n_jobs": 1},
    )
    predictions = predict_records(trained["pipeline"], trained["X_test"], positive_label="attack")
    rule_config = [{
        "name": "synthetic high-byte example", "field": "bytes_per_flow",
        "operator": ">", "value": 1100, "severity": "medium",
        "description": "bytes_per_flow exceeds a demonstration-only value",
    }]
    matches = evaluate_rules(trained["X_test"], rule_config)
    predictions["rule_match"] = [bool(row_matches) for row_matches in matches]

    alerts = []
    for index, result_row in predictions.iterrows():
        if result_row["prediction"] == "attack" or matches[index]:
            alerts.append(create_alert(
                result_row.drop(labels=["risk_note"]).to_dict(),
                prediction=str(result_row["prediction"]),
                confidence=result_row["confidence"],
                risk_score=result_row["risk_score"],
                risk_label=str(result_row["risk_label"]),
                rule_matches=matches[index],
                source="model+rule" if matches[index] else "model",
            ))

    evaluation = evaluate_classification(
        trained["y_test"], trained["predictions"],
        probabilities=trained["pipeline"].predict_proba(trained["X_test"]),
        labels=trained["labels"],
    )
    with tempfile.TemporaryDirectory(prefix="nids-smoke-") as tmp:
        output = Path(tmp)
        export_predictions(predictions, output / "synthetic_predictions.csv")
        export_alerts(alerts, output / "synthetic_alerts.json")
        export_evaluation(evaluation, output / "synthetic_evaluation.json")
        assert (output / "synthetic_predictions.csv").is_file()
        assert (output / "synthetic_alerts.json").is_file()
        assert (output / "synthetic_evaluation.json").is_file()

    print(f"Synthetic rows: {len(frame)} | held-out rows: {len(trained['y_test'])}")
    print(f"Synthetic test accuracy: {trained['metrics']['accuracy']:.3f} (software smoke test only)")
    print(f"Potential alerts created: {len(alerts)} | CSV/JSON exports: OK")
    print("No model, dataset, or report artifact was retained.")


if __name__ == "__main__":
    main()
