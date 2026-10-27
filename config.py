"""Non-secret project defaults. Tune all dataset-dependent values for context."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_MB = 200
DEFAULT_TEST_SIZE = 0.25
DEFAULT_RANDOM_STATE = 42
SUPPORTED_CLASSIFIERS = ("random_forest", "logistic_regression", "decision_tree")
ALERT_STATUSES = ("New", "Under Investigation", "Benign", "Confirmed Suspicious")
RISK_LEVELS = ("Low", "Medium", "High", "Critical")

# Example-only configurable thresholds; deliberately not applied universally.
# Rules are data-dependent analyst signals, not evidence of a confirmed incident.
DEFAULT_RULES = []
