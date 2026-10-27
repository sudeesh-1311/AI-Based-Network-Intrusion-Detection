"""AI-Based Network Intrusion Detection dashboard.

Run from the project root with: streamlit run app.py
All uploads and investigation state are kept in the current Streamlit session unless
an installed backend explicitly provides persistence. This app does not capture live
network traffic or automatically block hosts/connections.
"""
from __future__ import annotations

import importlib
import io
import json
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import streamlit as st


APP_ROOT = __file__.rsplit("/", 1)[0]
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)
from config import MAX_UPLOAD_MB

# Optional backend APIs. Import failures are handled at the point of use so the
# application can start before the backend modules have been completed.
def _backend(module: str, function: str):
    try:
        return getattr(importlib.import_module(f"src.{module}"), function)
    except Exception:
        try:
            return getattr(importlib.import_module(module), function)
        except Exception:
            return None


validate_dataframe = _backend("data_loader", "validate_dataframe")
load_csv_bytes = _backend("data_loader", "load_csv_bytes")
load_csv_api = _backend("data_loader", "load_csv")
fit_model_api = _backend("train_model", "fit_model")
evaluate_rules_api = _backend("rules", "evaluate_rules")
generate_detections_api = _backend("predict", "generate_detections")
explain_prediction_api = _backend("explainability", "explain_prediction")
# These are intentionally optional because the backend workflow/report interfaces
# may evolve independently of the dashboard.
alert_backend = {name: _backend("alerts", name) for name in ("create_alert", "update_alert", "list_alerts", "add_note")}
report_backend = {name: _backend("reporting", name) for name in ("generate_report", "export_report")}

st.set_page_config(page_title="Sentinel | Network IDS", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { --navy:#071426; --panel:#0d1d33; --panel2:#112540; --line:#233a56; --muted:#a7bbd1; --cyan:#55d6d0; --text:#edf4fb; }
.stApp { background:linear-gradient(150deg,#071426 0%,#0a192c 48%,#0b1a2e 100%); color:var(--text); }
[data-testid="stSidebar"] { background:#071426; border-right:1px solid #20354f; }
[data-testid="stSidebar"] * { color:#e8f1fa; }
[data-testid="stFileUploaderDropzone"] { background:#10233b !important; border:1px dashed #41627e !important; }
[data-testid="stFileUploaderDropzone"] * { color:#e8f1fa !important; }
[data-testid="stFileUploaderDropzone"] button { background:#17334d !important; border-color:#41627e !important; }
.block-container { padding-top:1.5rem; padding-bottom:3rem; max-width:1500px; }
h1,h2,h3 { color:#f4f8fc; letter-spacing:-.02em; }
p, label, [data-testid="stMarkdownContainer"] { color:#d6e2ef; }
[data-testid="stMetric"] { background:linear-gradient(135deg,#112540,#0e2037); border:1px solid #233d5b; padding:15px 17px; border-radius:13px; }
[data-testid="stMetricLabel"] { color:#b5c8db !important; }
[data-testid="stMetricValue"] { color:#f4f8fc !important; }
.stButton>button, .stDownloadButton>button { border-radius:8px; border:1px solid #3e7184; background:#12354b; color:#f3fbff; font-weight:600; }
.stButton>button:hover, .stDownloadButton>button:hover { border-color:#55d6d0; background:#17475a; color:white; }
[data-testid="stDataFrame"] { border:1px solid #203852; border-radius:10px; overflow:hidden; }
[data-baseweb="select"] > div, [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-testid="stTextArea"] textarea { background:#10233b; color:#f4f8fc; border-color:#29435f; }
hr { border-color:#263c56; }
.small-muted { color:#a9bdd2; font-size:.9rem; }
.status-pill { display:inline-block; background:#12384a; color:#75e6dc; border:1px solid #286070; border-radius:99px; padding:3px 10px; font-size:.78rem; }
</style>
""", unsafe_allow_html=True)

PAGES = ["Overview", "Dataset Explorer", "Preprocessing", "Model Training", "Model Evaluation", "Intrusion Detection", "Alert Investigation", "Reports & Export", "About"]


def _init_state():
    defaults = {
        "page": "Overview", "df": None, "filename": None, "target": None,
        "dataset_source": "User-supplied CSV (source not independently verified)", "dataset_license": "", "dataset_description": "",
        "model": None, "model_name": None, "metrics": None, "class_names": None,
        "test_truth": None, "test_pred": None, "X_test": None, "training_info": None,
        "detections": None, "alerts": [], "rules": [], "comparison_results": None, "upload_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


def _safe_load_csv(payload: bytes) -> pd.DataFrame:
    if load_csv_bytes:
        try:
            loaded = load_csv_bytes(payload)
            if isinstance(loaded, pd.DataFrame):
                return loaded
            if isinstance(loaded, tuple) and loaded and isinstance(loaded[0], pd.DataFrame):
                return loaded[0]
            if isinstance(loaded, dict):
                for key in ("dataframe", "df", "data"):
                    if isinstance(loaded.get(key), pd.DataFrame):
                        return loaded[key]
        except Exception:
            pass
    frame = load_csv_api(io.BytesIO(payload)) if load_csv_api else pd.read_csv(io.BytesIO(payload))
    st.session_state.infinite_value_count = int(np.isinf(frame.select_dtypes(include=["number"]).to_numpy()).sum()) if not frame.select_dtypes(include=["number"]).empty else 0
    cleaned = validate_dataframe(frame) if validate_dataframe else frame
    return cleaned if isinstance(cleaned, pd.DataFrame) else frame


def _validate(df: pd.DataFrame) -> tuple[bool, str]:
    if df.empty:
        return False, "The CSV contains no data rows."
    if df.shape[1] < 2:
        return False, "A supervised dataset must contain at least two columns: features and a target."
    if any(str(c).strip() == "" for c in df.columns):
        return False, "One or more column names are blank. Rename blank columns and upload again."
    if df.columns.duplicated().any():
        return False, "Duplicate column names were found. Rename them so every column is unique."
    if validate_dataframe:
        try:
            result = validate_dataframe(df)
            if isinstance(result, tuple):
                if len(result) >= 2 and not bool(result[0]):
                    return False, str(result[1])
                if len(result) == 1 and result[0] is False:
                    return False, "Backend validation rejected the dataset."
            elif isinstance(result, dict):
                if result.get("valid") is False or result.get("ok") is False:
                    return False, str(result.get("errors") or result.get("message") or "Backend validation rejected the dataset.")
            elif isinstance(result, pd.DataFrame):
                pass
            elif result is False:
                return False, "Backend validation rejected the dataset."
        except Exception as exc:
            return False, f"Dataset validation failed: {exc}"
    return True, "Dataset structure looks usable. Review data quality before training."


def _file_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _format_size(n: int) -> str:
    return f"{n / 1024:.1f} KB" if n < 1024 * 1024 else f"{n / (1024 * 1024):.2f} MB"


def _set_page(page: str):
    st.session_state.page = page


def _page_header(title: str, subtitle: str):
    st.title(title)
    st.caption(subtitle)
    st.markdown("<hr>", unsafe_allow_html=True)


def _train_local(data: pd.DataFrame, target: str, model_name: str, params=None, positive_label=None):
    """Use the backend pipeline so label-derived fields stay out of training."""
    if not fit_model_api:
        raise RuntimeError("The project training backend is unavailable. Install requirements and run from the project root.")
    model_keys = {
        "Random Forest": "random_forest",
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "Extra Trees": "extra_trees",
    }
    result = fit_model_api(data, target, model_name=model_keys[model_name], test_size=0.25, random_state=42, params=params or {}, positive_label=positive_label)
    raw_metrics = dict(result["metrics"])
    y_test = pd.Series(result["y_test"]).astype(str).tolist()
    y_pred = pd.Series(result["predictions"]).astype(str).tolist()
    metrics = {
        **raw_metrics,
        "confusion_matrix": result["confusion_matrix"],
        "labels": [str(x) for x in result["labels"]],
        "report": result["classification_report"],
        "test_size": len(result["y_test"]),
        "train_size": len(data) - len(result["y_test"]),
        "positive_class": str(raw_metrics["positive_class"]) if "positive_class" in raw_metrics else None,
        "config": result["config"],
        "feature_columns": result["config"]["feature_columns"],
    }
    return result["pipeline"], metrics, y_test, y_pred, result["X_test"]


def _make_predictions(data: pd.DataFrame) -> pd.DataFrame:
    model = st.session_state.model
    if model is None:
        raise ValueError("Train a model before generating detections.")
    target = st.session_state.target
    X = data.drop(columns=[target], errors="ignore")
    if hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=list(model.feature_names_in_), fill_value=pd.NA)
    pred = model.predict(X)
    out = data.copy().reset_index(drop=True)
    out["Predicted class"] = pd.Series(pred).astype(str)
    out["Model score (uncalibrated)"] = pd.NA
    score_label = "Model score (uncalibrated)"
    try:
        scores = model.predict_proba(X)
        classes = [str(c) for c in model.classes_]
        ix = [classes.index(str(p)) for p in pred]
        out[score_label] = [float(scores[row, col]) for row, col in enumerate(ix)]
    except Exception:
        try:
            raw = model.decision_function(X)
            if getattr(raw, "ndim", 1) == 1:
                out[score_label] = pd.Series(raw).astype(float)
            else:
                out[score_label] = [float(raw[i].max()) for i in range(len(raw))]
            score_label = "Decision score (uncalibrated)"
            out.rename(columns={"Model score (uncalibrated)": score_label}, inplace=True)
        except Exception:
            pass
    benign = st.session_state.get("benign_class")
    if benign is None or benign == "(none — flag all predictions)":
        out["Flagged"] = True
    else:
        out["Flagged"] = out["Predicted class"].astype(str) != str(benign)
    out["Detection source"] = "Model prediction"
    return out


def _build_alerts(detections: pd.DataFrame):
    flagged = detections[detections["Flagged"]].copy()
    alerts = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for i, (_, row) in enumerate(flagged.iterrows()):
        alert = {
            "Alert ID": f"A-{int(time.time())}-{i+1:04d}", "Created (UTC)": now,
            "Predicted class": str(row.get("Predicted class", "Unknown")),
            "Score (uncalibrated)": row.get("Model score (uncalibrated)", row.get("Decision score (uncalibrated)", None)),
            "Risk level": str(row.get("Risk level", "Unscored")),
            "Severity reason": str(row.get("Severity reason", "")),
            "Event timestamp": row.get("Event timestamp", None),
            "Triggered rules": row.get("Triggered rules", ""),
            "Source": str(row.get("Detection source", "Model prediction")),
            "Status": "New", "Analyst notes": "", "Status history": [],
        }
        # Preserve original fields for investigation/export; cap added columns only in display.
        for col, value in row.items():
            if col not in alert and col not in ("Flagged",):
                alert[str(col)] = value
        if explain_prediction_api and st.session_state.model is not None:
            feature_record = row.drop(labels=[st.session_state.target, "Predicted class", "Model score (uncalibrated)", "Decision score (uncalibrated)", "Flagged", "Detection source", "Triggered rules", "Rule alert", "Risk level", "Severity reason", "Event timestamp", "Analyst review status"], errors="ignore").to_dict()
            try:
                explanation = explain_prediction_api(st.session_state.model, feature_record, top_n=5)
                alert["Feature evidence"] = explanation.get("features", [])
                alert["Explanation note"] = explanation.get("note", "Feature evidence is descriptive model output, not forensic proof.")
                alert["Explanation method"] = explanation.get("method", "unavailable")
            except Exception as exc:
                alert["Feature evidence"] = []
                alert["Explanation note"] = f"Feature evidence unavailable: {exc}"
        alerts.append(alert)
    return alerts


with st.sidebar:
    st.markdown("## 🛡️ Sentinel IDS")
    st.caption("AI-based network intrusion analysis")
    selected_page = st.radio("Navigation", PAGES, index=0, key="navigation", label_visibility="collapsed")
    st.session_state.page = selected_page
    st.markdown("---")
    uploaded = st.file_uploader("Load a dataset", type=["csv"], help=f"CSV only, up to {MAX_UPLOAD_MB} MB. Data is processed in this Streamlit session.")
    if uploaded is not None:
        stamp = (uploaded.name, uploaded.size)
        if st.session_state.get("upload_stamp") != stamp:
            try:
                if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
                    raise ValueError(f"Upload exceeds the {MAX_UPLOAD_MB} MB project limit. Split or sample the CSV before upload.")
                frame = _safe_load_csv(uploaded.getvalue())
                ok, message = _validate(frame)
                if not ok:
                    st.session_state.upload_error = message
                else:
                    st.session_state.df = frame
                    st.session_state.filename = uploaded.name
                    st.session_state.upload_stamp = stamp
                    st.session_state.target = None
                    st.session_state.model = None
                    st.session_state.metrics = None
                    st.session_state.X_test = None
                    st.session_state.test_truth = None
                    st.session_state.test_pred = None
                    st.session_state.detections = None
                    st.session_state.alerts = []
                    st.session_state.comparison_results = None
                    st.session_state.benign_class = "(none — flag all predictions)"
                    st.session_state.upload_error = None
                    if message.startswith("Basic checks"):
                        st.session_state.validation_notice = message
            except Exception as exc:
                st.session_state.upload_error = f"Could not read this CSV: {exc}"
        if st.session_state.upload_error:
            st.error(st.session_state.upload_error)
    if st.session_state.df is not None:
        st.markdown(f"<span class='status-pill'>Dataset loaded</span> · {st.session_state.filename}", unsafe_allow_html=True)
        st.caption(f"{len(st.session_state.df):,} rows · {st.session_state.df.shape[1]} columns")
        if st.button("Clear dataset and session results", use_container_width=True):
            for key in ("df", "filename", "target", "model", "metrics", "class_names", "test_truth", "test_pred", "X_test", "training_info", "detections", "comparison_results", "alerts", "rules", "upload_stamp", "upload_error"):
                st.session_state[key] = None if key not in ("alerts", "rules") else []
            st.rerun()
    st.markdown("---")
    st.caption("Analysis only · no live capture or automatic blocking")

page = st.session_state.page
df = st.session_state.df
if st.session_state.get("validation_notice"):
    st.info(st.session_state.validation_notice)
    st.session_state.validation_notice = None

if page == "Overview":
    _page_header("Network Intrusion Detection", "A session-based workspace for dataset review, supervised model training, and alert investigation.")
    detections = st.session_state.detections
    if detections is not None:
        a, b, c, d = st.columns(4)
        a.metric("Analyzed records", f"{len(detections):,}")
        b.metric("Normal / not flagged", f"{int((~detections['Flagged']).sum()):,}")
        c.metric("Suspicious signals", f"{int(detections['Flagged'].sum()):,}")
        d.metric("Investigation records", f"{len(st.session_state.alerts):,}")
        risk_counts = detections.loc[detections["Flagged"], "Risk level"].value_counts().reindex(["Low", "Medium", "High", "Critical"], fill_value=0)
        left, right = st.columns([1, 2])
        with left:
            st.markdown("### Flagged signals by risk")
            st.bar_chart(risk_counts)
        with right:
            time_col = "Event timestamp"
            if time_col in detections and detections[time_col].notna().any():
                event_times = pd.to_datetime(detections[time_col], errors="coerce", utc=True)
                trend = detections.assign(_event_date=event_times).dropna(subset=["_event_date"])
                if not trend.empty:
                    trend = trend[trend["Flagged"]].assign(_day=lambda x: x["_event_date"].dt.floor("D")).groupby("_day").size().rename("Potential signals")
                    st.markdown("### Detection trend · event time")
                    if not trend.empty:
                        st.line_chart(trend)
            if st.session_state.alerts:
                st.markdown("### Recent alerts")
                st.dataframe(pd.DataFrame(st.session_state.alerts).tail(5), use_container_width=True, hide_index=True)
        if st.session_state.alerts:
            st.markdown("### Investigation status summary")
            st.bar_chart(pd.DataFrame(st.session_state.alerts)["Status"].value_counts())
    else:
        a, b, c, d = st.columns(4)
        a.metric("Dataset", "Loaded" if df is not None else "Not loaded")
        b.metric("Rows", f"{len(df):,}" if df is not None else "—")
        c.metric("Features", str(max(0, df.shape[1] - 1)) if df is not None and st.session_state.target else (str(df.shape[1]) if df is not None else "—"))
        d.metric("Potential alerts", str(len(st.session_state.alerts)))
    st.markdown("### Start here")
    if df is None:
        st.info("Upload a labeled CSV in the sidebar to begin. No dataset is bundled; the dashboard starts safely in an empty state.")
        st.markdown("A typical dataset contains one row per observation, feature columns, and a label column (for example, `normal` / `attack`). Avoid uploading sensitive production data unless your environment is approved for it.")
    else:
        st.success(f"**{st.session_state.filename}** is ready for review. Select a target label in Dataset Explorer or Model Training before fitting.")
        x, y, z = st.columns(3)
        x.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        y.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
        z.metric("Target", st.session_state.target or "Not selected")
        st.dataframe(df.head(8), use_container_width=True, hide_index=True)
    st.markdown("### Workflow")
    st.markdown("1. **Explore** data quality and choose the label column.  2. **Preprocess** by reviewing missingness and feature types.  3. **Train** a model and inspect held-out metrics.  4. **Detect** on uploaded rows and investigate generated alerts.")
    st.warning("This is an offline analysis dashboard. It does not sniff interfaces, receive live packets, or take blocking/remediation actions. Scores are model outputs and may be uncalibrated.")

elif page == "Dataset Explorer":
    _page_header("Dataset Explorer", "Review schema, label distribution, and sample records before model training.")
    if df is None:
        st.info("Upload a CSV using the sidebar to explore a dataset.")
    else:
        st.markdown("### Dataset provenance (user-declared)")
        st.session_state.dataset_source = st.text_input("Source / publisher / URL", value=st.session_state.dataset_source or "")
        st.session_state.dataset_license = st.text_input("License or permission notes", value=st.session_state.dataset_license or "")
        st.session_state.dataset_description = st.text_area("Dataset description", value=st.session_state.dataset_description or "", height=75)
        st.caption("Provenance and licensing details are supplied by the analyst and are not independently verified by this application.")
        target = st.selectbox("Target / label column", options=["(select target)"] + list(df.columns), index=(list(df.columns).index(st.session_state.target) + 1 if st.session_state.target in df.columns else 0), help="The selected column is excluded from model features.")
        if target != "(select target)":
            st.session_state.target = target
        else:
            st.session_state.target = None
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{len(df):,}"); m2.metric("Columns", str(df.shape[1])); m3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}"); m4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
        st.markdown("### Schema & quality")
        schema = pd.DataFrame({"Column": [str(c) for c in df.columns], "Type": [str(t) for t in df.dtypes], "Non-null": [int(df[c].notna().sum()) for c in df.columns], "Missing": [int(df[c].isna().sum()) for c in df.columns], "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns]})
        st.dataframe(schema, use_container_width=True, hide_index=True)
        if st.session_state.target:
            with st.expander(f"Label distribution · {st.session_state.target}", expanded=True):
                dist = df[st.session_state.target].astype("string").fillna("(missing)").value_counts().rename_axis("Label").reset_index(name="Rows")
                st.dataframe(dist, use_container_width=True, hide_index=True)
                st.bar_chart(dist.set_index("Label"))
        st.markdown("### Sample rows")
        row_limit = st.slider("Rows to preview", 5, min(100, max(5, len(df))), min(20, max(5, len(df))))
        st.dataframe(df.head(row_limit), use_container_width=True, hide_index=True)
        st.download_button("Download uploaded data as CSV", data=_file_bytes(df), file_name="dataset_review.csv", mime="text/csv")

elif page == "Preprocessing":
    _page_header("Preprocessing Review", "Transparent inspection of training inputs; source data is not silently changed.")
    if df is None:
        st.info("Upload a CSV to inspect preprocessing needs.")
    else:
        target = st.session_state.target
        if not target:
            target = st.selectbox("Select target column", ["(select target)"] + list(df.columns))
            if target != "(select target)": st.session_state.target = target
            else: target = None
        numeric = list(df.select_dtypes(include=["number", "bool"]).columns)
        categorical = [c for c in df.columns if c not in numeric]
        missing = df.isna().sum()
        x, y, z = st.columns(3)
        x.metric("Numeric columns", len(numeric)); y.metric("Categorical columns", len(categorical)); z.metric("Rows with missing cells", f"{int(df.isna().any(axis=1).sum()):,}")
        st.markdown("### Missing values")
        missing_summary = pd.DataFrame({"Column": missing.index.astype(str), "Missing": missing.values, "Missing %": (missing.values / max(1, len(df)) * 100).round(2)})
        missing_summary = missing_summary[missing_summary["Missing"] > 0].sort_values("Missing", ascending=False)
        if missing_summary.empty: st.success("No missing values detected.")
        else: st.dataframe(missing_summary, use_container_width=True, hide_index=True)
        st.markdown("### Training behavior")
        st.markdown("The training pipeline imputes numeric fields with the median and categorical fields with the most frequent value, then one-hot encodes categories. Numeric scaling is enabled for Logistic Regression; tree models use imputation without unnecessary scaling. Every fitted transformation is learned on the training split only. Resolve missing target labels before training; the backend rejects a target with missing values rather than silently dropping labeled examples.")
        infinite_count = int(st.session_state.get("infinite_value_count", 0))
        st.write(f"Infinite numeric values found at upload: **{infinite_count:,}**. They are converted to missing values for safe downstream preprocessing.")
        if target:
            label_counts = df[target].dropna().value_counts()
            if not label_counts.empty:
                ratio = float(label_counts.max() / max(1, label_counts.min()))
                st.markdown("### Class balance")
                st.dataframe(label_counts.rename_axis("Class").reset_index(name="Rows"), use_container_width=True, hide_index=True)
                st.caption(f"Largest-to-smallest observed class ratio: {ratio:.2f}:1 · labels with missing values: {int(df[target].isna().sum()):,}.")
                if ratio >= 5:
                    st.warning("Strong class imbalance detected. Inspect per-class recall/precision and the false-positive/false-negative trade-off; accuracy alone can be misleading.")
        st.caption(f"Target excluded from feature inputs: **{target or 'not selected'}**. No dataset edits are applied on this page.")
        st.warning("Review for leakage, identifiers, timestamps, label imbalance, and fields unavailable at inference time before training.")

elif page == "Model Training":
    _page_header("Model Training", "Fit a supervised classifier and calculate metrics on a held-out test split.")
    if df is None:
        st.info("Upload a labeled CSV to configure model training.")
    else:
        target = st.selectbox("Target / label column", options=["(select target)"] + list(df.columns), index=(list(df.columns).index(st.session_state.target) + 1 if st.session_state.target in df.columns else 0))
        if target == "(select target)": target = None
        else: st.session_state.target = target
        model_name = st.selectbox("Classifier", ["Random Forest", "Logistic Regression", "Decision Tree", "Extra Trees"], help="Models are trained locally on the uploaded dataset.")
        st.caption("Reproducible 25% hold-out target (random seed 42). Stratification is used when feasible; identical feature rows are grouped to avoid crossing the split, so the realized row fraction can vary. Metrics use only held-out records.")
        model_params = {}
        with st.expander("Model parameters", expanded=False):
            if model_name in {"Random Forest", "Extra Trees"}:
                model_params["n_estimators"] = st.slider("Number of trees", 50, 400, 100, 10)
                max_depth_ui = st.number_input("Maximum tree depth (0 = unlimited)", min_value=0, max_value=100, value=0)
                if max_depth_ui:
                    model_params["max_depth"] = int(max_depth_ui)
            elif model_name == "Decision Tree":
                max_depth_ui = st.number_input("Maximum tree depth (0 = unlimited)", min_value=0, max_value=100, value=12)
                if max_depth_ui:
                    model_params["max_depth"] = int(max_depth_ui)
                model_params["min_samples_leaf"] = st.slider("Minimum samples per leaf", 1, 20, 1)
            else:
                model_params["C"] = st.number_input("Regularization strength C", min_value=0.001, max_value=100.0, value=1.0, step=0.1, format="%.3f")
            st.caption("Other parameters remain at scikit-learn defaults. Random state is fixed at 42 for reproducibility.")
        if target:
            counts = df[target].dropna().astype(str).value_counts()
            st.write(f"**Labeled rows:** {int(df[target].notna().sum()):,} · **Classes:** {len(counts)}")
            if len(counts): st.dataframe(counts.rename_axis("Class").reset_index(name="Rows"), use_container_width=True, hide_index=True)
        positive_class_choice = None
        if target and df[target].dropna().nunique() == 2:
            raw_classes = sorted(df[target].dropna().unique().tolist(), key=lambda value: str(value))
            candidates = [value for value in raw_classes if any(word in str(value).lower() for word in ("attack", "malicious", "intrusion", "anomaly", "botnet", "dos", "ddos", "exploit"))]
            preferred_positive = candidates[0] if len(candidates) == 1 else raw_classes[-1]
            default_index = raw_classes.index(preferred_positive)
            positive_class_choice = st.selectbox("Positive class for binary metrics", options=raw_classes, index=default_index, format_func=str, help="False-positive rate, ROC-AUC, and precision-recall analysis use this class. The selection is recorded in the training configuration.")
        train_col, compare_col = st.columns(2)
        train_clicked = train_col.button("Train selected model", type="primary", disabled=not bool(target), use_container_width=True)
        compare_clicked = compare_col.button("Compare all models", disabled=not bool(target), use_container_width=True)
        if train_clicked:
            with st.spinner("Preparing features, fitting classifier, and evaluating held-out rows…"):
                try:
                    model, metrics, y_true, y_pred, X_test = _train_local(df, target, model_name, model_params, positive_class_choice)
                    st.session_state.model = model
                    st.session_state.model_name = model_name
                    st.session_state.target = target
                    st.session_state.metrics = metrics
                    st.session_state.class_names = metrics["labels"]
                    benign_candidates = [label for label in metrics["labels"] if str(label).strip().lower() in {"normal", "benign", "benign traffic", "normal traffic"}]
                    st.session_state.benign_class = benign_candidates[0] if benign_candidates else "(none — flag all predictions)"
                    st.session_state.test_truth = y_true
                    st.session_state.test_pred = y_pred
                    st.session_state.X_test = X_test
                    st.session_state.training_info = {"backend": "src.train_model.fit_model · train-only preprocessing · fixed seed 42", "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "dataset": st.session_state.filename}
                    st.session_state.detections = None
                    st.success("Training completed. Review the held-out evaluation before relying on detections.")
                except Exception as exc:
                    st.error(f"Training could not be completed: {exc}")
        if compare_clicked:
            model_ids = [("Random Forest", "random_forest"), ("Logistic Regression", "logistic_regression"), ("Decision Tree", "decision_tree"), ("Extra Trees", "extra_trees")]
            comparisons = []
            with st.spinner("Training classifiers on the same reproducible split…"):
                try:
                    for label, model_id in model_ids:
                        result = fit_model_api(df, target, model_name=model_id, test_size=0.25, random_state=42, positive_label=positive_class_choice)
                        m = result["metrics"]
                        comparisons.append({"Model": label, "Accuracy": m["accuracy"], "Precision · macro": m["precision_macro"], "Recall · macro": m["recall_macro"], "F1 · macro": m["f1_macro"], "ROC-AUC": m.get("roc_auc", m.get("roc_auc_ovr_macro"))})
                    st.session_state.comparison_results = comparisons
                except Exception as exc:
                    st.error(f"Model comparison could not be completed: {exc}")
        if st.session_state.comparison_results:
            st.markdown("### Classifier comparison · shared held-out split")
            st.dataframe(pd.DataFrame(st.session_state.comparison_results).style.format(precision=3), use_container_width=True, hide_index=True)
            st.caption("Each classifier used the same target, random seed, and duplicate-aware split policy. Choose based on class-specific errors and operational costs, not a single aggregate metric.")
        if st.session_state.metrics is not None:
            st.markdown("### Current trained model")
            st.write(f"**{st.session_state.model_name}** · target **{st.session_state.target}** · {st.session_state.training_info.get('timestamp', '')}")
            st.caption(st.session_state.training_info.get("backend", ""))
            with st.expander("Training configuration and selected features"):
                st.json(st.session_state.metrics.get("config", {}), expanded=False)
            st.info("A model is kept in this Streamlit session only. Reloading/ending the session may clear it unless your deployment persists session state.")

elif page == "Model Evaluation":
    _page_header("Model Evaluation", "Inspect genuine test-split metrics. No default, placeholder, or fabricated scores are shown.")
    metrics = st.session_state.metrics
    if metrics is None:
        st.info("Train a model first. This page will remain empty until real evaluation metrics are available.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{metrics['accuracy']:.3f}"); b.metric("Precision · macro", f"{metrics['precision_macro']:.3f}"); c.metric("Recall · macro", f"{metrics['recall_macro']:.3f}"); d.metric("F1 · macro", f"{metrics['f1_macro']:.3f}")
        st.caption(f"Computed on {metrics['test_size']:,} held-out rows; trained on {metrics['train_size']:,}. Model: {st.session_state.model_name}.")
        if len(metrics["labels"]) == 2:
            fpr = metrics.get("false_positive_rate")
            st.metric(f"False-positive rate · positive class {metrics['positive_class']}", f"{fpr:.3f}" if fpr is not None else "Undefined (no negatives in test set)")
        else:
            st.caption("Multiclass task: false-positive rate depends on a one-vs-rest class choice; inspect per-class results below.")
        if metrics.get("roc_auc") is not None:
            st.caption(f"ROC-AUC for positive class **{metrics['positive_class']}**: {metrics['roc_auc']:.3f}. ROC-AUC uses uncalibrated estimator scores; it is not a calibrated probability.")
        elif metrics.get("roc_auc_ovr_macro") is not None:
            st.caption(f"Macro one-vs-rest ROC-AUC: {metrics['roc_auc_ovr_macro']:.3f}.")
        st.markdown("### Confusion matrix")
        labels = metrics["labels"]
        cm = pd.DataFrame(metrics["confusion_matrix"], index=[f"Actual · {x}" for x in labels], columns=[f"Predicted · {x}" for x in labels])
        st.dataframe(cm, use_container_width=True)
        st.markdown("### Per-class results")
        report = metrics["report"]
        table_rows = [{"Class": cls, "Precision": report[cls]["precision"], "Recall": report[cls]["recall"], "F1": report[cls]["f1-score"], "Support": int(report[cls]["support"])} for cls in labels]
        st.dataframe(pd.DataFrame(table_rows).style.format({"Precision":"{:.3f}","Recall":"{:.3f}","F1":"{:.3f}"}), use_container_width=True, hide_index=True)
        model = st.session_state.model
        X_test = st.session_state.X_test
        if X_test is not None and hasattr(model, "predict_proba") and len(labels) == 2:
            try:
                import plotly.graph_objects as go
                from sklearn.metrics import precision_recall_curve, roc_curve
                class_order = [str(x) for x in model.classes_]
                positive_index = class_order.index(str(metrics["positive_class"]))
                scores = model.predict_proba(X_test)[:, positive_index]
                y_binary = (pd.Series(st.session_state.test_truth).astype(str) == str(metrics["positive_class"])).astype(int)
                if y_binary.nunique() == 2:
                    fpr_values, tpr_values, _ = roc_curve(y_binary, scores)
                    pr_precision, pr_recall, _ = precision_recall_curve(y_binary, scores)
                    left, right = st.columns(2)
                    roc_fig = go.Figure(go.Scatter(x=fpr_values, y=tpr_values, mode="lines", name="ROC"))
                    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line={"dash":"dash"}))
                    roc_fig.update_layout(title="ROC curve · held-out test", xaxis_title="False-positive rate", yaxis_title="True-positive rate", template="plotly_dark", margin={"l":10,"r":10,"t":45,"b":10})
                    pr_fig = go.Figure(go.Scatter(x=pr_recall, y=pr_precision, mode="lines", name="Precision–recall"))
                    pr_fig.update_layout(title="Precision–recall · held-out test", xaxis_title="Recall", yaxis_title="Precision", template="plotly_dark", margin={"l":10,"r":10,"t":45,"b":10})
                    left.plotly_chart(roc_fig, use_container_width=True)
                    right.plotly_chart(pr_fig, use_container_width=True)
                else:
                    st.info("ROC and precision–recall curves are not drawn because the selected positive class is absent from the test split.")
            except Exception as exc:
                st.info(f"Curve plotting was unavailable for this split: {exc}")
        st.warning("These are dataset-specific offline test results, not a guarantee of live-network performance. Check class balance, leakage, drift, and false-negative costs.")

elif page == "Intrusion Detection":
    _page_header("Intrusion Detection", "Score uploaded records with the current model and optional analyst-configured rules.")
    if st.session_state.model is None:
        st.info("Train a model first. Detections are generated only for uploaded dataset rows; no network traffic is captured.")
    else:
        classes = st.session_state.class_names or []
        default_benign = "(none — flag all predictions)"
        if "benign_class" not in st.session_state: st.session_state.benign_class = default_benign
        benign_opts = [default_benign] + classes
        benign_current = st.session_state.benign_class if st.session_state.benign_class in benign_opts else default_benign
        st.selectbox("Treat this predicted class as benign", benign_opts, index=benign_opts.index(benign_current), key="benign_class", help="Predictions other than the benign class are flagged. Choose the none option to flag every row for review.")
        score_col, score_col2 = st.columns(2)
        medium_cutoff = score_col.number_input("Medium score band starts at", min_value=0.0, max_value=1.0, value=0.50, step=0.05, help="Configurable triage band for an uncalibrated model score; not a universal probability threshold.")
        high_cutoff = score_col2.number_input("High score band starts at", min_value=float(medium_cutoff), max_value=1.0, value=max(0.80, float(medium_cutoff)), step=0.05, help="Configurable triage band; validate it against your dataset and review burden.")
        with st.expander("Configure simple threshold rules", expanded=False):
            usable = [c for c in df.columns if c != st.session_state.target and pd.api.types.is_numeric_dtype(df[c])]
            if not usable:
                st.caption("No numeric feature columns are available for threshold rules.")
            else:
                with st.form("rule_form"):
                    rc = st.selectbox("Numeric feature", usable)
                    operator = st.selectbox("Condition", [">", ">=", "<", "<="])
                    threshold = st.number_input("Threshold", value=float(pd.to_numeric(df[rc], errors="coerce").median()) if pd.to_numeric(df[rc], errors="coerce").notna().any() else 0.0)
                    severity = st.selectbox("Rule triage severity", ["Low", "Medium", "High", "Critical"])
                    enabled = st.checkbox("Enable rule", value=True)
                    save_rule = st.form_submit_button("Add rule")
                if save_rule:
                    st.session_state.rules.append({"name": f"{rc} {operator} {threshold:g}", "field": rc, "column": rc, "operator": operator, "value": threshold, "threshold": threshold, "severity": severity.lower(), "description": f"{rc} {operator} {threshold:g}", "enabled": enabled})
            if st.session_state.rules:
                st.dataframe(pd.DataFrame(st.session_state.rules), use_container_width=True, hide_index=True)
                if st.button("Clear rules"): st.session_state.rules = []
                st.caption("Rules supplement model predictions; they are not automatically interpreted as calibrated risk.")
        if st.button("Generate detections for current dataset", type="primary"):
            try:
                detections = _make_predictions(df)
                active_rules = [{k: r[k] for k in ("name", "field", "operator", "value", "severity", "description")} for r in st.session_state.rules if r.get("enabled", True)]
                rule_matches = evaluate_rules_api(df, active_rules) if evaluate_rules_api and active_rules else [[] for _ in range(len(df))]
                if not isinstance(rule_matches, list) or len(rule_matches) != len(df):
                    rule_matches = [[] for _ in range(len(df))]
                descriptions = []
                risk_levels = []
                risk_reasons = []
                severity_order = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
                score_col = "Model score (uncalibrated)"
                for row_idx, matches in enumerate(rule_matches):
                    row = detections.iloc[row_idx]
                    descriptions.append("; ".join(str(m.get("description") or f"{m.get('field')} {m.get('operator')} {m.get('value')}") for m in matches))
                    if matches:
                        detections.loc[row_idx, "Flagged"] = True
                        is_model_flag = str(row.get("Predicted class")) != str(st.session_state.benign_class)
                        detections.loc[row_idx, "Detection source"] = "Model + rule" if is_model_flag else "Rule-based signal"
                    level_candidates = [str(m.get("severity", "medium")).title() for m in matches]
                    if level_candidates:
                        risk = max(level_candidates, key=lambda value: severity_order.get(value, 2))
                    elif bool(row.get("Flagged")):
                        score = row.get(score_col)
                        score = float(score) if pd.notna(score) else 0.0
                        risk = "High" if score >= high_cutoff else "Medium" if score >= medium_cutoff else "Low"
                    else:
                        risk = "Low"
                    risk_levels.append(risk)
                    reasons = []
                    if bool(row.get("Flagged")):
                        score = row.get(score_col)
                        reasons.append(f"Model predicted {row.get('Predicted class')} with score {score if pd.notna(score) else 'unavailable'} (uncalibrated).")
                    reasons.extend(f"Configured rule {m.get('name')} matched: {m.get('description') or 'condition met'}." for m in matches)
                    risk_reasons.append(" ".join(reasons) if reasons else "No alert trigger; retained as a low-priority record.")
                detections["Triggered rules"] = descriptions
                detections["Rule alert"] = [bool(matches) for matches in rule_matches]
                detections["Risk level"] = risk_levels
                detections["Severity reason"] = risk_reasons
                time_col = next((c for c in df.columns if str(c).strip().lower() in {"timestamp", "event_time", "flow_start", "start_time", "datetime"}), None)
                detections["Event timestamp"] = df[time_col].values if time_col else None
                detections["Analyst review status"] = "New"
                st.session_state.detections = detections
                st.session_state.alerts = _build_alerts(detections)
                st.success(f"Scored {len(detections):,} rows; {int(detections['Flagged'].sum()):,} flagged for analyst review.")
            except Exception as exc:
                st.error(f"Could not generate detections: {exc}")
        detections = st.session_state.detections
        if detections is not None:
            nflag = int(detections["Flagged"].sum())
            x,y,z = st.columns(3); x.metric("Rows scored", f"{len(detections):,}"); y.metric("Flagged", f"{nflag:,}"); z.metric("Flag rate", f"{nflag/max(1,len(detections)):.1%}")
            st.warning("Scores are raw model outputs / uncalibrated estimates, not verified probabilities or a severity guarantee. Validate thresholds against your environment.")
            show_flagged = st.checkbox("Show flagged rows only", value=True)
            view = detections[detections["Flagged"]] if show_flagged else detections
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.download_button("Download detection results", data=_file_bytes(detections), file_name="detection_results.csv", mime="text/csv")

elif page == "Alert Investigation":
    _page_header("Alert Investigation", "Triage detection findings, update status, and record analyst notes.")
    alerts = st.session_state.alerts
    if not alerts:
        st.info("No alerts are available. Generate detections after training a model, or adjust the benign class and rules.")
    else:
        table = pd.DataFrame(alerts)
        f1, f2, f3 = st.columns(3)
        statuses = ["All"] + sorted(table["Status"].dropna().unique().tolist())
        status_filter = f1.selectbox("Status", statuses)
        class_filter = f2.selectbox("Predicted class", ["All"] + sorted(table["Predicted class"].dropna().unique().tolist()))
        risk_filter = f3.selectbox("Risk level", ["All"] + sorted(table["Risk level"].dropna().unique().tolist()))
        query = st.text_input("Search alert evidence, rule names, or notes")
        time_query = st.text_input("Event-time filter (text match)")
        indices = [i for i, alert in enumerate(alerts) if (status_filter == "All" or alert.get("Status") == status_filter) and (class_filter == "All" or alert.get("Predicted class") == class_filter) and (risk_filter == "All" or alert.get("Risk level") == risk_filter) and (not query or query.lower() in json.dumps(alert, default=str).lower()) and (not time_query or time_query.lower() in str(alert.get("Event timestamp", "")).lower())]
        brief_cols = [c for c in ("Alert ID", "Created (UTC)", "Event timestamp", "Predicted class", "Risk level", "Score (uncalibrated)", "Source", "Status") if c in table]
        st.dataframe(table.iloc[indices][brief_cols], use_container_width=True, hide_index=True)
        if not indices:
            st.info("No alerts match the selected filters.")
            selected = None
        else:
            selected = st.selectbox("Investigate alert", indices, format_func=lambda i: f"{alerts[i].get('Alert ID')} · {alerts[i].get('Predicted class')} · {alerts[i].get('Risk level')} · {alerts[i].get('Status')}")
        if selected is not None:
            alert = alerts[selected]
            st.markdown(f"### {alert.get('Alert ID')} · {alert.get('Predicted class')}")
            detail1, detail2 = st.columns(2)
            with detail1:
                st.write(f"**Created (UTC):** {alert.get('Created (UTC)')}")
                st.write(f"**Event timestamp:** {alert.get('Event timestamp', 'not available')}")
                st.write(f"**Detection source:** {alert.get('Source')}")
                st.write(f"**Score:** {alert.get('Score (uncalibrated)', 'not available')} (uncalibrated)")
                st.write(f"**Risk level:** {alert.get('Risk level')} · {alert.get('Severity reason')}")
                st.write(f"**Triggered rules:** {alert.get('Triggered rules') or 'None'}")
                st.write(f"**Feature evidence ({alert.get('Explanation method', 'not available')}):**")
                st.json(alert.get("Feature evidence", []), expanded=False)
                st.caption(alert.get("Explanation note", "No model feature evidence was available."))
            with detail2:
                status_options = ["New", "Under Investigation", "Benign", "Confirmed Suspicious"]
                current_status = alert.get("Status", "New")
                status = st.selectbox("Update status", status_options, index=status_options.index(current_status) if current_status in status_options else 0, key=f"alert_status_{selected}")
                note = st.text_area("Analyst notes", value=alert.get("Analyst notes", ""), key=f"alert_note_{selected}", height=110)
                if st.button("Save investigation", key=f"save_alert_{selected}"):
                    old_status = alerts[selected].get("Status", "New")
                    if status != old_status:
                        alerts[selected].setdefault("Status history", []).append({"from": old_status, "to": status, "changed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                    alerts[selected]["Status"] = status
                    alerts[selected]["Analyst notes"] = note
                    alerts[selected]["Updated (UTC)"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    st.success("Investigation updated for this session.")
                    st.rerun()
            with st.expander("Record fields"):
                st.json({k: (v.item() if hasattr(v, "item") else v) for k,v in alert.items()}, expanded=False)
        st.caption("Alert statuses and analyst notes are browser-session state unless your backend implements persistent alert storage.")
        st.download_button("Export filtered alerts", data=_file_bytes(table.iloc[indices]), file_name="alerts.csv", mime="text/csv")

elif page == "Reports & Export":
    _page_header("Reports & Export", "Download reproducible session outputs and a compact analysis summary.")
    if df is None:
        st.info("Upload a dataset to enable data exports. Model-specific reports appear after training.")
    else:
        st.download_button("Download source dataset", data=_file_bytes(df), file_name="source_dataset.csv", mime="text/csv")
        if st.session_state.detections is not None:
            st.download_button("Download latest detections", data=_file_bytes(st.session_state.detections), file_name="detection_results.csv", mime="text/csv")
        if st.session_state.alerts:
            st.download_button("Download alert register", data=_file_bytes(pd.DataFrame(st.session_state.alerts)), file_name="alert_register.csv", mime="text/csv")
        if st.session_state.model is not None:
            try:
                import joblib
                model_buffer = io.BytesIO()
                joblib.dump(st.session_state.model, model_buffer)
                st.download_button("Download locally trained model (.joblib)", data=model_buffer.getvalue(), file_name="nids_model.joblib", mime="application/octet-stream")
                st.caption("This artifact was generated from the current session’s fitted pipeline. Joblib files can execute code when deserialized; load only artifacts you generated and trust.")
            except Exception as exc:
                st.warning(f"Model artifact export unavailable: {exc}")
        metrics = st.session_state.metrics
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": st.session_state.filename, "rows": int(len(df)), "columns": [str(c) for c in df.columns],
            "dataset_source_user_declared": st.session_state.dataset_source,
            "dataset_license_user_declared": st.session_state.dataset_license,
            "dataset_description": st.session_state.dataset_description,
            "target": st.session_state.target, "missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()), "model": st.session_state.model_name,
            "evaluation": ({k: metrics[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted", "roc_auc", "roc_auc_ovr_macro", "average_precision", "false_positive_rate", "positive_class", "train_size", "test_size", "labels", "confusion_matrix", "report") if k in metrics} if metrics else None),
            "detections": ({"rows_scored": int(len(st.session_state.detections)), "flagged": int(st.session_state.detections["Flagged"].sum()), "score_note": "uncalibrated model output"} if st.session_state.detections is not None else None),
            "alerts": {"total": len(st.session_state.alerts), "by_status": pd.DataFrame(st.session_state.alerts)["Status"].value_counts().to_dict() if st.session_state.alerts else {}},
            "limitations": ["Offline uploaded data only", "No real-time traffic capture", "No automatic blocking", "Session-only unless backend persistence is configured"],
        }
        report_json = json.dumps(report, indent=2, default=str).encode("utf-8")
        st.download_button("Download analysis summary (JSON)", data=report_json, file_name="ids_analysis_report.json", mime="application/json")
        if report_backend.get("generate_report"):
            st.caption("Optional reporting backend detected; the standard JSON and CSV exports above remain available.")
        st.markdown("### Export coverage")
        st.write("The JSON summary contains dataset shape and quality counts, model metrics when available, detection totals, and operating limitations. CSV exports contain the underlying records needed for downstream review.")

elif page == "About":
    _page_header("About this dashboard", "AI-Based Network Intrusion Detection System · analyst-in-the-loop prototype.")
    st.markdown("### Intended use")
    st.write("This dashboard supports offline exploration of labeled tabular data, supervised classification, and investigation of flagged records. It is not a certified security appliance and should be validated with representative, correctly labeled data before operational use.")
    st.markdown("### Data handling")
    st.write("CSV content, fitted models, alerts, status changes, and analyst notes are held in the active Streamlit session. They are not durable records unless a separately configured backend persists them. Avoid sensitive information unless your hosting environment and retention controls have been reviewed.")
    st.markdown("### Safety and interpretation")
    st.write("No packet capture, network scanning, automatic host blocking, or remediation is performed. Classifier scores are labeled as uncalibrated model outputs; high scores should be investigated, not treated as confirmed incidents.")
    with st.expander("Backend integration status"):
        st.write({"validate_dataframe": bool(validate_dataframe), "load_csv_bytes": bool(load_csv_bytes), "fit_model": bool(fit_model_api), "evaluate_rules": bool(evaluate_rules_api), "generate_detections": bool(generate_detections_api), "alert_workflow": any(alert_backend.values()), "reporting": any(report_backend.values())})
