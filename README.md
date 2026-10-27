# Model artifacts

Trained model binaries are not bundled. The default dashboard keeps a fitted model in the active Streamlit session. If adding locally trained artifacts, store them only when needed and keep them out of version control. Never load untrusted pickle/joblib files: deserialization can execute code. Reproducibility depends on code version, package versions, training data, split configuration, and random seed.
