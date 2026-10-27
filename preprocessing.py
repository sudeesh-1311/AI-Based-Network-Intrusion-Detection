"""Leakage-aware feature selection, splitting, and preprocessing.

The public contract is ``split_dataset(df, target_col, test_size, random_state)``
returning ``X_train, X_test, y_train, y_test, preprocessor``. The returned
sklearn ``ColumnTransformer`` is deliberately *unfitted*: fit it only through a
Pipeline on the training partition. Obvious label/name columns other than the
chosen target are rejected instead of being allowed to leak target information.
"""
from __future__ import annotations

import re
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _normal_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def is_obvious_label_column(column: object, target_col: object) -> bool:
    """Return whether a non-target column clearly encodes attack labels/names."""
    if str(column) == str(target_col):
        return False
    n = _normal_name(column)
    tokens = set(n.split("_"))
    exact_label_names = {"label", "class", "target", "attack", "attack_type", "attack_name", "attack_label", "category"}
    return n in exact_label_names or bool(tokens & {"label", "class", "target"}) or ("attack" in tokens and bool(tokens & {"type", "name", "category", "label", "family"}))


def feature_frame(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Validate a supervised frame and return features with obvious labels excluded."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} was not found.")
    if df.empty:
        raise ValueError("Dataset has no rows.")
    if df[target_col].isna().any():
        raise ValueError("Target column contains missing values; remove or resolve them before training.")
    leakage = [c for c in df.columns if is_obvious_label_column(c, target_col)]
    X = df.drop(columns=[target_col, *leakage], errors="ignore").copy()
    for column in X.select_dtypes(include=[np.number]).columns:
        X[column] = X[column].replace([np.inf, -np.inf], np.nan)
    y = df[target_col].copy()
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after excluding the target and label-derived columns.")
    if y.nunique(dropna=True) < 2:
        raise ValueError("Classification requires at least two target classes.")
    return X, y


def make_preprocessor(X: pd.DataFrame, *, scale_numeric: bool = True) -> ColumnTransformer:
    """Create an unfitted transformer; optionally scale numeric columns."""
    if not isinstance(X, pd.DataFrame) or X.shape[1] == 0:
        raise ValueError("X must be a non-empty feature DataFrame.")
    numeric = list(X.select_dtypes(include=["number", "bool"]).columns)
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:  # scikit-learn < 1.2
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)


def split_dataset(
    df: pd.DataFrame, target_col: str, test_size: float = 0.25, random_state: int = 42,
    scale_numeric: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """Split supervised data and return an unfitted ColumnTransformer.

    Exact duplicate feature rows are grouped so copies do not cross the split.
    When no duplicate feature rows exist, stratification is used when feasible.
    All imputation, scaling, and encoding is learned later from training data
    only by the estimator Pipeline.
    """
    if not 0 < float(test_size) < 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive).")
    X, y = feature_frame(df, target_col)
    counts = y.value_counts(dropna=False)
    n_test = max(1, int(round(len(y) * float(test_size))))
    n_train = len(y) - n_test
    stratify = y if counts.min() >= 2 and n_test >= len(counts) and n_train >= len(counts) else None
    try:
        if X.duplicated(keep=False).any():
            groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
            if len(pd.unique(groups)) < 2:
                raise ValueError("All records share identical feature values; a duplicate-safe train/test split is not possible.")
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(splitter.split(X, y, groups=groups))
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            if y_train.nunique() < 2:
                raise ValueError("Duplicate feature groups leave fewer than two training classes; adjust the dataset or split settings.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify
            )
    except ValueError as exc:
        raise ValueError(f"Unable to create train/test split: {exc}") from exc
    return X_train, X_test, y_train, y_test, make_preprocessor(X_train, scale_numeric=scale_numeric)
