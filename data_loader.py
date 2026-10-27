"""CSV ingestion and schema validation for arbitrary tabular network-flow data.

``load_csv`` accepts a filesystem path or file-like object, normalizes column
whitespace/BOM, and raises actionable errors rather than silently accepting an
empty or structurally malformed file. It does not assume any CICIDS columns.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import IO, Union

import numpy as np
import pandas as pd

CsvSource = Union[str, Path, IO[str], IO[bytes]]


def validate_dataframe(df: pd.DataFrame, *, min_rows: int = 1) -> pd.DataFrame:
    """Validate and return a cleaned copy of a tabular dataset.

    Empty rows are preserved, but columns with blank names, duplicate names
    after trimming, an empty frame, or too few records are rejected. Numeric
    infinities are converted to missing values for downstream imputers.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if df.empty or len(df.index) < min_rows:
        raise ValueError(f"Dataset must contain at least {min_rows} data row(s).")
    if len(df.columns) == 0:
        raise ValueError("Dataset must contain at least one column.")
    out = df.copy()
    names = [str(c).replace("\ufeff", "").strip() for c in out.columns]
    if any(not name for name in names):
        raise ValueError("Column names cannot be blank.")
    if len(set(names)) != len(names):
        raise ValueError("Column names must be unique after whitespace normalization.")
    out.columns = names
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
    return out


def load_csv(source: CsvSource, *, encoding: str = "utf-8-sig", **read_csv_kwargs) -> pd.DataFrame:
    """Read and validate a CSV from a local path or file-like object.

    Defaults use pandas' Python-engine delimiter sniffing to handle common
    comma/semicolon/tab exports. Pass ``sep=','`` for a known delimiter. No
    code, formulas, or serialized objects are executed or deserialized.
    """
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"CSV file does not exist or is not a file: {path}")
        if path.stat().st_size == 0:
            raise ValueError("CSV file is empty.")
        source = path
    kwargs = dict(read_csv_kwargs)
    kwargs.setdefault("encoding", encoding)
    kwargs.setdefault("sep", None)
    kwargs.setdefault("engine", "python")
    kwargs.setdefault("on_bad_lines", "error")
    try:
        frame = pd.read_csv(source, **kwargs)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError, csv.Error) as exc:
        raise ValueError(f"Could not read a valid CSV (file may be empty or malformed): {exc}") from exc
    return validate_dataframe(frame)


# Friendly alias for callers that prefer an explicit verb.
read_csv = load_csv
