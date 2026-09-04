"""Carga y filtros compartidos por las páginas del dashboard."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.data.make_dataset import load_cluster_results


@st.cache_data(show_spinner="Cargando resultados...")
def cached_cluster_data(path: str) -> pd.DataFrame:
    return load_cluster_results(Path(path))


@st.cache_data(show_spinner=False)
def cached_summary(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_filters(data: pd.DataFrame, filters: dict[str, list[Any]]) -> pd.DataFrame:
    filtered = data
    for column, values in filters.items():
        if values:
            filtered = filtered[filtered[column].isin(values)]
    return filtered.copy()
