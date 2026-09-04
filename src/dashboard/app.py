"""Punto de entrada del dashboard Streamlit."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.data import (  # noqa: E402
    apply_filters,
    cached_cluster_data,
    cached_summary,
)
from src.dashboard.views import (  # noqa: E402
    dbscan,
    methodology,
    overview,
    profiles,
    resilience,
)
from src.dashboard.ui import inject_styles, render_global_filters  # noqa: E402
from src.data.make_dataset import default_paths  # noqa: E402

st.set_page_config(
    page_title="Resiliencia Saber Pro 2024",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

paths = default_paths()
missing = [str(paths[key]) for key in ["cluster_data", "summary"] if not paths[key].exists()]
if missing:
    st.error("Faltan artefactos necesarios para iniciar el dashboard.")
    st.code("python -m src.models.train_model")
    st.stop()

data = cached_cluster_data(str(paths["cluster_data"]))
summary = cached_summary(str(paths["summary"]))
names = {int(key): value for key, value in summary["kmeans"]["cluster_names"].items()}

st.sidebar.markdown("## ◈ Saber Pro")
st.sidebar.caption("Resiliencia académica · 2024")
page = st.sidebar.radio(
    "Navegación",
    [
        "Resumen ejecutivo",
        "Perfiles K-Means",
        "Resiliencia académica",
        "DBSCAN",
        "Metodología",
    ],
    label_visibility="collapsed",
)

filtered = data
if page in {"Resumen ejecutivo", "Perfiles K-Means", "Resiliencia académica", "DBSCAN"}:
    filters = render_global_filters(data, names)
    filtered = apply_filters(data, filters)
    if filtered.empty:
        st.warning("La combinación de filtros no contiene registros.")
        st.stop()

if page == "Resumen ejecutivo":
    overview.render(filtered, data, summary, names)
elif page == "Perfiles K-Means":
    profiles.render(filtered, data, names)
elif page == "Resiliencia académica":
    resilience.render(filtered, names)
elif page == "DBSCAN":
    dbscan.render(filtered, summary)
else:
    methodology.render(summary)

st.sidebar.divider()
st.sidebar.caption("Fuente: ICFES · Examen Saber Pro 2024")
