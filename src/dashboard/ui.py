"""Elementos visuales y controles comunes del dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


CSS = """
<style>
    .stApp { background: #f7f9fc; color: #0f172a; }
    [data-testid="stSidebar"] {
        background: #0f172a;
        color: #e2e8f0;
    }
    /* El texto del panel oscuro es claro; no se fuerza ese color en los
       controles que Streamlit muestra sobre una superficie blanca. */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label * {
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] [data-baseweb="select"],
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-baseweb="popover"],
    [data-baseweb="popover"] * {
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] {
        background: #ffffff;
        border-radius: 8px;
    }
    [data-testid="stSidebar"] .stButton > button {
        color: #e2e8f0;
        border-color: #64748b;
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,.92);
        border: 1px solid #e5eaf1;
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 8px 26px rgba(15,23,42,.05);
    }
    .hero {
        padding: 24px 28px;
        border-radius: 20px;
        background: linear-gradient(120deg, #0f172a 0%, #155e75 62%, #0f766e 100%);
        color: white;
        margin-bottom: 22px;
    }
    .hero h1 { margin: 0 0 6px; font-size: 2rem; }
    .hero h1 { color: #ffffff; }
    .hero p { margin: 0; color: #dbeafe; }
    .insight {
        background: white;
        color: #0f172a;
        border-left: 4px solid #f97316;
        border-radius: 12px;
        padding: 16px 18px;
        margin: 8px 0 16px;
        box-shadow: 0 6px 20px rgba(15,23,42,.04);
    }
    .muted { color: #64748b; font-size: .92rem; }
    [data-testid="stMetric"] [data-testid="stMetricLabel"],
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #0f172a;
    }
    div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
</style>
"""


def inject_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<section class="hero"><h1>{title}</h1><p>{subtitle}</p></section>',
        unsafe_allow_html=True,
    )


def insight(text: str) -> None:
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)


def format_integer(value: float | int) -> str:
    return f"{int(value):,}".replace(",", ".")


def render_global_filters(
    data: pd.DataFrame, names: dict[int, str]
) -> dict[str, list[Any]]:
    st.sidebar.markdown("### Filtros")
    cluster_options = sorted(data["cluster_kmeans"].astype(int).unique())
    selected_clusters = st.sidebar.multiselect(
        "Perfil K-Means",
        cluster_options,
        format_func=lambda value: names.get(int(value), f"Perfil {value}"),
    )
    filters: dict[str, list[Any]] = {"cluster_kmeans": selected_clusters}
    choices = [
        ("estu_prgm_departamento", "Departamento"),
        ("estu_metodo_prgm", "Metodología"),
        ("inst_origen", "Origen institucional"),
        ("estu_nucleo_pregrado", "Núcleo académico"),
    ]
    for column, label in choices:
        options = sorted(data[column].dropna().astype(str).unique().tolist())
        filters[column] = st.sidebar.multiselect(label, options)
    if st.sidebar.button("Limpiar filtros", width="stretch"):
        st.rerun()
    return filters
