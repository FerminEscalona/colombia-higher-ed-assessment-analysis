"""Análisis específico de resiliencia académica."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.ui import hero, insight
from src.models.evaluate_model import kmeans_profile
from src.visualization.visualize import category_comparison_figure, resilience_figure


def render(data: pd.DataFrame, names: dict[int, str]) -> None:
    hero(
        "Resiliencia académica",
        "Qué diferencia a quienes alcanzan el percentil 90 o superior dentro del bajo INSE.",
    )
    frame = data.copy()
    frame["grupo_resiliencia"] = np.where(
        frame["resiliencia_flag"], "Resiliente", "Bajo INSE no sobresaliente"
    )
    rate = frame["resiliencia_flag"].mean() * 100
    profile = kmeans_profile(frame).reset_index()
    profile["nombre"] = profile["cluster_kmeans"].astype(int).map(names)
    st.plotly_chart(resilience_figure(profile, rate), width="stretch")

    numeric = frame.groupby("grupo_resiliencia")[[
        "punt_global",
        "percentil_global",
        "estu_inse_individual",
        "indice_bienes_hogar",
    ]].mean()
    st.subheader("Comparación numérica")
    st.dataframe(numeric.round(2), width="stretch")

    selected = st.selectbox(
        "Característica para comparar",
        options=[
            ("fami_educacionmadre", "Educación de la madre"),
            ("fami_educacionpadre", "Educación del padre"),
            ("estu_metodo_prgm", "Metodología del programa"),
            ("inst_caracter_academico", "Carácter de la institución"),
            ("inst_origen", "Origen de la institución"),
            ("estu_prgm_departamento", "Departamento del programa"),
        ],
        format_func=lambda option: option[1],
    )
    st.plotly_chart(
        category_comparison_figure(frame, selected[0], selected[1]),
        width="stretch",
    )
    insight(
        "La evidencia central es una mayor presencia de educación parental media y superior "
        "entre los resilientes. La modalidad presencial, las universidades y las instituciones "
        "oficiales también tienen mayor participación relativa."
    )
    st.warning(
        "Estas diferencias son descriptivas. No demuestran que una característica familiar, "
        "institucional o territorial cause el resultado sobresaliente."
    )
