"""Resumen ejecutivo."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.ui import format_integer, hero, insight
from src.models.evaluate_model import kmeans_profile
from src.visualization.visualize import cluster_size_figure, resilience_figure


def render(
    data: pd.DataFrame,
    full_data: pd.DataFrame,
    summary: dict,
    names: dict[int, str],
) -> None:
    hero(
        "Resiliencia académica · Saber Pro 2024",
        "Perfiles de estudiantes con bajo INSE que alcanzan resultados sobresalientes.",
    )
    resilient = int(data["resiliencia_flag"].sum())
    rate = data["resiliencia_flag"].mean() * 100
    columns = st.columns(4)
    columns[0].metric("Estudiantes", format_integer(len(data)))
    columns[1].metric("Resilientes", format_integer(resilient))
    columns[2].metric("Tasa de resiliencia", f"{rate:.2f}%")
    columns[3].metric("Puntaje promedio", f"{data['punt_global'].mean():.1f}")

    st.caption(
        f"La selección actual representa {len(data) / len(full_data) * 100:.1f}% "
        "de la población modelada. Los indicadores responden a los filtros laterales."
    )
    profile = kmeans_profile(data).reset_index()
    profile["nombre"] = profile["cluster_kmeans"].astype(int).map(names)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(cluster_size_figure(profile), width="stretch")
    with right:
        st.plotly_chart(resilience_figure(profile, rate), width="stretch")

    overall = summary["population"]
    insight(
        f"En la población completa se identificaron <b>{format_integer(overall['resilient_students'])}</b> "
        f"estudiantes resilientes, equivalentes al <b>{overall['resilience_rate']:.2f}%</b>. "
        "El perfil con mayor concentración combina bajo nivel material con mayor educación parental."
    )
    st.subheader("Lectura ejecutiva")
    st.markdown(
        """
        - La resiliencia está presente en todos los perfiles, pero no con la misma frecuencia.
        - La educación de los padres distingue mejor el perfil con mayor resiliencia que la cantidad de bienes.
        - K-Means ofrece perfiles amplios para toda la población; DBSCAN complementa la lectura con estructuras densas y casos poco comunes.
        - Los resultados describen asociaciones y no prueban relaciones causales.
        """
    )
