"""Resultados del modelo DBSCAN."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.ui import format_integer, hero, insight
from src.models.evaluate_model import dbscan_profile
from src.visualization.visualize import dbscan_size_figure


def render(data: pd.DataFrame, summary: dict) -> None:
    hero("DBSCAN", "Estructuras densas y perfiles contextuales poco comunes.")
    sample = data[data["cluster_dbscan"].notna()].copy()
    if sample.empty:
        st.info("Los filtros actuales no contienen estudiantes incluidos en la muestra DBSCAN.")
        return
    grouped = sample[sample["cluster_dbscan"] != -1]
    profile = dbscan_profile(data)
    noise_rate = (sample["cluster_dbscan"] == -1).mean() * 100
    columns = st.columns(4)
    columns[0].metric("Muestra", format_integer(len(sample)))
    columns[1].metric("Clusters", grouped["cluster_dbscan"].nunique())
    columns[2].metric("Ruido", f"{noise_rate:.2f}%")
    columns[3].metric("Silhouette global", f"{summary['dbscan']['metrics']['silhouette']:.3f}")
    st.plotly_chart(dbscan_size_figure(profile), width="stretch")

    plot = sample.sample(n=min(12_000, len(sample)), random_state=42).copy()
    plot["grupo"] = plot["cluster_dbscan"].map(
        lambda value: "Ruido" if value == -1 else f"Cluster {int(value)}"
    )
    figure = px.scatter(
        plot,
        x="dbscan_pca_1",
        y="dbscan_pca_2",
        color="grupo",
        opacity=0.45,
        template="plotly_white",
        title="DBSCAN en sus dos primeras componentes",
    )
    figure.update_traces(marker={"size": 5})
    st.plotly_chart(figure, width="stretch")
    st.dataframe(profile.round(2), width="stretch")
    insight(
        "DBSCAN complementa a K-Means, pero no define por sí solo la resiliencia. "
        "La etiqueta de ruido significa que el contexto del estudiante es poco común dentro "
        "de la muestra, no que su desempeño sea necesariamente sobresaliente."
    )
