"""Exploración de perfiles K-Means."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.ui import hero, insight
from src.features.build_features import BINARY_COLUMNS
from src.models.evaluate_model import kmeans_profile
from src.visualization.visualize import (
    goods_heatmap_figure,
    pca_scatter_figure,
    profile_heatmap_figure,
)


def render(data: pd.DataFrame, full_data: pd.DataFrame, names: dict[int, str]) -> None:
    hero("Perfiles K-Means", "Cuatro contextos familiares y de trayectoria dentro del bajo INSE.")
    profile = kmeans_profile(data).reset_index()
    profile["nombre"] = profile["cluster_kmeans"].astype(int).map(names)
    display_columns = [
        "nombre",
        "estudiantes",
        "porcentaje_poblacion",
        "puntaje_promedio",
        "inse_promedio",
        "bienes_promedio",
        "tasa_resiliencia",
    ]
    st.dataframe(
        profile[display_columns].style.format(
            {
                "porcentaje_poblacion": "{:.1f}%",
                "puntaje_promedio": "{:.1f}",
                "inse_promedio": "{:.1f}",
                "bienes_promedio": "{:.2f}",
                "tasa_resiliencia": "{:.2f}%",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.plotly_chart(profile_heatmap_figure(profile), width="stretch")
    st.plotly_chart(pca_scatter_figure(data, names), width="stretch")

    named = data.copy()
    named["nombre_perfil"] = named["cluster_kmeans"].astype(int).map(names)
    st.plotly_chart(goods_heatmap_figure(named, BINARY_COLUMNS), width="stretch")

    insight(
        "El perfil <b>Mayor capital educativo familiar</b> tiene la mayor tasa de resiliencia "
        "sin poseer la mayor cantidad de bienes. El perfil de <b>Mayor disponibilidad material</b> "
        "concentra más computador, internet, lavadora y televisión."
    )
    st.subheader("Descripción de los perfiles")
    descriptions = {
        "Mayor vulnerabilidad observada": "Predomina el estrato 1 y la educación primaria incompleta de los padres.",
        "Mayor disponibilidad material": "Predominan estratos 2 y 3 y una mayor disponibilidad de bienes y conectividad.",
        "Preparación académica reportada": "Concentra estudiantes con respuestas disponibles sobre refuerzos, cursos y simulacros.",
        "Mayor capital educativo familiar": "Combina estratos bajos con niveles educativos parentales relativamente mayores.",
    }
    for name in profile["nombre"]:
        st.markdown(f"**{name}.** {descriptions.get(name, 'Perfil contextual identificado por K-Means.')}")

    st.download_button(
        "Descargar selección visible",
        data=data.to_csv(index=False).encode("utf-8-sig"),
        file_name="saber_pro_clusters_filtrados.csv",
        mime="text/csv",
        width="stretch",
    )
