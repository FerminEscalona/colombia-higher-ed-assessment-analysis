"""Metodología, métricas y limitaciones."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.ui import hero


def render(summary: dict) -> None:
    hero("Metodología", "Cómo se construyeron y cómo deben interpretarse los perfiles.")
    st.subheader("Definición de resiliencia")
    st.markdown(
        f"Un estudiante resiliente pertenece al 25% inferior del INSE "
        f"(umbral **{summary['population']['inse_threshold']:.2f}**) y alcanza un "
        "percentil global igual o superior a 90."
    )
    st.subheader("K-Means")
    kmeans = summary["kmeans"]
    st.dataframe(
        pd.DataFrame(
            {
                "Indicador": ["Clusters", "Componentes PCA", "Varianza conservada", "Silhouette", "Davies-Bouldin"],
                "Valor": [
                    str(kmeans["clusters"]),
                    str(kmeans["pca_components"]),
                    f"{kmeans['explained_variance'] * 100:.2f}%",
                    f"{kmeans['metrics']['silhouette']:.3f}",
                    f"{kmeans['metrics']['davies_bouldin']:.3f}",
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.subheader("DBSCAN")
    dbscan = summary["dbscan"]
    st.dataframe(
        pd.DataFrame(
            {
                "Indicador": ["Muestra", "Clusters", "Componentes PCA", "Ruido", "Silhouette", "Davies-Bouldin"],
                "Valor": [
                    str(dbscan["sample_size"]),
                    str(dbscan["clusters"]),
                    str(dbscan["pca_components"]),
                    f"{dbscan['noise_rate']:.2f}%",
                    f"{dbscan['metrics']['silhouette']:.3f}",
                    f"{dbscan['metrics']['davies_bouldin']:.3f}",
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.subheader("Variables y alcance")
    st.markdown(
        """
        - Los clusters usan contexto familiar, bienes del hogar y trayectoria de preparación.
        - Puntaje, percentil, INSE y resiliencia no participan en la creación de los grupos.
        - Las variables ordinales se estandarizan; variables binarias y one-hot permanecen como 0/1.
        - Categorías con menos del 0,5% se agrupan para evitar columnas casi vacías.
        - K-Means puede asignar nuevos registros dentro de la población estudiada; DBSCAN se presenta de forma descriptiva.
        """
    )
    st.markdown(
    """
    <style>
    div[data-testid="stAlert"] * {
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    st.warning(
        "Los clusters son perfiles exploratorios. No son diagnósticos individuales y no permiten establecer causalidad."
    )
