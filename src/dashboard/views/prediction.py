"""Carga de archivos y asignación de perfiles K-Means."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src.dashboard.ui import format_integer, hero
from src.models.predict_model import predict_kmeans, prediction_template


def render(bundle: dict) -> None:
    hero(
        "Clasificar nuevos registros",
        "Asigna perfiles K-Means usando exactamente la preparación del entrenamiento.",
    )
    st.info(
        f"El modelo fue construido para estudiantes con INSE menor o igual a "
        f"{bundle['inse_threshold']:.2f}. Los registros fuera de ese rango no serán clasificados."
    )
    template = prediction_template(bundle)
    st.download_button(
        "Descargar plantilla CSV",
        data=template.to_csv(index=False).encode("utf-8-sig"),
        file_name="plantilla_prediccion_kmeans.csv",
        mime="text/csv",
        width="stretch",
    )
    uploaded = st.file_uploader("Cargar archivo CSV", type=["csv"])
    if uploaded is None:
        st.caption("La aplicación procesa el archivo en memoria y no lo almacena.")
        return

    try:
        raw = uploaded.getvalue()
        incoming = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
        result = predict_kmeans(incoming, bundle)
    except Exception as error:
        st.error(f"No fue posible procesar el archivo: {error}")
        return

    assigned = result["cluster_kmeans"].notna().sum()
    columns = st.columns(3)
    columns[0].metric("Registros recibidos", format_integer(len(result)))
    columns[1].metric("Asignados", format_integer(assigned))
    columns[2].metric("No asignados", format_integer(len(result) - assigned))
    st.dataframe(result, width="stretch", hide_index=True)
    st.caption(
        "La distancia al centroide permite comparar cercanía dentro del modelo, pero no es una probabilidad."
    )
    st.download_button(
        "Descargar resultados",
        data=result.to_csv(index=False).encode("utf-8-sig"),
        file_name="predicciones_kmeans.csv",
        mime="text/csv",
        width="stretch",
    )
