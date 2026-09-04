"""Gráficas Plotly reutilizables para los resultados de clustering."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLORS = ["#155e75", "#0f766e", "#f97316", "#6366f1", "#db2777", "#64748b"]


def _layout(figure: go.Figure, title: str) -> go.Figure:
    figure.update_layout(
        title=title,
        template="plotly_white",
        font={"family": "Inter, ui-sans-serif, system-ui", "color": "#172033"},
        title_font={"size": 18},
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        legend_title_text="",
    )
    return figure


def cluster_size_figure(profile: pd.DataFrame, name_column: str = "nombre") -> go.Figure:
    frame = profile.reset_index() if profile.index.name else profile.copy()
    figure = px.bar(
        frame,
        x=name_column,
        y="porcentaje_poblacion",
        color=name_column,
        color_discrete_sequence=COLORS,
        text=frame["porcentaje_poblacion"].map(lambda value: f"{value:.1f}%"),
    )
    figure.update_layout(showlegend=False)
    figure.update_xaxes(title="")
    figure.update_yaxes(title="Población (%)")
    return _layout(figure, "Distribución de estudiantes por perfil")


def resilience_figure(profile: pd.DataFrame, overall_rate: float) -> go.Figure:
    frame = profile.reset_index() if profile.index.name else profile.copy()
    figure = px.bar(
        frame,
        x="nombre",
        y="tasa_resiliencia",
        color="nombre",
        color_discrete_sequence=COLORS,
        text=frame["tasa_resiliencia"].map(lambda value: f"{value:.2f}%"),
    )
    figure.add_hline(
        y=overall_rate,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"Promedio {overall_rate:.2f}%",
    )
    figure.update_layout(showlegend=False)
    figure.update_xaxes(title="")
    figure.update_yaxes(title="Resiliencia (%)")
    return _layout(figure, "Concentración de resiliencia por perfil")


def pca_scatter_figure(
    data: pd.DataFrame, names: dict[int, str], sample_size: int = 12_000
) -> go.Figure:
    sample = data.sample(n=min(sample_size, len(data)), random_state=42).copy()
    sample["perfil"] = sample["cluster_kmeans"].astype(int).map(names)
    figure = px.scatter(
        sample,
        x="pca_plot_1",
        y="pca_plot_2",
        color="perfil",
        color_discrete_sequence=COLORS,
        opacity=0.42,
        hover_data={"pca_plot_1": ":.2f", "pca_plot_2": ":.2f", "perfil": True},
    )
    figure.update_traces(marker={"size": 5})
    figure.update_xaxes(title="Componente principal 1")
    figure.update_yaxes(title="Componente principal 2")
    return _layout(figure, "Proyección de los perfiles en dos dimensiones")


def profile_heatmap_figure(profile: pd.DataFrame) -> go.Figure:
    columns = [
        "puntaje_promedio",
        "percentil_promedio",
        "inse_promedio",
        "bienes_promedio",
        "tasa_resiliencia",
    ]
    values = profile[columns]
    standard_deviation = values.std().replace(0, 1)
    scaled = ((values - values.mean()) / standard_deviation).fillna(0)
    figure = px.imshow(
        scaled,
        x=["Puntaje", "Percentil", "INSE", "Bienes", "Resiliencia"],
        y=profile["nombre"],
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        text_auto=".1f",
        aspect="auto",
    )
    figure.update_xaxes(title="")
    figure.update_yaxes(title="")
    return _layout(figure, "Diferencias relativas entre perfiles")


def goods_heatmap_figure(data: pd.DataFrame, binary_columns: list[str]) -> go.Figure:
    percentages = data.groupby("nombre_perfil")[binary_columns].agg(
        lambda column: column.eq("Si").mean() * 100
    )
    labels = [column.replace("fami_tiene", "").replace("hornomicroogas", "microondas").title() for column in binary_columns]
    figure = px.imshow(
        percentages,
        x=labels,
        y=percentages.index,
        color_continuous_scale="Teal",
        text_auto=".0f",
        aspect="auto",
    )
    figure.update_xaxes(title="")
    figure.update_yaxes(title="")
    return _layout(figure, "Disponibilidad de bienes y servicios (%)")


def category_comparison_figure(data: pd.DataFrame, column: str, title: str) -> go.Figure:
    table = pd.crosstab(data[column], data["grupo_resiliencia"], normalize="columns") * 100
    frame = table.reset_index().melt(id_vars=column, var_name="Grupo", value_name="Porcentaje")
    figure = px.bar(
        frame,
        x=column,
        y="Porcentaje",
        color="Grupo",
        barmode="group",
        color_discrete_map={
            "Bajo INSE no sobresaliente": "#94a3b8",
            "Resiliente": "#f97316",
        },
    )
    figure.update_xaxes(title="", tickangle=-25)
    figure.update_yaxes(title="Porcentaje dentro del grupo")
    return _layout(figure, title)


def dbscan_size_figure(profile: pd.DataFrame) -> go.Figure:
    frame = profile.reset_index()
    frame["grupo"] = frame["cluster_dbscan"].map(
        lambda value: "Ruido" if value == -1 else f"Cluster {int(value)}"
    )
    figure = px.bar(
        frame,
        x="grupo",
        y="estudiantes",
        color="grupo",
        color_discrete_sequence=COLORS,
    )
    figure.update_layout(showlegend=False)
    figure.update_xaxes(title="")
    figure.update_yaxes(title="Estudiantes")
    return _layout(figure, "Distribución de DBSCAN")
