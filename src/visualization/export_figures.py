"""Exporta figuras estáticas para informes a partir del clustering entrenado.

Uso:
    python -m src.visualization.export_figures
"""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.make_dataset import default_paths, load_cluster_results
from src.features.build_features import BINARY_COLUMNS, EDUCATION_ORDER
from src.models.evaluate_model import dbscan_profile, kmeans_profile

COLORS = ["#155e75", "#0f766e", "#f97316", "#6366f1", "#db2777", "#64748b"]
TEXT_COLOR = "#172033"


def _prepare_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "axes.titleweight": "bold",
        }
    )


def _save(figure: plt.Figure, output_dir: Path, filename: str) -> Path:
    output = output_dir / filename
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def _profile(data: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    profile = kmeans_profile(data).reset_index()
    profile["nombre"] = profile["cluster_kmeans"].astype(int).map(names)
    return profile


def _name(value: str) -> str:
    return fill(value, width=24)


def export_static_figures(
    data: pd.DataFrame, summary: dict, output_dir: str | Path
) -> list[Path]:
    """Create the report-ready PNG figures used to explain the clustering."""
    _prepare_style()
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    names = {int(key): value for key, value in summary["kmeans"]["cluster_names"].items()}
    profile = _profile(data, names)
    profile["etiqueta"] = profile["nombre"].map(_name)
    outputs: list[Path] = []

    figure, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.bar(profile["etiqueta"], profile["porcentaje_poblacion"], color=COLORS[: len(profile)])
    axis.bar_label(bars, labels=[f"{value:.1f}%" for value in profile["porcentaje_poblacion"]], padding=4)
    axis.set(title="Distribución de estudiantes por perfil K-Means", ylabel="Población (%)", xlabel="")
    axis.set_ylim(0, profile["porcentaje_poblacion"].max() * 1.18)
    outputs.append(_save(figure, target, "01_distribucion_perfiles_kmeans.png"))

    overall_rate = data["resiliencia_flag"].mean() * 100
    figure, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.bar(profile["etiqueta"], profile["tasa_resiliencia"], color=COLORS[: len(profile)])
    axis.bar_label(bars, labels=[f"{value:.2f}%" for value in profile["tasa_resiliencia"]], padding=4)
    axis.axhline(overall_rate, color="#dc2626", linestyle="--", label=f"Promedio: {overall_rate:.2f}%")
    axis.set(title="Concentración de resiliencia por perfil", ylabel="Resiliencia (%)", xlabel="")
    axis.legend(frameon=False)
    axis.set_ylim(0, max(profile["tasa_resiliencia"].max(), overall_rate) * 1.24)
    outputs.append(_save(figure, target, "02_resiliencia_por_perfil.png"))

    sample = data.sample(n=min(12_000, len(data)), random_state=42).copy()
    figure, axis = plt.subplots(figsize=(10, 6.5))
    for position, cluster in enumerate(sorted(sample["cluster_kmeans"].unique())):
        group = sample[sample["cluster_kmeans"] == cluster]
        axis.scatter(
            group["pca_plot_1"], group["pca_plot_2"], s=9, alpha=0.35,
            color=COLORS[position], label=names[int(cluster)], edgecolors="none"
        )
    axis.set(title="Proyección PCA de los perfiles K-Means", xlabel="Componente principal 1", ylabel="Componente principal 2")
    axis.legend(markerscale=2, frameon=False, fontsize=9)
    outputs.append(_save(figure, target, "03_proyeccion_pca_kmeans.png"))

    dimensions = ["puntaje_promedio", "percentil_promedio", "inse_promedio", "bienes_promedio", "tasa_resiliencia"]
    labels = ["Puntaje", "Percentil", "INSE", "Bienes", "Resiliencia"]
    values = profile[dimensions]
    scaled = (values - values.mean()) / values.std(ddof=0).replace(0, 1)
    figure, axis = plt.subplots(figsize=(10, 4.8))
    sns.heatmap(
        scaled, annot=True, fmt=".1f", center=0, cmap="RdBu_r", linewidths=.8,
        cbar_kws={"label": "Diferencia estandarizada"}, xticklabels=labels,
        yticklabels=profile["nombre"], ax=axis,
    )
    axis.set(title="Diferencias relativas entre perfiles K-Means", xlabel="", ylabel="")
    outputs.append(_save(figure, target, "04_perfiles_contextuales_estandarizados.png"))

    named = data.copy()
    named["perfil"] = named["cluster_kmeans"].astype(int).map(names)
    goods = named.groupby("perfil")[BINARY_COLUMNS].agg(lambda column: column.eq("Si").mean() * 100)
    goods = goods.reindex(profile["nombre"])
    good_labels = [
        "Automóvil", "Computador", "Microondas", "Internet", "Lavadora", "Motocicleta", "TV"
    ]
    figure, axis = plt.subplots(figsize=(11, 4.8))
    sns.heatmap(goods, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=.8,
                cbar_kws={"label": "Disponibilidad (%)"}, xticklabels=good_labels, ax=axis)
    axis.set(title="Disponibilidad de bienes y servicios por perfil", xlabel="", ylabel="")
    outputs.append(_save(figure, target, "05_bienes_y_servicios_por_perfil.png"))

    comparison = data.copy()
    comparison["grupo"] = np.where(comparison["resiliencia_flag"], "Resiliente", "Bajo INSE no sobresaliente")
    education = pd.crosstab(comparison["fami_educacionmadre"], comparison["grupo"], normalize="columns") * 100
    education = education.reindex(EDUCATION_ORDER).dropna(how="all")
    figure, axis = plt.subplots(figsize=(11, 6))
    education.plot.bar(ax=axis, color=["#94a3b8", "#f97316"], width=.78)
    axis.set(title="Educación de la madre según condición de resiliencia", xlabel="", ylabel="Porcentaje dentro del grupo")
    axis.tick_params(axis="x", rotation=28)
    axis.legend(title="", frameon=False)
    outputs.append(_save(figure, target, "06_educacion_materna_y_resiliencia.png"))

    dbscan = dbscan_profile(data).reset_index()
    dbscan["grupo"] = dbscan["cluster_dbscan"].map(lambda value: "Ruido" if value == -1 else f"Cluster {int(value)}")
    figure, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.bar(dbscan["grupo"], dbscan["estudiantes"], color=COLORS[: len(dbscan)])
    axis.bar_label(bars, labels=[f"{int(value):,}".replace(",", ".") for value in dbscan["estudiantes"]], padding=4)
    axis.set(title="Distribución de registros en DBSCAN", xlabel="", ylabel="Estudiantes")
    outputs.append(_save(figure, target, "07_distribucion_dbscan.png"))

    dbscan_sample = data[data["cluster_dbscan"].notna()].sample(n=min(12_000, int(data["cluster_dbscan"].notna().sum())), random_state=42)
    figure, axis = plt.subplots(figsize=(10, 6.5))
    for position, group_name in enumerate(sorted(dbscan_sample["cluster_dbscan"].unique())):
        group = dbscan_sample[dbscan_sample["cluster_dbscan"] == group_name]
        label = "Ruido" if group_name == -1 else f"Cluster {int(group_name)}"
        color = "#94a3b8" if group_name == -1 else COLORS[position % len(COLORS)]
        axis.scatter(group["dbscan_pca_1"], group["dbscan_pca_2"], s=9, alpha=.35, color=color, label=label, edgecolors="none")
    axis.set(title="Proyección PCA de la muestra DBSCAN", xlabel="Componente principal 1", ylabel="Componente principal 2")
    axis.legend(markerscale=2, frameon=False, fontsize=9)
    outputs.append(_save(figure, target, "08_proyeccion_pca_dbscan.png"))

    figure, axis = plt.subplots(figsize=(10, 3.2))
    axis.axis("off")
    metrics = summary
    table = axis.table(
        cellText=[
            ["K-Means", metrics["kmeans"]["clusters"], metrics["kmeans"]["pca_components"], f"{metrics['kmeans']['metrics']['silhouette']:.3f}", f"{metrics['kmeans']['metrics']['davies_bouldin']:.3f}", f"{metrics['kmeans']['metrics']['calinski_harabasz']:.0f}"],
            ["DBSCAN", metrics["dbscan"]["clusters"], metrics["dbscan"]["pca_components"], f"{metrics['dbscan']['metrics']['silhouette']:.3f}", f"{metrics['dbscan']['metrics']['davies_bouldin']:.3f}", f"{metrics['dbscan']['metrics']['calinski_harabasz']:.0f}"],
        ],
        colLabels=["Modelo", "Clusters", "Componentes PCA", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz"],
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.8)
    axis.set_title("Resumen de métricas internas de clustering", pad=18, weight="bold")
    outputs.append(_save(figure, target, "09_metricas_internas_modelos.png"))
    return outputs


def main() -> None:
    """Load the saved clustering results and export the report figures."""
    paths = default_paths()
    if not paths["cluster_data"].exists() or not paths["summary"].exists():
        raise FileNotFoundError(
            "Faltan resultados del modelado. Ejecute primero: python -m src.models.train_model"
        )
    data = load_cluster_results(paths["cluster_data"])
    summary = pd.read_json(paths["summary"], typ="series").to_dict()
    output_dir = paths["root"] / "reports/figures"
    for output in export_static_figures(data, summary, output_dir):
        print(output)


if __name__ == "__main__":
    main()
