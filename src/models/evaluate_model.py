"""Métricas y perfiles compartidos por notebooks y dashboard."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.features.build_features import EDUCATION_MAP


def internal_metrics(
    matrix: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 10_000,
    random_state: int = 42,
) -> dict[str, float]:
    """Calculate standard internal clustering metrics."""
    size = min(sample_size, len(matrix))
    return {
        "silhouette": float(
            silhouette_score(
                matrix, labels, sample_size=size, random_state=random_state
            )
        ),
        "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
    }


def kmeans_profile(data: pd.DataFrame) -> pd.DataFrame:
    """Create the main K-Means business profile table."""
    profile = data.groupby("cluster_kmeans").agg(
        estudiantes=("cluster_kmeans", "size"),
        puntaje_promedio=("punt_global", "mean"),
        percentil_promedio=("percentil_global", "mean"),
        inse_promedio=("estu_inse_individual", "mean"),
        bienes_promedio=("indice_bienes_hogar", "mean"),
        resilientes=("resiliencia_flag", "sum"),
        tasa_resiliencia=("resiliencia_flag", "mean"),
    )
    profile["porcentaje_poblacion"] = profile["estudiantes"] / len(data) * 100
    profile["tasa_resiliencia"] *= 100
    total_resilient = max(int(data["resiliencia_flag"].sum()), 1)
    profile["participacion_resilientes"] = (
        profile["resilientes"] / total_resilient * 100
    )
    overall_rate = data["resiliencia_flag"].mean() * 100
    profile["indice_concentracion"] = (
        profile["tasa_resiliencia"] / overall_rate if overall_rate > 0 else np.nan
    )
    return profile


def dbscan_profile(data: pd.DataFrame) -> pd.DataFrame:
    """Create the DBSCAN profile table, including label -1 as noise."""
    sample = data[data["cluster_dbscan"].notna()].copy()
    profile = sample.groupby("cluster_dbscan").agg(
        estudiantes=("cluster_dbscan", "size"),
        resilientes=("resiliencia_flag", "sum"),
        tasa_resiliencia=("resiliencia_flag", "mean"),
        puntaje_promedio=("punt_global", "mean"),
        inse_promedio=("estu_inse_individual", "mean"),
        bienes_promedio=("indice_bienes_hogar", "mean"),
    )
    profile["tasa_resiliencia"] *= 100
    return profile


def semantic_cluster_names(data: pd.DataFrame) -> dict[int, str]:
    """Assign stable business names from observed cluster characteristics."""
    clusters = sorted(data["cluster_kmeans"].dropna().astype(int).unique())
    names = {cluster: f"Perfil {cluster}" for cluster in clusters}
    if not clusters:
        return names

    preparation = data["estu_actividadrefuerzogeneric"].ne(
        "NO INFORMADO - BLOQUE NO DILIGENCIADO"
    )
    preparation_rate = preparation.groupby(data["cluster_kmeans"]).mean()
    preparation_cluster = int(preparation_rate.idxmax())
    names[preparation_cluster] = "Preparación académica reportada"

    remaining = [cluster for cluster in clusters if cluster != preparation_cluster]
    if remaining:
        education = (
            data["fami_educacionmadre"].map(EDUCATION_MAP).fillna(0)
            + data["fami_educacionpadre"].map(EDUCATION_MAP).fillna(0)
        ) / 2
        education_mean = education.groupby(data["cluster_kmeans"]).mean()
        education_cluster = int(education_mean.loc[remaining].idxmax())
        names[education_cluster] = "Mayor capital educativo familiar"
        remaining.remove(education_cluster)

    if remaining:
        goods_mean = data.groupby("cluster_kmeans")["indice_bienes_hogar"].mean()
        material_cluster = int(goods_mean.loc[remaining].idxmax())
        names[material_cluster] = "Mayor disponibilidad material"
        remaining.remove(material_cluster)

    for cluster in remaining:
        names[cluster] = "Mayor vulnerabilidad observada"
    return names


def summary_payload(
    data: pd.DataFrame,
    kmeans_metrics: dict[str, float],
    dbscan_metrics: dict[str, float],
    model_info: dict[str, Any],
) -> dict[str, Any]:
    """Build a JSON-serializable summary for the application."""
    k_profile = kmeans_profile(data).reset_index()
    d_profile = dbscan_profile(data).reset_index()
    names = semantic_cluster_names(data)
    k_profile["nombre"] = k_profile["cluster_kmeans"].map(names)
    return {
        "schema_version": "1.0",
        "population": {
            "students": int(len(data)),
            "resilient_students": int(data["resiliencia_flag"].sum()),
            "resilience_rate": float(data["resiliencia_flag"].mean() * 100),
            "average_score": float(data["punt_global"].mean()),
            "inse_threshold": float(model_info["inse_threshold"]),
        },
        "kmeans": {
            **model_info["kmeans"],
            "metrics": kmeans_metrics,
            "cluster_names": {str(key): value for key, value in names.items()},
            "profiles": k_profile.to_dict(orient="records"),
        },
        "dbscan": {
            **model_info["dbscan"],
            "metrics": dbscan_metrics,
            "profiles": d_profile.to_dict(orient="records"),
        },
    }
