"""Funciones de carga para datos preparados y resultados del proyecto."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def project_root() -> Path:
    """Return the repository root from this module location."""
    return Path(__file__).resolve().parents[2]


def load_clean_data(path: str | Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Load the cleaned Saber Pro dataset."""
    return pd.read_csv(path, usecols=usecols, encoding="utf-8-sig", low_memory=False)


def load_cluster_results(path: str | Path) -> pd.DataFrame:
    """Load the student-level clustering output."""
    data = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if "resiliencia_flag" in data:
        if data["resiliencia_flag"].dtype == "object":
            data["resiliencia_flag"] = data["resiliencia_flag"].map(
                {"True": True, "False": False}
            )
        data["resiliencia_flag"] = data["resiliencia_flag"].astype(bool)
    return data


def default_paths() -> dict[str, Path]:
    """Return canonical local paths used by training and dashboard modules."""
    root = project_root()
    return {
        "root": root,
        "clean_data": root / "data/processed/Examen_Saber_Pro_Genericas_2024_limpio.csv",
        "cluster_data": root / "data/processed/saber_pro_2024_clusters_estudiantes.csv",
        "bundle": root / "models/saber_pro_clustering_bundle.joblib",
        "summary": root / "models/model_summary.json",
    }
