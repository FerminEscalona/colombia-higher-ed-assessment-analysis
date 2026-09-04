"""Carga de artefactos y asignación de perfiles con K-Means."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import (
    MODEL_INPUT_COLUMNS,
    transform_features,
    validate_prediction_rows,
)


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the serialized model bundle."""
    bundle = joblib.load(path)
    required = {
        "inse_threshold",
        "feature_metadata",
        "pca_kmeans",
        "kmeans",
        "cluster_names",
    }
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"El bundle no contiene: {sorted(missing)}")
    return bundle


def predict_kmeans(data: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Assign K-Means profiles to eligible rows without storing input data."""
    result = data.copy()
    valid, status = validate_prediction_rows(data, bundle["inse_threshold"])
    result["estado_prediccion"] = status
    result["cluster_kmeans"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["nombre_perfil"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["distancia_centroide"] = np.nan

    if valid.any():
        valid_data = data.loc[valid, ["estu_inse_individual"] + MODEL_INPUT_COLUMNS]
        matrix = transform_features(valid_data, bundle["feature_metadata"])
        reduced = bundle["pca_kmeans"].transform(matrix).astype("float32")
        labels = bundle["kmeans"].predict(reduced)
        distances = bundle["kmeans"].transform(reduced)
        names = bundle["cluster_names"]
        result.loc[valid, "cluster_kmeans"] = labels
        result.loc[valid, "nombre_perfil"] = [names[str(int(label))] for label in labels]
        result.loc[valid, "distancia_centroide"] = distances.min(axis=1)
        result.loc[valid, "estado_prediccion"] = "ASIGNADO"
    return result


def prediction_template(bundle: dict[str, Any]) -> pd.DataFrame:
    """Create a one-row input template using known training categories."""
    metadata = bundle["feature_metadata"]
    row: dict[str, Any] = {"estu_inse_individual": bundle["inse_threshold"]}
    row.update(
        {
            "fami_estratovivienda": "Estrato 1",
            "fami_educacionmadre": "Secundaria (Bachillerato) completa",
            "fami_educacionpadre": "Secundaria (Bachillerato) completa",
        }
    )
    for column in metadata["binary_columns"]:
        row[column] = "No"
    for column in metadata["categorical_columns"]:
        categories = [
            value
            for value in metadata["known_categories"][column]
            if value != "OTRAS CATEGORÍAS"
        ]
        row[column] = categories[0] if categories else "OTRAS CATEGORÍAS"
    return pd.DataFrame([row])
