"""Construcción reproducible de variables para los modelos de clustering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

OTHER_CATEGORY = "OTRAS CATEGORÍAS"

ORDINAL_COLUMNS = [
    "fami_estratovivienda",
    "fami_educacionmadre",
    "fami_educacionpadre",
]

BINARY_COLUMNS = [
    "fami_tieneautomovil",
    "fami_tienecomputador",
    "fami_tienehornomicroogas",
    "fami_tieneinternet",
    "fami_tienelavadora",
    "fami_tienemotocicleta",
    "fami_tieneserviciotv",
]

CATEGORICAL_COLUMNS = [
    "fami_ocupacionmadre",
    "fami_ocupacionpadre",
    "fami_trabajolabormadre",
    "fami_trabajolaborpadre",
    "estu_tituloobtenidobachiller",
    "estu_actividadrefuerzogeneric",
    "estu_actividadrefuerzoareas",
    "estu_cursoiesexterna",
    "estu_cursoiesapoyoexterno",
    "estu_cursodocentesies",
    "estu_simulacrotipoicfes",
]

PROFILING_COLUMNS = [
    "estu_metodo_prgm",
    "estu_nivel_prgm_academico",
    "estu_nucleo_pregrado",
    "inst_caracter_academico",
    "inst_origen",
    "estu_prgm_departamento",
    "estu_prgm_academico",
    "estu_snies_prgmacademico",
]

MODEL_INPUT_COLUMNS = ORDINAL_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS

ESTRATO_MAP = {"Sin Estrato": 0, "Sin estrato": 0}
ESTRATO_MAP.update({f"Estrato {number}": number for number in range(1, 7)})

EDUCATION_ORDER = [
    "Ninguno",
    "Primaria incompleta",
    "Primaria completa",
    "Secundaria (Bachillerato) incompleta",
    "Secundaria (Bachillerato) completa",
    "Técnica o tecnológica incompleta",
    "Técnica o tecnológica completa",
    "Educación profesional incompleta",
    "Educación profesional completa",
    "Postgrado",
]
EDUCATION_MAP = {category: position for position, category in enumerate(EDUCATION_ORDER)}
EDUCATION_MAP.update({"No sabe": 0, "No Aplica": 0})


def required_training_columns() -> list[str]:
    """Return the cleaned-data columns required by the clustering pipeline."""
    return list(
        dict.fromkeys(
            [
                "estu_consecutivo",
                "estu_estudiante",
                "estado_contexto_socioeconomico",
                "punt_global",
                "percentil_global",
                "estu_inse_individual",
            ]
            + MODEL_INPUT_COLUMNS
            + PROFILING_COLUMNS
        )
    )


def select_model_population(
    data: pd.DataFrame, inse_threshold: float | None = None
) -> tuple[pd.DataFrame, float]:
    """Select valid students in the lowest INSE quartile."""
    valid = (
        data["estu_estudiante"].eq("ESTUDIANTE")
        & data["estado_contexto_socioeconomico"].eq("DISPONIBLE")
        & data["punt_global"].ne(-1)
        & data["percentil_global"].ne(-1)
    )
    valid_data = data.loc[valid].copy()
    threshold = float(
        valid_data["estu_inse_individual"].quantile(0.25)
        if inse_threshold is None
        else inse_threshold
    )
    selected = valid_data.loc[
        valid_data["estu_inse_individual"] <= threshold
    ].copy()
    selected = selected.reset_index(drop=True)
    selected["resiliencia_flag"] = selected["percentil_global"] >= 90
    return selected, threshold


def _ordinal_frame(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "estrato_ordinal": data["fami_estratovivienda"].map(ESTRATO_MAP),
            "educacion_madre_ordinal": data["fami_educacionmadre"].map(
                EDUCATION_MAP
            ),
            "educacion_padre_ordinal": data["fami_educacionpadre"].map(
                EDUCATION_MAP
            ),
        },
        index=data.index,
    )


def fit_feature_transformer(
    data: pd.DataFrame, rare_threshold: float = 0.005
) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
    """Fit preprocessing rules and return the model matrix and metadata."""
    missing = [column for column in MODEL_INPUT_COLUMNS if column not in data]
    if missing:
        raise ValueError(f"Faltan columnas para construir variables: {missing}")

    ordinal = _ordinal_frame(data)
    if ordinal.isna().any().any():
        invalid = ordinal.columns[ordinal.isna().any()].tolist()
        raise ValueError(f"Hay categorías ordinales desconocidas en: {invalid}")

    ordinal_scaler = StandardScaler()
    ordinal_scaled = pd.DataFrame(
        ordinal_scaler.fit_transform(ordinal),
        columns=ordinal.columns,
        index=data.index,
    )

    binary = data[BINARY_COLUMNS].apply(lambda column: column.map({"Si": 1, "No": 0}))
    if binary.isna().any().any():
        invalid = binary.columns[binary.isna().any()].tolist()
        raise ValueError(f"Hay valores binarios desconocidos en: {invalid}")
    binary = binary.astype("float32")

    grouped = data[CATEGORICAL_COLUMNS].astype("string").copy()
    rare_categories: dict[str, list[str]] = {}
    known_categories: dict[str, list[str]] = {}
    rare_rows: list[dict[str, Any]] = []

    for column in CATEGORICAL_COLUMNS:
        frequencies = grouped[column].value_counts(normalize=True)
        rare_values = frequencies[frequencies < rare_threshold].index.astype(str).tolist()
        rare_categories[column] = rare_values
        affected = int(grouped[column].isin(rare_values).sum())
        grouped.loc[grouped[column].isin(rare_values), column] = OTHER_CATEGORY
        categories = sorted(grouped[column].dropna().astype(str).unique().tolist())
        if OTHER_CATEGORY not in categories:
            categories.append(OTHER_CATEGORY)
        known_categories[column] = categories
        grouped[column] = pd.Categorical(grouped[column], categories=categories)
        rare_rows.append(
            {
                "variable": column,
                "categorias_agrupadas": len(rare_values),
                "registros_afectados": affected,
            }
        )

    one_hot = pd.get_dummies(
        grouped, prefix=CATEGORICAL_COLUMNS, dtype="float32"
    )
    matrix = pd.concat([ordinal_scaled, binary, one_hot], axis=1).astype("float32")

    metadata: dict[str, Any] = {
        "ordinal_scaler": ordinal_scaler,
        "rare_threshold": rare_threshold,
        "rare_categories": rare_categories,
        "known_categories": known_categories,
        "feature_names": matrix.columns.tolist(),
        "model_input_columns": MODEL_INPUT_COLUMNS,
        "ordinal_columns": ORDINAL_COLUMNS,
        "binary_columns": BINARY_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
    }
    return matrix.to_numpy(dtype="float32"), metadata, pd.DataFrame(rare_rows)


def validate_prediction_rows(
    data: pd.DataFrame, inse_threshold: float
) -> tuple[pd.Series, pd.Series]:
    """Validate rows for K-Means inference and return mask plus status text."""
    required = ["estu_inse_individual"] + MODEL_INPUT_COLUMNS
    missing_columns = [column for column in required if column not in data]
    if missing_columns:
        raise ValueError(f"Faltan columnas obligatorias: {missing_columns}")

    valid = pd.Series(True, index=data.index)
    status = pd.Series("LISTO", index=data.index, dtype="string")
    missing_values = data[required].isna().any(axis=1)
    valid.loc[missing_values] = False
    status.loc[missing_values] = "DATOS FALTANTES"

    inse = pd.to_numeric(data["estu_inse_individual"], errors="coerce")
    invalid_inse = inse.isna() & ~missing_values
    valid.loc[invalid_inse] = False
    status.loc[invalid_inse] = "INSE NO NUMÉRICO"
    outside = (inse > inse_threshold) & valid
    valid.loc[outside] = False
    status.loc[outside] = "FUERA DE LA POBLACIÓN DE BAJO INSE"

    invalid_ordinal = _ordinal_frame(data).isna().any(axis=1) & valid
    valid.loc[invalid_ordinal] = False
    status.loc[invalid_ordinal] = "CATEGORÍA ORDINAL DESCONOCIDA"

    invalid_binary = ~data[BINARY_COLUMNS].isin(["Si", "No"]).all(axis=1) & valid
    valid.loc[invalid_binary] = False
    status.loc[invalid_binary] = "VALOR BINARIO DISTINTO DE Si/No"
    return valid, status


def transform_features(data: pd.DataFrame, metadata: dict[str, Any]) -> np.ndarray:
    """Apply fitted preprocessing rules to valid inference rows."""
    ordinal = _ordinal_frame(data)
    ordinal_scaled = pd.DataFrame(
        metadata["ordinal_scaler"].transform(ordinal),
        columns=ordinal.columns,
        index=data.index,
    )
    binary = data[BINARY_COLUMNS].apply(lambda column: column.map({"Si": 1, "No": 0}))
    binary = binary.astype("float32")

    grouped = data[CATEGORICAL_COLUMNS].astype("string").copy()
    for column in CATEGORICAL_COLUMNS:
        allowed = set(metadata["known_categories"][column])
        rare = set(metadata["rare_categories"][column])
        replace_mask = grouped[column].isin(rare) | ~grouped[column].isin(allowed)
        grouped.loc[replace_mask, column] = OTHER_CATEGORY
        grouped[column] = pd.Categorical(
            grouped[column], categories=metadata["known_categories"][column]
        )

    one_hot = pd.get_dummies(grouped, prefix=CATEGORICAL_COLUMNS, dtype="float32")
    matrix = pd.concat([ordinal_scaled, binary, one_hot], axis=1)
    matrix = matrix.reindex(columns=metadata["feature_names"], fill_value=0)
    return matrix.to_numpy(dtype="float32")
