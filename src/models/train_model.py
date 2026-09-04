"""Entrenamiento reproducible de los modelos de clustering de Saber Pro."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from src.data.make_dataset import default_paths, load_clean_data
from src.features.build_features import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    ORDINAL_COLUMNS,
    PROFILING_COLUMNS,
    fit_feature_transformer,
    required_training_columns,
    select_model_population,
)
from src.models.evaluate_model import internal_metrics, summary_payload


def _python_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"No se puede serializar {type(value)!r}")


def _select_kmeans(matrix: np.ndarray, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    rows: list[dict[str, float | int]] = []
    for retained_variance in [0.70, 0.80, 0.85, 0.90]:
        pca = PCA(n_components=retained_variance, random_state=random_state)
        reduced = pca.fit_transform(matrix).astype("float32")
        for clusters in range(3, 11):
            model = MiniBatchKMeans(
                n_clusters=clusters,
                n_init=10,
                batch_size=4096,
                max_iter=200,
                random_state=random_state,
            )
            labels = model.fit_predict(reduced)
            rows.append(
                {
                    "varianza_pca": retained_variance,
                    "componentes": reduced.shape[1],
                    "k": clusters,
                    "silhouette": silhouette_score(
                        reduced,
                        labels,
                        sample_size=5_000,
                        random_state=random_state,
                    ),
                    "davies_bouldin": davies_bouldin_score(reduced, labels),
                    "cluster_mayor_pct": pd.Series(labels)
                    .value_counts(normalize=True)
                    .max()
                    * 100,
                }
            )
    search = pd.DataFrame(rows)
    candidates = search[search["cluster_mayor_pct"] <= 40]
    best = candidates.sort_values(
        ["silhouette", "davies_bouldin"], ascending=[False, True]
    ).iloc[0]
    return search, best


def _select_dbscan(
    matrix: np.ndarray, random_state: int
) -> tuple[PCA, np.ndarray, pd.DataFrame, pd.Series]:
    pca = PCA(n_components=3, whiten=True, random_state=random_state)
    reduced = pca.fit_transform(matrix).astype("float32")
    index = np.random.default_rng(random_state).choice(
        len(reduced), size=min(30_000, len(reduced)), replace=False
    )
    sample = reduced[index]
    rows: list[dict[str, float | int]] = []

    for minimum in [30, 40, 50]:
        distances = (
            NearestNeighbors(n_neighbors=minimum, n_jobs=-1)
            .fit(sample)
            .kneighbors(sample)[0][:, -1]
        )
        for quantile in [0.90, 0.92, 0.94, 0.96]:
            eps = float(np.quantile(distances, quantile))
            labels = DBSCAN(eps=eps, min_samples=minimum, n_jobs=-1).fit_predict(
                sample
            )
            grouped = labels != -1
            cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
            rows.append(
                {
                    "min_samples": minimum,
                    "percentil_distancia": quantile,
                    "eps": eps,
                    "clusters": cluster_count,
                    "ruido_pct": (~grouped).mean() * 100,
                    "cluster_mayor_pct": pd.Series(labels[grouped])
                    .value_counts(normalize=True)
                    .max()
                    * 100,
                    "silhouette": silhouette_score(
                        sample[grouped],
                        labels[grouped],
                        sample_size=min(5_000, int(grouped.sum())),
                        random_state=random_state,
                    )
                    if cluster_count > 1
                    else np.nan,
                    "davies_bouldin": davies_bouldin_score(
                        sample[grouped], labels[grouped]
                    )
                    if cluster_count > 1
                    else np.nan,
                }
            )
    search = pd.DataFrame(rows)
    candidates = search[
        search["clusters"].between(3, 12)
        & search["ruido_pct"].between(2, 15)
        & (search["cluster_mayor_pct"] < 65)
    ]
    best = candidates.sort_values(
        ["silhouette", "davies_bouldin"], ascending=[False, True]
    ).iloc[0]
    return pca, index, search, best


def train_clustering_pipeline(
    clean_path: str | Path,
    cluster_output_path: str | Path,
    bundle_path: str | Path,
    summary_path: str | Path,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train, evaluate and persist the complete clustering workflow."""
    clean_path = Path(clean_path)
    cluster_output_path = Path(cluster_output_path)
    bundle_path = Path(bundle_path)
    summary_path = Path(summary_path)
    data = load_clean_data(clean_path, usecols=required_training_columns())
    model_data, inse_threshold = select_model_population(data)
    matrix, feature_metadata, rare_summary = fit_feature_transformer(model_data)

    kmeans_search, best_kmeans = _select_kmeans(matrix, random_state)
    pca_kmeans = PCA(
        n_components=float(best_kmeans["varianza_pca"]),
        random_state=random_state,
    )
    kmeans_matrix = pca_kmeans.fit_transform(matrix).astype("float32")
    kmeans = KMeans(
        n_clusters=int(best_kmeans["k"]),
        n_init=30,
        max_iter=500,
        random_state=random_state,
    )
    model_data["cluster_kmeans"] = kmeans.fit_predict(kmeans_matrix)
    k_metrics = internal_metrics(
        kmeans_matrix, model_data["cluster_kmeans"].to_numpy(), random_state=random_state
    )

    pca_dbscan, dbscan_index, dbscan_search, best_dbscan = _select_dbscan(
        matrix, random_state
    )
    dbscan_matrix = pca_dbscan.transform(matrix).astype("float32")
    dbscan_sample = dbscan_matrix[dbscan_index]
    dbscan = DBSCAN(
        eps=float(best_dbscan["eps"]),
        min_samples=int(best_dbscan["min_samples"]),
        n_jobs=-1,
    )
    dbscan_labels = dbscan.fit_predict(dbscan_sample)
    model_data["cluster_dbscan"] = np.nan
    model_data.loc[dbscan_index, "cluster_dbscan"] = dbscan_labels
    grouped = dbscan_labels != -1
    kmeans_columns = [f"pca_{number + 1}" for number in range(kmeans_matrix.shape[1])]
    dbscan_columns = [f"dbscan_pca_{number + 1}" for number in range(3)]
    kmeans_frame = pd.DataFrame(kmeans_matrix, columns=kmeans_columns)
    kmeans_frame["pca_plot_1"] = kmeans_matrix[:, 0]
    kmeans_frame["pca_plot_2"] = kmeans_matrix[:, 1]
    dbscan_frame = pd.DataFrame(dbscan_matrix, columns=dbscan_columns)
    export_columns = (
        [
            "estu_consecutivo",
            "cluster_kmeans",
            "cluster_dbscan",
            "resiliencia_flag",
            "punt_global",
            "percentil_global",
            "estu_inse_individual",
        ]
        + ORDINAL_COLUMNS
        + BINARY_COLUMNS
        + CATEGORICAL_COLUMNS
        + PROFILING_COLUMNS
    )
    model_data["indice_bienes_hogar"] = (
        model_data[BINARY_COLUMNS]
        .apply(lambda column: column.map({"Si": 1, "No": 0}))
        .sum(axis=1)
    )
    export_columns.insert(7, "indice_bienes_hogar")
    exported = pd.concat(
        [model_data[export_columns].reset_index(drop=True), kmeans_frame, dbscan_frame],
        axis=1,
    )
    canonical_dbscan = exported[
        exported["cluster_dbscan"].notna() & exported["cluster_dbscan"].ne(-1)
    ]
    d_metrics = internal_metrics(
        canonical_dbscan[dbscan_columns].to_numpy(dtype="float32"),
        canonical_dbscan["cluster_dbscan"].astype(int).to_numpy(),
        random_state=random_state,
    )

    model_info = {
        "inse_threshold": inse_threshold,
        "kmeans": {
            "clusters": int(kmeans.n_clusters),
            "pca_components": int(kmeans_matrix.shape[1]),
            "explained_variance": float(pca_kmeans.explained_variance_ratio_.sum()),
            "n_init": 30,
        },
        "dbscan": {
            "sample_size": int(len(dbscan_index)),
            "clusters": int(len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)),
            "pca_components": 3,
            "eps": float(dbscan.eps),
            "min_samples": int(dbscan.min_samples),
            "noise_rate": float((dbscan_labels == -1).mean() * 100),
        },
    }
    summary = summary_payload(exported, k_metrics, d_metrics, model_info)
    summary["training"] = {
        "random_state": random_state,
        "rare_threshold": feature_metadata["rare_threshold"],
        "feature_count": len(feature_metadata["feature_names"]),
        "sklearn_version": sklearn.__version__,
        "source_file": clean_path.name,
    }

    bundle = {
        "schema_version": "1.0",
        "random_state": random_state,
        "inse_threshold": inse_threshold,
        "feature_metadata": feature_metadata,
        "pca_kmeans": pca_kmeans,
        "kmeans": kmeans,
        "pca_dbscan": pca_dbscan,
        "dbscan": dbscan,
        "dbscan_training_indices": dbscan_index,
        "cluster_names": summary["kmeans"]["cluster_names"],
        "metrics": {"kmeans": k_metrics, "dbscan": d_metrics},
    }

    cluster_output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    exported.to_csv(cluster_output_path, index=False, encoding="utf-8-sig")
    joblib.dump(bundle, bundle_path, compress=3)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_python_value) + "\n",
        encoding="utf-8",
    )
    return {
        "data": exported,
        "bundle": bundle,
        "summary": summary,
        "rare_summary": rare_summary,
        "kmeans_search": kmeans_search,
        "dbscan_search": dbscan_search,
        "best_kmeans": best_kmeans,
        "best_dbscan": best_dbscan,
        "paths": {
            "cluster_data": cluster_output_path,
            "bundle": bundle_path,
            "summary": summary_path,
        },
    }


def main() -> None:
    paths = default_paths()
    result = train_clustering_pipeline(
        paths["clean_data"], paths["cluster_data"], paths["bundle"], paths["summary"]
    )
    print(f"Estudiantes modelados: {len(result['data']):,}")
    print(f"Clusters K-Means: {result['summary']['kmeans']['clusters']}")
    print(f"Clusters DBSCAN: {result['summary']['dbscan']['clusters']}")
    print(f"Bundle: {result['paths']['bundle']}")


if __name__ == "__main__":
    main()
