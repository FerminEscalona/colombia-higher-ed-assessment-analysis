"""Pruebas de reproducibilidad, validación y arranque del dashboard."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.data.make_dataset import default_paths
from src.models.predict_model import (
    load_model_bundle,
    predict_kmeans,
    prediction_template,
)


class PredictionPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.paths = default_paths()
        cls.bundle = load_model_bundle(cls.paths["bundle"])

    def test_training_rows_keep_their_cluster(self) -> None:
        training = pd.read_csv(cls_path := self.paths["cluster_data"], nrows=50)
        self.assertTrue(cls_path.exists())
        predicted = predict_kmeans(training, self.bundle)
        self.assertTrue((predicted["estado_prediccion"] == "ASIGNADO").all())
        self.assertListEqual(
            predicted["cluster_kmeans"].astype(int).tolist(),
            training["cluster_kmeans"].astype(int).tolist(),
        )

    def test_unknown_nominal_category_uses_other(self) -> None:
        template = prediction_template(self.bundle)
        template.loc[0, "fami_ocupacionmadre"] = "CATEGORÍA NUEVA"
        predicted = predict_kmeans(template, self.bundle)
        self.assertEqual(predicted.loc[0, "estado_prediccion"], "ASIGNADO")

    def test_outside_population_is_not_assigned(self) -> None:
        template = prediction_template(self.bundle)
        template.loc[0, "estu_inse_individual"] = self.bundle["inse_threshold"] + 1
        predicted = predict_kmeans(template, self.bundle)
        self.assertTrue(pd.isna(predicted.loc[0, "cluster_kmeans"]))
        self.assertEqual(
            predicted.loc[0, "estado_prediccion"],
            "FUERA DE LA POBLACIÓN DE BAJO INSE",
        )

    def test_missing_required_column_fails_clearly(self) -> None:
        template = prediction_template(self.bundle).drop(columns=["fami_tieneinternet"])
        with self.assertRaisesRegex(ValueError, "Faltan columnas obligatorias"):
            predict_kmeans(template, self.bundle)


class DashboardSmokeTest(unittest.TestCase):
    def test_all_pages_start_without_exceptions(self) -> None:
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).resolve().parents[1] / "src/dashboard/app.py"
        app = AppTest.from_file(str(app_path), default_timeout=30).run()
        for page in app.radio[0].options:
            app.radio[0].set_value(page)
            app.run()
            self.assertEqual(list(app.exception), [], msg=f"Falló la página {page}")


if __name__ == "__main__":
    unittest.main()
