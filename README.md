# Colombia Higher Education Assessment Analysis

An end-to-end Data Science and Machine Learning project structured around the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology and Cookiecutter Data Science standards.

---

## 📁 Project Structure

```text
.
├── .gitignore               <- Rules for ignoring files in Git (data, models, env, checkpoints)
├── .env.example             <- Template for environment variables and secrets
├── README.md                <- Top-level project documentation
├── requirements.txt         <- Project dependencies and Python package requirements
├── setup.py                 <- Makes project pip-installable (`pip install -e .`)
├── config/                  <- Centralized configuration files
│   ├── config.yaml          <- Global project configuration
│   ├── data_config.yaml     <- Data pipeline paths and data ingestion parameters
│   └── model_config.yaml    <- Modeling parameters, hyperparameters, and split settings
├── data/                    <- Data storage (git-ignored, except .gitkeep)
│   ├── raw/                 <- Original, immutable raw data dump
│   ├── interim/             <- Intermediate transformed data
│   └── processed/           <- Canonical data sets for modeling
├── notebooks/               <- Jupyter notebooks organized by CRISP-DM phases
│   ├── 1.0-business-understanding.ipynb
│   ├── 2.0-data-understanding.ipynb
│   ├── 3.0-data-prep.ipynb
│   ├── 4.0-modeling.ipynb
│   └── 5.0-evaluation.ipynb
├── src/                     <- Source code for use in this project
│   ├── __init__.py          <- Makes src a Python module
│   ├── data/                <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   ├── features/            <- Scripts to turn raw data into features for modeling
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/              <- Scripts to train models and make predictions
│   │   ├── __init__.py
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── visualization/       <- Scripts to create exploratory and results-oriented visualizations
│       ├── __init__.py
│       └── visualize.py
├── models/                  <- Trained and serialized models, model predictions, or model summaries
├── reports/                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures/             <- Generated graphics and figures to be used in reporting
└── docs/                    <- Documentation files
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- `virtualenv` or `conda`

### 2. Setup Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install source package in editable mode
pip install -e .
```

### 3. Environment Variables
Copy `.env.example` to `.env` and fill in necessary configuration parameters:
```bash
cp .env.example .env
```

## Dashboard de resiliencia académica

La aplicación permite explorar los perfiles de estudiantes con bajo INSE, comparar
la concentración de resiliencia, revisar K-Means y DBSCAN, y aplicar filtros
territoriales, institucionales y académicos.

### 1. Generar datos y modelos

Después de ejecutar la preparación de datos, el pipeline completo se puede lanzar
desde `4.0-modeling.ipynb` o desde la terminal:

```bash
python -m src.models.train_model
```

Este comando genera localmente:

```text
data/processed/saber_pro_2024_clusters_estudiantes.csv
models/saber_pro_clustering_bundle.joblib
models/model_summary.json
```

### 2. Iniciar Streamlit

Desde la raíz del repositorio:

```bash
streamlit run src/dashboard/app.py
```

El dashboard se abrirá normalmente en `http://localhost:8501`.

DBSCAN se utiliza únicamente para análisis descriptivo porque no dispone de una
operación `predict` para observaciones nuevas.

### Código reutilizable

```text
src/
├── data/                 # Carga de datos y rutas
├── features/             # Preparación reproducible de variables
├── models/               # Entrenamiento, evaluación y predicción
├── visualization/        # Gráficas Plotly reutilizables
└── dashboard/
    ├── app.py            # Entrada de Streamlit
    ├── data.py           # Caché y filtros
    ├── ui.py             # Estilos y componentes comunes
    └── views/            # Vistas internas del dashboard
```

### Pruebas

```bash
python -m unittest discover -v
```
