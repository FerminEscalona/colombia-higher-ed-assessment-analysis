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
