# 🧠 Applied ML Projects

**Production-oriented Machine Learning projects spanning classification, regression, time series, and risk analysis — each with CLI pipelines, FastAPI serving, Docker packaging, and automated testing.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

---

## 📋 Projects

| Project | Domain | Type | Key Metric | Techniques |
|---------|--------|------|------------|------------|
| **[OilWell Location Optimizer](OilWell-Location-Optimizer/)** | Energy | Regression + Risk | Loss prob < 2.5% | Bootstrap CI, Linear Regression, Financial modeling |
| **[GoldRecovery Process Optimizer](GoldRecovery-Process-Optimizer/)** | Mining / Industrial | Multi-target Regression | sMAPE 8.8% | XGBoost, LightGBM, Ensemble, Custom sMAPE metric |
| **[Chicago Mobility Analytics](Chicago-Mobility-Analytics/)** | Transportation | Time Series Regression | RMSE 48.2 | Random Forest, Temporal features, Weather fusion |
| **[Gaming Market Intelligence](Gaming-Market-Intelligence/)** | Entertainment | Classification + EDA | AUC 0.85 | Random Forest, Hypothesis testing, Regional analysis |

---

## 🏗️ Shared Architecture

Every project follows the same production-oriented structure:

```
<Project>/
├── main.py                  # CLI entry point (train / eval / predict)
├── evaluate.py              # Standalone evaluation script
├── app/
│   └── fastapi_app.py       # REST API with Pydantic schemas + /health
├── data/
│   └── preprocess.py        # Data loading and feature engineering
├── configs/
│   └── default.yaml         # Externalized hyperparameters
├── tests/                   # pytest suite (unit + integration)
├── monitoring/
│   └── check_drift.py       # Data drift detection
├── notebooks/               # EDA, demos, presentations
├── Dockerfile               # Container packaging
├── docker-compose.yml       # Local orchestration
├── Makefile                 # Standard targets (train, eval, api, test)
├── model_card.md            # Model documentation (v2.0 format)
├── data_card.md             # Dataset documentation
└── pyproject.toml           # Modern Python packaging
```

### Common Patterns

- **CLI-first design** — `python main.py --mode train|eval|predict --config configs/default.yaml`
- **Config-driven** — All hyperparameters externalized in YAML
- **Reproducible** — Shared `common_utils.seed` module, deterministic splits
- **Servable** — FastAPI with Pydantic request/response schemas and health checks
- **Containerized** — Single Dockerfile per project, docker-compose for local dev
- **Tested** — pytest with fixtures, data validation, and model sanity checks
- **Monitored** — Drift detection scripts with configurable thresholds

---

## 🚀 Quick Start

### Run Any Project

```bash
# Clone
git clone https://github.com/DuqueOM/Applied-ML-Projects.git
cd Applied-ML-Projects

# Pick a project
cd OilWell-Location-Optimizer

# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train → Evaluate → Serve
make train
make eval
make api        # FastAPI on http://localhost:8000/docs
```

### Docker

```bash
cd Chicago-Mobility-Analytics
docker-compose up --build
# API available at http://localhost:8000/docs
```

---

## 🔧 Tech Stack

| Layer | Technologies |
|-------|-------------|
| **ML** | Scikit-learn, XGBoost, LightGBM, Optuna |
| **Data** | Pandas, NumPy, SciPy |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Quality** | pytest, Black, Flake8, Mypy, Bandit |
| **Ops** | Docker, Docker Compose, Makefile |
| **Tracking** | MLflow (optional per-project) |
| **Monitoring** | Evidently-based drift detection |

---

## 📊 Project Highlights

### OilWell Location Optimizer
Financial risk analysis for $100M oil drilling investment across 3 geological regions. Uses **Bootstrap sampling** (1,000 iterations) to estimate profit distributions and confidence intervals. Recommends Region 1 (loss probability 0.8%, expected profit $24.8M).

### GoldRecovery Process Optimizer
Industrial process optimization predicting gold recovery rates across rougher and final purification stages. **Multi-model ensemble** (XGBoost + LightGBM + Random Forest) with custom **sMAPE metric** designed for industrial process values near zero.

### Chicago Mobility Analytics
Taxi demand prediction system combining temporal patterns with weather data. Features **geospatial analysis** (PostGIS schema included), temporal feature engineering (lags, rolling statistics), and weather condition fusion for 26,000+ hourly observations.

### Gaming Market Intelligence
Video game market analysis spanning 1980–2016 with 16,700+ titles. Includes **statistical hypothesis testing** (t-test, Mann-Whitney), regional market segmentation (NA/EU/JP), and platform lifecycle modeling. Predicts commercial success probability for new releases.

---

## 📦 Data

Each project includes **sample data** (~100 rows) in `data/raw/sample_*.csv` so `make train` works immediately after cloning. For full-scale training, download the original datasets:

| Project | Sample File | Full Dataset Source |
|---------|-------------|-------------------|
| **OilWell Location Optimizer** | `sample_geo_data_0.csv` | [TripleTen Data Science Program](https://tripleten.com/) — 3 geological region CSVs (500 wells each) |
| **GoldRecovery Process Optimizer** | `sample_gold_recovery.csv` | [TripleTen Data Science Program](https://tripleten.com/) — Industrial process telemetry (87 columns, 16K+ rows) |
| **Chicago Mobility Analytics** | `sample_trips.csv` | [Chicago Open Data Portal](https://data.cityofchicago.org/) — Taxi trips + NOAA weather data |
| **Gaming Market Intelligence** | `sample_games.csv` | [TripleTen Data Science Program](https://tripleten.com/) — VGChartz/Metacritic aggregation (16,700+ titles) |

Place downloaded files in the respective `data/raw/` directory and update `configs/*.yaml` paths if needed.

---

## 🧪 Testing

```bash
# Run tests for a specific project
cd OilWell-Location-Optimizer
pytest tests/ -v --cov=. --cov-report=term-missing

# Run all tests (from root)
for d in *-*/; do (cd "$d" && pytest tests/ -v) ; done
```

---

## 📂 Repository Layout

```
Applied-ML-Projects/
├── Chicago-Mobility-Analytics/      # 🚕 Taxi demand prediction
├── Gaming-Market-Intelligence/      # 🎮 Game sales analysis
├── GoldRecovery-Process-Optimizer/  # ⚙️ Industrial process ML
├── OilWell-Location-Optimizer/      # 🛢️ Risk-optimized drilling
├── common_utils/                    # 🔧 Shared reproducibility module
├── .github/workflows/ci.yml        # ⚡ CI pipeline
└── README.md                        # ← You are here
```

---

## 🔧 Development Process

This repository contains applied ML projects originally developed during the TripleTen Data Science program, then **refactored into production-oriented structures** with:

- CLI pipelines replacing notebook-only workflows
- FastAPI serving layers with Pydantic validation
- Docker packaging for reproducible environments
- Automated testing and data validation
- Model and data cards following industry standards

AI-assisted tools (Cursor / Cascade) were used for code generation and boilerplate acceleration. All modeling decisions, feature engineering, evaluation methodology, and business analysis were performed by the author.

---

## 👤 Author

**Duque Ortega Mutis (DuqueOM)**

[![GitHub](https://img.shields.io/badge/GitHub-DuqueOM-181717?style=flat-square&logo=github)](https://github.com/DuqueOM)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-duqueom-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/duqueom)

---

*MIT License — See individual project [LICENSE](OilWell-Location-Optimizer/LICENSE) files.*
