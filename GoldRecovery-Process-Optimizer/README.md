# ⚙️ GoldRecovery Process Optimizer

**Industrial Process Optimization for Gold Recovery Prediction using Multi-Model Ensemble**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-F7931E?style=flat-square)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-9ACD32?style=flat-square)](https://lightgbm.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Coverage](https://img.shields.io/badge/Coverage-70%25-green.svg?style=flat-square)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

> Predict gold recovery rates across rougher and final purification stages using a weighted ensemble (XGBoost + LightGBM + Random Forest) with custom sMAPE metric — served via FastAPI with Streamlit dashboard.

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python main.py --mode train --config configs/config.yaml
python main.py --mode eval  --config configs/config.yaml
python main.py --mode predict --config configs/config.yaml \
    --input data/raw/gold_recovery_test.csv --output results/predictions.csv
```

---

## 🎯 Problem & Solution

**Problem**: Zyfra develops industrial efficiency solutions. They need to **predict gold recovery coefficients** to optimize the metallurgical process and avoid unprofitable operating parameters.

**Solution**:
- ✅ Multi-target regression (rougher + final recovery)
- ✅ Custom metric: **sMAPE** (Symmetric Mean Absolute Percentage Error)
- ✅ Weighted ensemble of 3 models (XGBoost 40%, LightGBM 35%, RF 25%)
- ✅ Feature engineering from ~40 process parameters
- ✅ Validation with real production data

---

## 🔧 Tech Stack

| Layer | Technologies |
|-------|-------------|
| **ML** | XGBoost, LightGBM, Scikit-learn (Random Forest) |
| **Optimization** | Optuna (hyperparameter tuning) |
| **Data** | Pandas, NumPy |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Dashboard** | Streamlit |
| **Ops** | Docker, Docker Compose, Makefile |
| **Quality** | pytest, Mypy, Black |
| **Monitoring** | Drift detection (Evidently-based) |

---

## 💻 Installation

```bash
cd GoldRecovery-Process-Optimizer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## � Usage

### CLI

```bash
# Train ensemble model
python main.py --mode train --config configs/config.yaml

# Evaluate on test set
python main.py --mode eval --config configs/config.yaml

# Batch predictions
python main.py --mode predict --config configs/config.yaml \
    --input data/raw/gold_recovery_test.csv --output results/predictions.csv

# Process optimization (demo)
python main.py --mode optimize --config configs/config.yaml

# Monitoring dashboard
python main.py --mode monitor --dashboard --port 8501
```

### FastAPI

```bash
make api   # http://localhost:8000/docs
```

### Streamlit Dashboard

```bash
streamlit run app/streamlit_dashboard.py   # http://localhost:8501
```

### Docker

```bash
docker-compose up --build   # API at http://localhost:8000
```

### Makefile

```bash
make install      # Install dependencies
make train        # Train ensemble
make eval         # Evaluate
make api          # Start FastAPI server
make check-drift  # Run drift detection
make clean        # Remove artifacts
```

---

## 🎓 Model

### Algorithm: Weighted Multi-Model Ensemble

**Approach**: Two independent ensembles for rougher and final recovery stages.

Each ensemble combines:
- **XGBoost** (weight: 0.40) — 500 estimators, depth 8
- **LightGBM** (weight: 0.35) — 500 estimators, depth 8
- **Random Forest** (weight: 0.25) — 300 estimators, depth 15

### Key Features
- Au, Ag, Pb concentrations (gold, silver, lead)
- Flotation parameters
- Volumes and flow rates
- Feed size / granulometry
- Derived: recovery ratios between stages, temporal rolling statistics

### Custom Metric: sMAPE

```
sMAPE = (100/n) × Σ |y_true - y_pred| / (|y_true| + |y_pred|)
```

Designed for industrial processes where values near zero make standard MAPE unreliable.

### Results

| Model | sMAPE Train | sMAPE Test |
|-------|-------------|------------|
| **Rougher Recovery** | 7.2% | 8.5% |
| **Final Recovery** | 6.8% | 9.1% |
| **Combined** | 7.0% | **8.8%** ✅ |

**Target**: sMAPE < 10% — **achieved** ✅

### Dataset
- **Source**: Zyfra — Gold processing plant
- **Records**: ~16,000 observations
- **Features**: ~40 process parameters (concentrations, volumes, temperatures)
- **Targets**:
  - `rougher.output.recovery` — Rougher stage recovery
  - `final.output.recovery` — Final stage recovery

---

## 📁 Project Structure

```
GoldRecovery-Process-Optimizer/
├── main.py                      # CLI (train / eval / predict / optimize / monitor)
├── evaluate.py                  # Standalone evaluation
├── app/
│   ├── fastapi_app.py           # REST API with /predict + /health
│   ├── streamlit_dashboard.py   # Interactive monitoring dashboard
│   └── example_load.py          # Demo script
├── data/
│   ├── raw/                     # Source CSVs (not tracked — see data_card.md)
│   └── preprocess.py            # Data loading & feature engineering
├── configs/config.yaml          # Model hyperparameters & ensemble weights
├── scripts/
│   ├── run_mlflow.py            # MLflow experiment tracking
│   ├── run_optuna.py            # Hyperparameter optimization
│   └── recovery_simulation.py   # Process simulation
├── tests/                       # pytest suite
├── monitoring/check_drift.py    # Data drift detection
├── notebooks/                   # EDA, demos, presentations
├── Dockerfile                   # Container packaging
├── docker-compose.yml           # Local orchestration
├── Makefile                     # Standard targets
├── model_card.md                # Model documentation
└── data_card.md                 # Dataset documentation
```

---

## 🧪 Testing

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## 📈 Key Insights

- **Gold concentration** is the most important feature (45% importance)
- **Air flow** in rougher stage significantly affects recovery
- **Optimal feed size**: 60–80 microns
- Model predicts with **91% accuracy** within acceptable tolerance

---

## 📄 License

MIT License — See [LICENSE](LICENSE)

**Author**: [Duque Ortega Mutis (DuqueOM)](https://github.com/DuqueOM)
