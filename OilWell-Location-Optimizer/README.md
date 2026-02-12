# 🛢️ OilWell Location Optimizer

**Oil Well Selection System with Bootstrap Risk Analysis and Profit Maximization**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Coverage](https://img.shields.io/badge/Coverage-70%25-green.svg?style=flat-square)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

> Select the most profitable drilling locations from 100,000 candidate wells across 3 geological regions using Linear Regression + Bootstrap profit simulation (1,000 iterations) with 95% confidence intervals — served via FastAPI with Docker support.

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python main.py --mode train --config configs/default.yaml --seed 12345
python main.py --mode eval  --config configs/default.yaml
python main.py --mode predict --config configs/default.yaml --region 1 \
    --payload '{"records":[{"f0":1.0,"f1":-2.0,"f2":3.0}]}'
```

---

## 🎯 Problem & Solution

**Problem**: OilyGiant needs to decide where to drill 200 new oil wells across 3 candidate regions, maximizing profit while keeping loss probability below 2.5%.

**Solution**:
- ✅ Regression models to predict reserve volumes per well
- ✅ Bootstrap technique (1,000 iterations) to estimate profit distributions
- ✅ Risk analysis: loss probability < 2.5% threshold
- ✅ Top-200 well selection per region
- ✅ 95% confidence intervals for profit estimates

### Financial Parameters

| Parameter | Value |
|-----------|-------|
| **Total budget** | $100M USD |
| **Wells to develop** | 200 |
| **Cost per well** | $500K |
| **Revenue per barrel** | $4.50 |
| **Maximum tolerable risk** | 2.5% |

---

## 🔧 Tech Stack

| Layer | Technologies |
|-------|-------------|
| **ML** | Scikit-learn (Linear Regression) |
| **Stats** | Bootstrap sampling, Confidence intervals |
| **Data** | Pandas, NumPy |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Ops** | Docker, Docker Compose, Makefile |
| **Quality** | pytest, Mypy, Black |
| **Monitoring** | Drift detection (Evidently-based) |

---

## 💻 Installation

```bash
cd OilWell-Location-Optimizer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## 📖 Usage

### CLI

```bash
# Train models for all 3 regions
python main.py --mode train --config configs/default.yaml --seed 12345

# Bootstrap risk evaluation
python main.py --mode eval --config configs/default.yaml

# Predict reserve volume for new wells
python main.py --mode predict --config configs/default.yaml --region 1 \
    --payload '{"records":[{"f0":1.0,"f1":-2.0,"f2":3.0}]}'
```

### FastAPI

```bash
make api   # http://localhost:8000/docs
```

### Docker

```bash
docker-compose up --build   # API at http://localhost:8000
```

### Makefile

```bash
make install      # Install dependencies
make train        # Train all region models
make eval         # Bootstrap evaluation
make api          # Start FastAPI server
make check-drift  # Run drift detection
make clean        # Remove artifacts
```

---

## 🎓 Methodology

### 1. Predictive Modeling

**Algorithm**: Linear Regression

**Features**: 3 geological parameters per region (f0, f1, f2)
**Target**: Reserve volume (thousands of barrels)

### 2. Bootstrap Profit Simulation

1. Train model on training sample
2. Predict volumes on validation sample
3. Select top 200 wells with highest predicted volumes
4. Calculate total profit
5. Repeat 1,000 times with Bootstrap samples
6. Analyze profit distribution

### 3. Profit Calculation

```python
profit = (total_volume_top200 * revenue_per_barrel) - (n_wells * cost_per_well)
```

---

## 📊 Results by Region

| Metric | Region 0 | Region 1 | Region 2 |
|--------|----------|----------|----------|
| **Expected profit** | $33.2M | $24.8M | $27.1M |
| **95% CI** | [$25.1M, $41.3M] | [$18.3M, $31.2M] | [$19.7M, $34.5M] |
| **Loss probability** | 1.2% ✅ | 0.8% ✅ | 5.2% ❌ |
| **Model RMSE** | 37.5 | 0.89 | 40.1 |
| **Recommendation** | APPROVED | APPROVED | **REJECTED** (risk > 2.5%) |

### Dataset
- **Source**: OilyGiant — Geological survey data
- **Records**: 100,000 wells (3 regions × ~33K each)
- **Features per region**: 3 geological parameters (f0, f1, f2)
- **Target**: Reserve volume (thousands of barrels)

---

## 📁 Project Structure

```
OilWell-Location-Optimizer/
├── main.py                    # CLI (train / eval / predict)
├── evaluate.py                # Bootstrap profit simulation
├── app/
│   ├── fastapi_app.py         # REST API with /predict + /health
│   └── example_load.py        # Demo script
├── data/
│   ├── raw/                   # Source CSVs (not tracked — see data_card.md)
│   └── preprocess.py          # Data loading & cleaning
├── configs/default.yaml       # Hyperparameters & financial params
├── scripts/
│   ├── run_mlflow.py          # MLflow experiment tracking
│   ├── optimize_selection.py  # Well selection optimization
│   └── sensitivity.py         # Sensitivity analysis
├── tests/                     # pytest suite (unit + E2E API tests)
├── monitoring/check_drift.py  # Data drift detection
├── notebooks/                 # EDA, risk pipeline demo
├── Dockerfile                 # Container packaging
├── docker-compose.yml         # Local orchestration
├── Makefile                   # Standard targets
├── model_card.md              # Model documentation
└── data_card.md               # Dataset documentation
```

---

## 🧪 Testing

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## 📈 Final Recommendation

### **Region 1** — Best Choice

- ✅ Lowest loss probability (0.8%)
- ✅ Lowest variability (narrowest CI)
- ✅ Best model accuracy (RMSE = 0.89)
- ✅ Expected profit: $24.8M

**Next steps**:
1. Verify permits and regulations for Region 1
2. Conduct detailed geological surveys on selected 200 wells
3. Plan drilling logistics
4. Monitor actual results vs predictions

---

## 📄 License

MIT License — See [LICENSE](LICENSE)

**Author**: [Duque Ortega Mutis (DuqueOM)](https://github.com/DuqueOM)
