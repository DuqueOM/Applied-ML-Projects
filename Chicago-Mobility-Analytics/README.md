# 🚕 Chicago Mobility Analytics

**Taxi Demand Prediction System with Temporal Feature Engineering and Weather Fusion**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Coverage](https://img.shields.io/badge/Coverage-62%25-yellowgreen.svg?style=flat-square)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

> Predict hourly taxi demand in Chicago by combining temporal patterns, lag features, and weather conditions — served via FastAPI with Docker support.

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python main.py --mode train --config configs/default.yaml --seed 42
python main.py --mode eval  --config configs/default.yaml
python main.py --mode predict --config configs/default.yaml \
    --start_ts "2017-11-11 10:00:00" --weather_conditions Good
```

---

## 🎯 Problem & Solution

**Problem**: Sweet Lift Taxi needs to predict hourly demand to optimize driver allocation during peak hours — especially around airports.

**Solution**:
- ✅ Regression model predicting trips per hour
- ✅ Temporal pattern analysis (hour, day of week, weekend)
- ✅ Feature engineering with lags and rolling statistics
- ✅ Weather condition fusion (Good / Bad)
- ✅ RMSE < 50 trips (85% accuracy)

---

## 🔧 Tech Stack

| Layer | Technologies |
|-------|-------------|
| **ML** | Scikit-learn (Random Forest) |
| **Data** | Pandas, NumPy |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Ops** | Docker, Docker Compose, Makefile |
| **Quality** | pytest, Mypy, Black |
| **Monitoring** | Drift detection (Evidently-based) |

---

## 💻 Installation

```bash
cd Chicago-Mobility-Analytics
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Or with dev dependencies
pip install -e ".[dev]"
```

---

## 📖 Usage

### CLI

```bash
# Train
python main.py --mode train --config configs/default.yaml --seed 42

# Evaluate
python main.py --mode eval --config configs/default.yaml

# Predict single trip
python main.py --mode predict --config configs/default.yaml \
    --start_ts "2017-11-11 10:00:00" --weather_conditions Good
```

### FastAPI

```bash
make api   # http://localhost:8000/docs
```

```bash
curl -X POST http://localhost:8000/predict_duration \
  -H "Content-Type: application/json" \
  -d '{"start_ts": "2017-11-11T10:00:00", "weather_conditions": "Good"}'
```

### Docker

```bash
docker-compose up --build   # API at http://localhost:8000
```

### Makefile

```bash
make install      # Install dependencies
make train        # Train model
make eval         # Evaluate
make api          # Start FastAPI server
make check-drift  # Run drift detection
make clean        # Remove artifacts
```

---

## 🎓 Model

### Algorithm: Random Forest Regressor

**Features**:
- `hour` — Hour of day (0–23)
- `day_of_week` — Day of week (0–6)
- `is_weekend` — Weekend indicator
- `lag_1h`, `lag_24h` — Demand in previous hours
- `rolling_mean_3h` — 3-hour rolling average
- `weather_conditions` — Good / Bad (binary)

### Metrics

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **RMSE** | 48.2 | < 50 ✅ |
| **MAE** | 35.1 | < 40 ✅ |
| **R²** | 0.82 | > 0.75 ✅ |

### Dataset
- **Source**: Sweet Lift Taxi — Chicago
- **Records**: ~26,000 hourly observations
- **Period**: Summer 2017
- **Target**: Number of trips per hour

---

## 📁 Project Structure

```
Chicago-Mobility-Analytics/
├── main.py                    # CLI (train / eval / predict)
├── evaluate.py                # Standalone evaluation
├── app/
│   ├── fastapi_app.py         # REST API with /predict_duration + /health
│   └── example_load.py        # Demo script
├── data/
│   ├── raw/                   # Source CSVs (not tracked — see data_card.md)
│   └── preprocess.py          # Feature engineering pipeline
├── configs/default.yaml       # Hyperparameters
├── tests/                     # pytest suite
├── monitoring/check_drift.py  # Data drift detection
├── notebooks/                 # EDA + geospatial demos
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

## 📈 Key Insights

- **Peak demand**: 18:00–20:00 (+35% above average)
- **Busiest day**: Friday (+28% vs average)
- **Airports**: 40% of trips during peak hours
- **Prediction error**: ±35 trips average

---

## 📄 License

MIT License — See [LICENSE](LICENSE)

**Author**: [Duque Ortega Mutis (DuqueOM)](https://github.com/DuqueOM)
