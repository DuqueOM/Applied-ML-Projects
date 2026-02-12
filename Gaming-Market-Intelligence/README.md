# 🎮 Gaming Market Intelligence

**Video Game Market Analysis & Commercial Success Prediction System**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Coverage](https://img.shields.io/badge/Coverage-70%25-green.svg?style=flat-square)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

> Analyze 36 years of video game sales data (1980–2016, 16,700+ titles) to identify market trends, test statistical hypotheses, and predict commercial success — served via FastAPI with Docker support.

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python main.py --mode train --config configs/config.yaml --seed 42
python main.py --mode eval  --config configs/config.yaml
python main.py --mode predict --config configs/config.yaml \
    --payload '{"platform":"PS4","genre":"Action","year_of_release":2015,"critic_score":85,"user_score":8.2,"rating":"M"}'
```

---

## 🎯 Problem & Solution

**Problem**: Ice (online game store) needs to identify success patterns to plan advertising campaigns and stock for 2017.

**Solution**:
- ✅ Historical sales analysis (1980–2016, 16,700+ titles)
- ✅ Platform and genre success identification
- ✅ Regional market segmentation (NA, EU, JP)
- ✅ Statistical hypothesis testing (t-test, Mann-Whitney)
- ✅ Commercial success prediction (>1M global sales)

---

## 🔧 Tech Stack

| Layer | Technologies |
|-------|-------------|
| **ML** | Scikit-learn (Random Forest Classifier) |
| **Stats** | SciPy (t-test, Mann-Whitney U) |
| **Data** | Pandas, NumPy |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Ops** | Docker, Docker Compose, Makefile |
| **Quality** | pytest, Mypy, Black |
| **Monitoring** | Drift detection (Evidently-based) |

---

## 💻 Installation

```bash
cd Gaming-Market-Intelligence
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## 📖 Usage

### CLI

```bash
# Train classifier
python main.py --mode train --config configs/config.yaml --seed 42

# Evaluate on test set
python main.py --mode eval --config configs/config.yaml

# Predict success for a new game
python main.py --mode predict --config configs/config.yaml \
    --payload '{"platform":"PS4","genre":"Action","year_of_release":2015,"critic_score":85,"user_score":8.2,"rating":"M"}'
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
make train        # Train model
make eval         # Evaluate
make api          # Start FastAPI server
make check-drift  # Run drift detection
make clean        # Remove artifacts
```

---

## 🎓 Analysis

### Top Platforms (2014–2016)

| Platform | Global Sales | Titles | Avg Rating |
|----------|-------------|--------|------------|
| **PS4** | 385M | 342 | 7.2 |
| **XOne** | 245M | 287 | 7.0 |
| **PC** | 189M | 412 | 6.8 |

### Top Genres

1. **Action** — 35% market share
2. **Sports** — 18%
3. **Shooter** — 15%
4. **Role-Playing** — 12%

### Regional Insights

| Region | Preferred Genres | Leading Platform |
|--------|-----------------|-----------------|
| **North America** | Action, Shooter, Sports | Xbox One |
| **Europe** | Action, Sports, Racing | PS4 |
| **Japan** | Role-Playing, Action, Platform | 3DS |

### Hypotheses Tested

| Hypothesis | p-value | Result |
|-----------|---------|--------|
| Xbox One vs PC avg ratings are equal | 0.23 | Not significant — fail to reject H₀ |
| Action vs Sports avg ratings are equal | 0.04 | **Significant** — reject H₀ ✅ |

---

## 📊 Model

### Algorithm: Random Forest Classifier

**Target**: Binary — commercial success (>1M global sales)

**Features**: platform, year_of_release, genre, critic_score, user_score, rating

### Dataset
- **Source**: Historical game sales data
- **Records**: ~16,700 titles
- **Period**: 1980–2016
- **Features**: Platform, genre, publisher, ESRB rating, regional sales

---

## 📁 Project Structure

```
Gaming-Market-Intelligence/
├── main.py                    # CLI (train / eval / predict)
├── evaluate.py                # Standalone evaluation
├── evaluate_business.py       # Business metric evaluation
├── app/
│   ├── fastapi_app.py         # REST API with /predict + /health
│   └── example_load.py        # Demo script
├── data/
│   ├── raw/                   # Source CSVs (not tracked — see data_card.md)
│   └── preprocess.py          # Feature engineering pipeline
├── configs/config.yaml        # Hyperparameters & feature config
├── tests/                     # pytest suite
├── monitoring/check_drift.py  # Data drift detection
├── notebooks/                 # EDA, statistical tests, presentation
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

## 📈 2017 Predictions

- **Platform #1**: PS4 (continued dominance)
- **Emerging genre**: Battle Royale
- **Rating impact**: M-rated games +15% in sales

---

## 📄 License

MIT License — See [LICENSE](LICENSE)

**Author**: [Duque Ortega Mutis (DuqueOM)](https://github.com/DuqueOM)
