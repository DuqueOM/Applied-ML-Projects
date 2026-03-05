# Applied-ML-Projects — Assessment & Improvement Plan

> Local guide (not for public repo). Created March 2026.
> **Last verified: March 5, 2026** (Python 3.13.5, conda `ml` env)

## Current State Assessment

### Strong Points
- **4 projects** across diverse domains (energy, mining, transportation, gaming)
- **Consistent structure**: CLI pipeline, FastAPI serving, Docker, tests, model/data cards
- **21 test files** across all projects (5 per project + 1 common_utils)
- **4 Dockerfiles** with docker-compose for local orchestration
- **Comprehensive README** with metrics table, tech stack, and quick start
- **CI pipeline** exists (`.github/workflows/ci.yml`)
- **Model cards + Data cards** for all projects

### Assessment Per Project

| Project | Key Metric | Tests | Coverage | Docker | README | Model Card | Data Card |
|---------|-----------|-------|----------|--------|--------|------------|-----------|
| **OilWell** | RMSE 0.88 (region 1) | 28 passed | 63% | Build OK | Comprehensive | Yes | Yes |
| **GoldRecovery** | sMAPE 2.64%, R2 0.954 | 24 passed | 43% | Build OK | Comprehensive | Yes | Yes |
| **Chicago Mobility** | RMSE 520.8 (test) | 28 passed | 62% | Build OK | Comprehensive | Yes | Yes |
| **Gaming Market** | AUC 0.834 | 23 passed | 49% | Build OK | Comprehensive | Yes | Yes |

### Verdict: This repo is well-positioned. All checks verified below.

---

## Verification Results (March 5, 2026)

### Priority 1: Tests — ALL PASS

| Project | Tests | Coverage | Warnings | Notes |
|---------|-------|----------|----------|-------|
| OilWell-Location-Optimizer | 28 passed | 62.53% | 2 (PydanticV2 deprecation, unknown mark `slow`) | All pass |
| GoldRecovery-Process-Optimizer | 24 passed | 42.65% | 36 (LightGBM no-further-splits) | All pass |
| Chicago-Mobility-Analytics | 28 passed | 62.00% | 1 (unknown mark `slow`) | Fixed `freq='H'` -> `'h'` deprecation |
| Gaming-Market-Intelligence | 23 passed | 49.00% | 0 | All pass |

**Total: 103 tests passing, 0 failures.**

Run command (from each project dir):
```bash
python3 -m pytest tests/ -v --tb=short -c pyproject.toml \
  --override-ini="addopts=-v --tb=short --cov=. --cov-report=term-missing"
```

### Priority 2: Coverage Badges — UPDATED

All 4 README badges previously showed `Coverage-70%`. Updated to actual values:

| Project | Old Badge | New Badge |
|---------|-----------|-----------|
| OilWell | 70% (green) | 63% (yellowgreen) |
| GoldRecovery | 70% (green) | 43% (yellow) |
| Chicago | 70% (green) | 62% (yellowgreen) |
| Gaming | 70% (green) | 49% (yellow) |

### Priority 3: Docker Builds — ALL PASS

All 4 images build successfully:

| Image | Base | Status |
|-------|------|--------|
| `applied-ml-oilwell` | python:3.10-slim | Build OK |
| `applied-ml-goldrecovery` | python:3.11-slim | Build OK |
| `applied-ml-chicago` | python:3.10-slim | Build OK |
| `applied-ml-gaming` | python:3.10-slim | Build OK |

### Priority 4: Training Pipelines — ALL PASS

All 4 projects train successfully with their shipped sample data:

| Project | Command | Key Metric | Model Output |
|---------|---------|------------|--------------|
| OilWell | `python3 main.py --mode train --config configs/default.yaml` | Region 1 RMSE 0.88 | `artifacts/models/region_*.joblib` |
| GoldRecovery | `python3 main.py --mode train --config configs/config.yaml --input data/raw/gold_recovery_train.csv` | sMAPE 2.64%, R2 0.954 | `models/metallurgical_model.pkl` |
| Chicago | `python3 main.py --mode train --config configs/default.yaml --seed 42` | Test RMSE 520.8, R2 0.607 | `models/duration_model.pkl` |
| Gaming | `python3 main.py --mode train --config configs/config.yaml` | AUC 0.834, Accuracy 81.6% | `artifacts/model/model.joblib` |

**Note**: Makefiles use `PY=python` which may not exist on all systems. Use `python3` directly or set `PY=python3` when invoking make.

---

## Fixes Applied

1. **Chicago `test_preprocessing.py`**: Changed `freq="H"` to `freq="h"` (pandas FutureWarning fix)
2. **Coverage badges**: Updated all 4 READMEs from hardcoded 70% to actual measured values

---

## Docker on GKE — Assessment

### Should you deploy Applied-ML-Projects on GKE?

**Short answer: No, it's not worth the effort for portfolio value.**

**Reasoning:**
1. Your ML-MLOps-Portfolio already demonstrates GKE deployment with 3 services
2. Deploying 4 more services adds cost (~$20-30/month) with diminishing returns
3. The Applied-ML-Projects serve a different purpose: **domain versatility**, not infra
4. Time is better spent on the ML-MLOps-Portfolio infra improvements

**What to say in interviews:**
> "My Applied-ML-Projects demonstrate versatility across domains — each has
> Docker packaging and FastAPI serving ready for deployment. My main portfolio
> shows the full Kubernetes deployment with monitoring and CI/CD."

### If you still want to deploy (not recommended)

The simplest approach would be Cloud Run (not GKE):
```bash
# Build and push to GCR
cd OilWell-Location-Optimizer
docker build -t gcr.io/YOUR_PROJECT/oilwell:v1 .
docker push gcr.io/YOUR_PROJECT/oilwell:v1

# Deploy to Cloud Run (serverless, pay-per-request)
gcloud run deploy oilwell \
    --image gcr.io/YOUR_PROJECT/oilwell:v1 \
    --region us-central1 \
    --memory 512Mi \
    --allow-unauthenticated
```

Cloud Run is cheaper ($0 when not in use) and still shows cloud deployment capability.

---

## Summary

| Action | Status | Date |
|--------|--------|------|
| Run all tests, fix any failures | **DONE** — 103/103 pass | Mar 5, 2026 |
| Update coverage badges | **DONE** — 4 READMEs updated | Mar 5, 2026 |
| Verify Docker builds | **DONE** — 4/4 build OK | Mar 5, 2026 |
| Verify `make train` with sample data | **DONE** — 4/4 train OK | Mar 5, 2026 |
| Deploy to GKE | Skipped — diminishing returns | — |
| Deploy to Cloud Run | Optional | — |

**Bottom line**: All verification steps completed. Repo is solid and ready for portfolio use.
