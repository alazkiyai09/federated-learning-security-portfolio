# 30-Day Fraud Detection & Federated Learning Portfolio
## Part 1: Days 1-10

---

# Day 1: Fraud Detection EDA Dashboard

## 🎯 Session Setup (Copy This First)

```
You are an expert Python data scientist helping me build a production-grade EDA dashboard for credit card fraud detection.

PROJECT CONTEXT:
- Name: Fraud Detection EDA Dashboard  
- Purpose: Interactive exploratory data analysis for credit card fraud data
- Tech Stack: Plotly Dash, Pandas, NumPy, scikit-learn
- Dataset: Kaggle Credit Card Fraud Detection (284,807 transactions, 31 features)

MY BACKGROUND:
- 3+ years experience in fraud detection with SAS Fraud Management
- Building this for my AI/ML portfolio targeting financial services roles
- I need clean, well-documented, production-quality code

REQUIREMENTS:
- Interactive Plotly Dash dashboard
- 5 visualizations: class distribution, amount histogram, correlation heatmap, time patterns, PCA scatter
- Summary statistics card
- Interactive filters (amount range, log scale toggle)
- Export to standalone HTML
- Type hints and docstrings on all functions
- Unit tests with pytest (minimum 80% coverage)
- README.md with screenshots and usage instructions

STRICT RULES:
- Use EXACT function signatures I provide
- Return types must match EXACTLY what I specify
- Colors: fraud=#FF6B6B, legitimate=#4ECDC4
- Do NOT change function names or parameter names

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 2: Imbalanced Classification Techniques

## 🎯 Session Setup (Copy This First)

```
You are an expert ML engineer helping me benchmark imbalanced classification techniques for fraud detection.

PROJECT CONTEXT:
- Name: Imbalanced Classification Benchmark
- Purpose: Compare 6 techniques for handling 0.17% fraud rate
- Tech Stack: scikit-learn, imbalanced-learn, PyTorch, XGBoost

MY BACKGROUND:
- 3+ years fraud detection experience with SAS Fraud Management
- Building portfolio for AI Engineer roles in financial services
- Need rigorous cross-validation methodology

REQUIREMENTS:
- 6 techniques: baseline, random_undersampling, SMOTE, ADASYN, class_weight, focal_loss
- Stratified 5-fold cross-validation
- Metrics: accuracy, precision, recall, F1, AUPRC, AUROC, Recall@FPR
- Focal Loss in PyTorch
- Publication-quality visualizations
- Unit tests for metric calculations
- README.md with results table and methodology

STRICT RULES:
- Function signatures must match EXACTLY
- random_state=42 everywhere
- Focal Loss must handle numerical stability
- Results must be reproducible

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 3: Feature Engineering for Fraud Detection

## 🎯 Session Setup (Copy This First)

```
You are an expert ML engineer helping me build a feature engineering pipeline for fraud detection.

PROJECT CONTEXT:
- Name: Fraud Feature Engineering Pipeline
- Purpose: Create domain-specific features that improve fraud detection
- Tech Stack: scikit-learn (fit/transform), Pandas, SHAP

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Familiar with velocity features, risk scores, behavioral patterns
- Building portfolio for AI Engineer roles in financial services

REQUIREMENTS:
- sklearn-compatible transformers (BaseEstimator, TransformerMixin)
- VelocityFeatures: transaction count/amount in time windows
- DeviationFeatures: compare to user's historical behavior
- MerchantRiskFeatures: merchant-level fraud rates
- SHAP-based feature selection
- Pipeline serializable with joblib
- Unit tests for each transformer
- README.md with feature importance analysis

STRICT RULES:
- All transformers: fit() returns self, transform() returns DataFrame
- Handle unseen users/merchants at inference
- No data leakage in velocity computation
- Bayesian smoothing for merchant fraud rates

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 4: Real-Time Fraud Scoring API

## 🎯 Session Setup (Copy This First)

```
You are an expert ML engineer helping me build a production-ready fraud scoring API using FastAPI.

PROJECT CONTEXT:
- Name: Real-Time Fraud Scoring API
- Purpose: Deploy fraud detection model as a scalable REST API
- Tech Stack: FastAPI, Pydantic, XGBoost, Redis, Docker
- Integration: Uses model from Day 2 and features from Day 3

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Building portfolio for AI Engineer roles in financial services
- Need production-quality deployment patterns

REQUIREMENTS:
- FastAPI application with:
  - POST `/predict` - single transaction scoring (return probability + risk tier)
  - POST `/batch_predict` - batch scoring (max 1000 transactions)
  - GET `/model_info` - model metadata (version, metrics, features)
  - GET `/health` - health check endpoint
- Input validation with Pydantic models
- Response caching with Redis (TTL: 5 minutes)
- API key authentication (X-API-Key header)
- Request logging with structured JSON
- Rate limiting (100 requests/minute per API key)
- Docker container with multi-stage build
- docker-compose.yml with Redis service
- Unit tests with pytest + TestClient
- README.md with API documentation and curl examples

STRICT RULES:
- All endpoints must have OpenAPI documentation
- Response time < 100ms for single prediction (p95)
- Proper error handling with meaningful HTTP status codes
- Environment variables for configuration (no hardcoded secrets)
- Type hints on ALL functions
- Async endpoints where appropriate

API RESPONSE FORMAT:
```json
{
  "transaction_id": "string",
  "fraud_probability": 0.0-1.0,
  "risk_tier": "LOW|MEDIUM|HIGH|CRITICAL",
  "top_risk_factors": ["factor1", "factor2", "factor3"],
  "model_version": "string",
  "latency_ms": number
}
```

RISK TIER THRESHOLDS:
- LOW: probability < 0.1
- MEDIUM: 0.1 <= probability < 0.5
- HIGH: 0.5 <= probability < 0.9
- CRITICAL: probability >= 0.9

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 5: LSTM Sequence Modeling

## 🎯 Session Setup (Copy This First)

```
You are an expert deep learning engineer helping me build an LSTM for sequential fraud detection.

PROJECT CONTEXT:
- Name: LSTM Fraud Detection
- Purpose: Use transaction history to improve fraud detection
- Tech Stack: PyTorch
- Hypothesis: Sequential patterns improve over single-transaction models

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Building portfolio for AI Engineer roles and PhD applications
- Need to demonstrate deep learning expertise

REQUIREMENTS:
- LSTM with attention mechanism
- Variable-length sequences (last N transactions per user)
- Compare with single-transaction baseline
- Visualize attention weights
- Export to ONNX
- Unit tests for model components
- README.md with architecture diagram and results

STRICT RULES:
- PyTorch only (not TensorFlow)
- Temporal train/val/test split (no leakage)
- pack_padded_sequence for efficiency
- Attention weights extractable

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 6: Anomaly Detection Approaches

## 🎯 Session Setup (Copy This First)

```
You are an expert ML engineer helping me compare anomaly detection methods for fraud.

PROJECT CONTEXT:
- Name: Anomaly Detection Benchmark
- Purpose: Compare unsupervised methods for detecting novel fraud patterns
- Tech Stack: scikit-learn, PyTorch

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Building portfolio for AI Engineer roles in financial services
- Interested in detecting zero-day fraud attacks without labels

REQUIREMENTS:
- 4 methods: Isolation Forest, One-Class SVM, Autoencoder, LOF
- Train on legitimate transactions ONLY
- Evaluate against labeled fraud
- Ensemble combining all methods (voting + stacking)
- Failure case analysis with visualizations
- Unit tests for anomaly scoring
- README.md with method comparison and recommendations

STRICT RULES:
- Train ONLY on Class=0 data
- Autoencoder in PyTorch with reconstruction loss
- Report detection rate AND false positive rate
- Contamination parameter tuning with validation set

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 7: Model Explainability for Fraud Detection

## 🎯 Session Setup (Copy This First)

```
You are an expert ML engineer helping me build explainability tools for fraud detection models.

PROJECT CONTEXT:
- Name: Fraud Model Explainability
- Purpose: Make fraud detection models interpretable for analysts and regulators
- Tech Stack: SHAP, LIME, scikit-learn, Streamlit
- Use Case: Fraud analysts need to understand WHY a transaction was flagged

MY BACKGROUND:
- 3+ years fraud detection where model governance is critical
- Building portfolio for AI Engineer roles in financial services
- Regulatory compliance (SR 11-7, EU AI Act) requires explainable AI

REQUIREMENTS:
- SHAP explanations (global and local)
- LIME for individual predictions
- Partial Dependence Plots
- "Fraud explanation report" generator (HTML)
- Simple Streamlit UI for analysts
- Support multiple model types (XGBoost, Random Forest, Neural Net)
- Unit tests for explanation consistency
- README.md with regulatory compliance notes

STRICT RULES:
- Explanations must be consistent (same input = same explanation)
- Local explanations must highlight top 5 risk factors
- Reports must be professional and readable by non-technical users
- Performance: <2 seconds per prediction explanation

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 8: Federated Learning from Scratch

## 🎯 Session Setup (Copy This First)

```
You are an expert ML researcher helping me implement Federated Averaging (FedAvg) from scratch.

PROJECT CONTEXT:
- Name: FedAvg from Scratch
- Purpose: Deep understanding of federated learning mechanics
- Tech Stack: PyTorch only (NO Flower, PySyft, or other FL frameworks)
- Reference: "Communication-Efficient Learning of Deep Networks" (McMahan et al., 2017)

MY BACKGROUND:
- Research focus on federated learning for fraud detection
- Building portfolio for PhD applications in trustworthy ML
- Need clean, well-documented implementation for research extension

REQUIREMENTS:
- Pure PyTorch implementation
- FederatedClient class for local training
- FederatedServer class for aggregation
- FedAvg weighted averaging by sample count
- Test on MNIST first (sanity check)
- Then apply to fraud detection
- Convergence visualization
- Unit tests for aggregation logic
- README.md with algorithm explanation and convergence plots

STRICT RULES:
- NO external FL frameworks
- Implement weight serialization from scratch
- Handle heterogeneous client data sizes
- Reproducible with fixed random seeds
- Must match FedAvg algorithm exactly

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 9: Non-IID Data Partitioning

## 🎯 Session Setup (Copy This First)

```
You are an expert FL researcher helping me create a data partitioning tool for FL experiments.

PROJECT CONTEXT:
- Name: Non-IID Data Partitioner
- Purpose: Simulate realistic heterogeneous data distributions in FL
- Tech Stack: NumPy, Pandas, scikit-learn, Matplotlib

MY BACKGROUND:
- Research focus on federated learning for fraud detection
- Building portfolio for PhD applications in trustworthy ML
- Non-IID data is THE core challenge in real FL deployments

REQUIREMENTS:
- Multiple partition strategies:
  - IID (random uniform)
  - Label skew (Dirichlet distribution with configurable alpha)
  - Quantity skew (power law distribution for dataset sizes)
  - Feature skew (different feature distributions per client)
  - Realistic bank simulation (geography + customer demographics)
- Visualization of partition statistics (heatmaps, bar charts)
- Reusable module for all FL experiments
- Unit tests for partition correctness
- README.md with partition strategy explanations and visualizations

STRICT RULES:
- Dirichlet alpha correctly controls heterogeneity (alpha→0 = extreme non-IID)
- Support both classification labels and continuous features
- Reproducible with random_state parameter
- Clear visualization showing class distribution per client

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 10: Flower Framework Deep Dive

## 🎯 Session Setup (Copy This First)

```
You are an expert FL engineer helping me master the Flower framework for production FL.

PROJECT CONTEXT:
- Name: Flower Fraud Detection
- Purpose: Production FL using industry-standard framework
- Tech Stack: Flower (flwr >= 1.5), PyTorch, Hydra

MY BACKGROUND:
- Implemented FedAvg from scratch (Day 8)
- Built Non-IID partitioner (Day 9)
- Now need production-ready FL for fraud detection
- Building portfolio for PhD applications and AI Engineer roles

REQUIREMENTS:
- Flower server and client setup
- Custom FlClient class extending fl.client.NumPyClient
- Custom strategies: FedAvg, FedProx, FedAdam
- Simulation with 10 clients using Day 9 partitioner
- Configuration system with Hydra for hyperparameter management
- Compare strategies on IID vs Non-IID data
- TensorBoard logging for experiment tracking
- Unit tests for client logic
- README.md with strategy comparison results

STRICT RULES:
- Use Flower's latest stable API (check documentation)
- Custom strategy must extend fl.server.strategy.Strategy
- Proper metrics aggregation (weighted by sample count)
- Support both simulation and distributed modes
- All hyperparameters configurable via Hydra YAML

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

## 📋 Summary of Improvements Made

### Day 4 (Complete Rewrite)
- Added full session setup prompt matching other days
- Included detailed API specifications with response format
- Added Redis caching, rate limiting, authentication
- Specified risk tier thresholds
- Added Docker + docker-compose requirements
- Included testing and documentation requirements

### Consistency Fixes (All Days)
1. **Added "MY BACKGROUND" section** to Days 6, 9, 10 where missing
2. **Added testing requirements** to all days (unit tests with pytest)
3. **Added README.md requirements** to all days for portfolio presentation
4. **Updated Flower version reference** in Day 10 to be flexible (>= 1.5)
5. **Enhanced Day 6** with ensemble methods (voting + stacking)
6. **Enhanced Day 9** with more specific distribution types

### Portfolio Value Additions
- Explicit regulatory compliance mention (Day 7: SR 11-7, EU AI Act)
- Connection between days made explicit (Day 4 uses models from Days 2-3)
- TensorBoard logging added (Day 10) for experiment tracking
- Architecture diagrams mentioned in READMEs
