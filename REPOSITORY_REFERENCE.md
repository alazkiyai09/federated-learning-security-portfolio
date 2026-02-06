# REPOSITORY REFERENCE: federated-learning-security-portfolio
> Auto-generated on 2026-02-06 | Projects: 30/30 | Domain: Federated Learning Security / Fraud Detection

## QUICK NAVIGATION

### By Category
- **[CAT-01] Fraud Detection Core** → P-01, P-02, P-03, P-04, P-05, P-06, P-07
- **[CAT-02] FL Foundations** → P-08, P-09, P-10, P-11, P-12, P-13, P-20, P-22
- **[CAT-03] Adversarial Attacks** → P-14, P-15, P-16
- **[CAT-04] Defensive Techniques** → P-17, P-18, P-19, P-21
- **[CAT-05] Security Research** → P-23, P-24, P-25, P-26, P-27, P-28, P-29, P-30

### By Complexity
- 🔴 High Complexity: P-24 (SignGuard), P-28 (Privacy Pipeline), P-05 (LSTM+Attention), P-13 (Vertical FL), P-20 (Personalized FL), P-22 (DP-SGD)
- 🟡 Medium Complexity: P-04 (Scoring API), P-10 (Flower Framework), P-11 (Communication Efficient), P-12 (Cross-Silo Bank), P-17 (Byzantine-Robust), P-21 (Defense Benchmark), P-23 (Secure Aggregation), P-25 (Membership Inference), P-26 (Gradient Leakage)
- 🟢 Low Complexity: P-01 (EDA Dashboard), P-02 (Classification Benchmark), P-03 (Feature Engineering), P-06 (Anomaly Detection), P-07 (Explainability), P-08 (FedAvg), P-09 (Non-IID Partitioner), P-14 (Label Flipping), P-15 (Backdoor), P-16 (Model Poisoning), P-18 (FL Anomaly Detection), P-19 (FoolsGold), P-27 (Property Inference), P-29 (Security Dashboard)

### Cross-Project Dependencies
```
P-04 (Scoring API) ← depends on ← P-02, P-03
P-05 (LSTM) ← depends on ← P-03
P-07 (Explainability) ← depends on ← P-02
P-09 (Non-IID) ← depends on ← P-08
P-21 (Defense Benchmark) ← uses defenses from ← P-17, P-18, P-19
P-24 (SignGuard) ← attacks from ← P-14, P-15, P-16
P-28 (Privacy Pipeline) ← integrates ← P-22 (DP-SGD), P-23 (Secure Aggregation)
P-29 (Dashboard) ← visualizes results from ← all categories
```

---

## REPO_INDEX

```yaml
repo_name: "federated-learning-security-portfolio"
repo_url: "https://github.com/alazkiyai09/federated-learning-security-portfolio"
total_projects: 30
primary_language: "Python"
domain: "Federated Learning Security / Fraud Detection"
last_analyzed: "2026-02-06"
```

## CATEGORIES

| Category ID | Category Name | Project Count | Projects (IDs) |
|-------------|---------------|---------------|-----------------|
| CAT-01 | Fraud Detection Core | 7 | P-01, P-02, P-03, P-04, P-05, P-06, P-07 |
| CAT-02 | FL Foundations | 8 | P-08, P-09, P-10, P-11, P-12, P-13, P-20, P-22 |
| CAT-03 | Adversarial Attacks | 3 | P-14, P-15, P-16 |
| CAT-04 | Defensive Techniques | 4 | P-17, P-18, P-19, P-21 |
| CAT-05 | Security Research | 8 | P-23, P-24, P-25, P-26, P-27, P-28, P-29, P-30 |

## PROJECT REGISTRY

| Project ID | Day | Name | Category | Path | Status | Score |
|------------|-----|------|----------|------|--------|-------|
| P-01 | 1 | EDA Dashboard | CAT-01 | `01_fraud_detection_core/fraud_detection_eda_dashboard/` | ✅ | 9/10 |
| P-02 | 2 | Classification Benchmark | CAT-01 | `01_fraud_detection_core/imbalanced_classification_benchmark/` | ✅ | 10/10 |
| P-03 | 3 | Feature Engineering | CAT-01 | `01_fraud_detection_core/fraud_feature_engineering/` | ✅ | 10/10 |
| P-04 | 4 | Real-time Scoring API | CAT-01 | `01_fraud_detection_core/fraud_scoring_api/` | ✅ | 10/10 |
| P-05 | 5 | LSTM Sequence Modeling | CAT-01 | `01_fraud_detection_core/lstm_fraud_detection/` | ✅ | 9/10 |
| P-06 | 6 | Anomaly Detection | CAT-01 | `01_fraud_detection_core/anomaly_detection_benchmark/` | ✅ | 9/10 |
| P-07 | 7 | Model Explainability | CAT-01 | `01_fraud_detection_core/fraud_model_explainability/` | ✅ | 10/10 |
| P-08 | 8 | FedAvg from Scratch | CAT-02 | `02_federated_learning_foundations/fedavg_from_scratch/` | ✅ | 10/10 |
| P-09 | 9 | Non-IID Partitioner | CAT-02 | `02_federated_learning_foundations/non_iid_partitioner/` | ✅ | 9/10 |
| P-10 | 10 | Flower Framework | CAT-02 | `02_federated_learning_foundations/flower_fraud_detection/` | ✅ | 10/10 |
| P-11 | 11 | Communication Efficient FL | CAT-02 | `02_federated_learning_foundations/communication_efficient_fl/` | ✅ | 10/10 |
| P-12 | 12 | Cross-Silo Bank Simulation | CAT-02 | `02_federated_learning_foundations/cross_silo_bank_fl/` | ✅ | 8/10 |
| P-13 | 13 | Vertical FL | CAT-02 | `02_federated_learning_foundations/vertical_fraud_detection/` | ✅ | 10/10 |
| P-14 | 14 | Label Flipping Attack | CAT-03 | `03_adversarial_attacks/label_flipping_attack/` | ✅ | 9/10 |
| P-15 | 15 | Backdoor Attack | CAT-03 | `03_adversarial_attacks/backdoor_attack_fl/` | ✅ | 9/10 |
| P-16 | 16 | Model Poisoning | CAT-03 | `03_adversarial_attacks/model_poisoning_fl/` | ✅ | 9/10 |
| P-17 | 17 | Byzantine-Robust FL | CAT-04 | `04_defensive_techniques/byzantine_robust_fl/` | ✅ | 10/10 |
| P-18 | 18 | FL Anomaly Detection | CAT-04 | `04_defensive_techniques/fl_anomaly_detection/` | ✅ | 9/10 |
| P-19 | 19 | FoolsGold Defense | CAT-04 | `04_defensive_techniques/foolsgold_defense/` | ✅ | 10/10 |
| P-20 | 20 | Personalized FL | CAT-02 | `02_federated_learning_foundations/personalized_fl_fraud/` | ✅ | 9/10 |
| P-21 | 21 | Defense Benchmark | CAT-04 | `04_defensive_techniques/fl_defense_benchmark/` | ✅ | 9/10 |
| P-22 | 22 | Differential Privacy | CAT-02 | `02_federated_learning_foundations/dp_federated_learning/` | ✅ | 9/10 |
| P-23 | 23 | Secure Aggregation | CAT-05 | `05_security_research/secure_aggregation_fl/` | ✅ | 10/10 |
| P-24 | 24 | SignGuard | CAT-05 | `05_security_research/signguard/` | ✅ | 10/10 |
| P-25 | 25 | Membership Inference | CAT-05 | `05_security_research/membership_inference_attack/` | ✅ | 10/10 |
| P-26 | 26 | Gradient Leakage | CAT-05 | `05_security_research/gradient_leakage_attack/` | ✅ | 10/10 |
| P-27 | 27 | Property Inference | CAT-05 | `05_security_research/property_inference_attack/` | ✅ | 10/10 |
| P-28 | 28 | Privacy Pipeline | CAT-05 | `05_security_research/privacy_preserving_fl_fraud/` | ✅ | 10/10 |
| P-29 | 29 | Security Dashboard | CAT-05 | `05_security_research/fl_security_dashboard/` | ✅ | 10/10 |
| P-30 | 30 | Capstone Research Paper | CAT-05 | `docs/capstone_research_paper.md` | ✅ | 10/10 |

## DEPENDENCY MAP

| Project ID | Depends On | Shared Modules | External Libs |
|------------|------------|----------------|---------------|
| P-01 | None | utils/data_loader.py | plotly, pandas, numpy, dash |
| P-02 | None | — | scikit-learn, imbalanced-learn, torch, xgboost |
| P-03 | P-01 | — | pandas, numpy, scikit-learn, shap |
| P-04 | P-02, P-03 | — | fastapi, uvicorn, pydantic, redis |
| P-05 | P-03 | — | torch, numpy, onnx |
| P-06 | None | — | scikit-learn, torch |
| P-07 | P-02 | — | shap, lime, scikit-learn, streamlit |
| P-08 | None | — | torch, torchvision, numpy |
| P-09 | P-08 | — | numpy, scikit-learn, scipy |
| P-10 | None | — | flwr, torch, hydra-core, ray |
| P-11 | None | — | numpy, torch, flwr |
| P-12 | None | — | torch, flwr, ray, pandas |
| P-13 | None | — | torch, numpy, pandas |
| P-14 | None | — | torch, flwr, numpy, scikit-learn |
| P-15 | None | — | torch, numpy, matplotlib |
| P-16 | None | — | torch, flwr, numpy, scipy |
| P-17 | None | — | torch, numpy, pandas, scipy |
| P-18 | None | — | numpy, scikit-learn, flwr, torch |
| P-19 | None | — | flwr, torch, numpy, scipy |
| P-20 | None | — | torch, flwr, omegaconf |
| P-21 | P-17, P-18, P-19 | — | torch, flwr, hydra-core, mlflow |
| P-22 | None | — | torch, numpy |
| P-23 | None | — | torch, numpy, pycryptodome |
| P-24 | None | — | torch, cryptography, hydra-core |
| P-25 | None | — | torch, scikit-learn, numpy |
| P-26 | None | — | torch, torchvision, numpy |
| P-27 | None | — | torch, scikit-learn, scipy |
| P-28 | P-22, P-23 | — | torch, flwr, opacus, fastapi, mlflow |
| P-29 | None | — | streamlit, plotly, numpy, pydantic |
| P-30 | P-24 | — | (documentation only — no code dependencies) |

## ARCHITECTURE NOTES

- **Config style**: Mix of YAML (Hydra/OmegaConf) and Python dataclasses. Projects P-10, P-20, P-21, P-24, P-28 use Hydra. Others use plain YAML or argparse.
- **Model pattern**: Most projects use PyTorch `nn.Module` with factory functions (`get_model()`, `create_model()`). Scikit-learn used for baselines and ensemble components.
- **FL framework**: Projects P-10, P-12, P-14, P-16, P-18, P-19, P-21, P-28 use Flower (flwr). Projects P-08, P-15 use custom FL implementations.
- **Testing**: All projects include pytest-based tests. Coverage varies; metrics projects tend to have highest coverage.
- **Common abstractions**: `BaseExplainer(ABC)` in P-07, `AnomalyDetector(ABC)` in P-06, `RobustAggregator(ABC)` in P-17, `BaseAttack(ABC)` in P-21, `BaseDefense(ABC)` in P-21, `PersonalizationMethod(ABC)` in P-20, `ModelPoisoningAttack(ABC)` in P-16, `BaseDetector(ABC)` in P-18.
- **Data**: Most projects use synthetic fraud data generated via `sklearn.datasets.make_classification()` or custom generators. Real data (Kaggle creditcard.csv) supported in P-01, P-02.

---

## PROJECT CARDS
# CAT-01: Fraud Detection Core -- PROJECT CARDS

---

## P-01: fraud_detection_eda_dashboard (Day 1 | Score 9/10)

### 1. PURPOSE

Interactive exploratory data analysis dashboard for credit card fraud detection built with Plotly Dash. Loads the Kaggle Credit Card Fraud Detection dataset (`creditcard.csv`), preprocesses it (adds `Hour`, `Log_Amount` columns), computes summary statistics, and renders five interactive charts: class distribution bar chart, transaction amount histogram, correlation heatmap, time-pattern subplots, and PCA scatter plot. Supports live filtering by amount range with a log-scale toggle and one-click HTML export of the current dashboard state.

### 2. ARCHITECTURE

```
fraud_detection_eda_dashboard/
  fraud_detection_dashboard/
    __init__.py
    app.py                  # Dash app factory: create_app(), main()
    data_loader.py          # load_fraud_data(), preprocess_data(), validate_data()
    layout.py               # create_dashboard_layout(), create_filters(), create_charts_grid()
    visualizations.py       # plot_class_distribution(), plot_correlation_heatmap(), plot_pca_scatter() ...
    callbacks.py            # register_callbacks() -> update_charts(), export_dashboard()
    utils.py                # calculate_summary_statistics(), export_to_html(), format_currency()
  run_dashboard.py          # CLI entry point (argparse)
  tests/
    test_data_loader.py
    test_layout.py
    test_visualizations.py
    test_callbacks.py
    test_utils.py
  setup.py
  pytest.ini
  requirements.txt
```

### 3. KEY COMPONENTS

| File (rel. path) | Class / Function | Lines | Role |
|---|---|---|---|
| `fraud_detection_dashboard/app.py` | `create_app(data_path)` | 25-91 | Factory: loads data, creates layout, registers callbacks, returns `dash.Dash` |
| `fraud_detection_dashboard/app.py` | `main(data_path, host, port, debug)` | 94-147 | Runs Dash dev server on configurable host/port |
| `fraud_detection_dashboard/data_loader.py` | `load_fraud_data(path, validate)` | -- | Reads CSV, optionally validates schema |
| `fraud_detection_dashboard/data_loader.py` | `preprocess_data(df, normalize_time)` | -- | Adds `Hour` (from Time), `Log_Amount` columns |
| `fraud_detection_dashboard/layout.py` | `create_dashboard_layout(stats)` | -- | Builds full Dash layout: header, summary cards, filters, 5 chart placeholders |
| `fraud_detection_dashboard/layout.py` | `create_filters()` | -- | RangeSlider for amount + log-scale toggle checkbox |
| `fraud_detection_dashboard/visualizations.py` | `plot_pca_scatter()` | -- | Applies `StandardScaler` + PCA(n_components=2), returns Plotly scatter |
| `fraud_detection_dashboard/visualizations.py` | `plot_correlation_heatmap()` | -- | Plotly heatmap of Pearson correlations |
| `fraud_detection_dashboard/callbacks.py` | `register_callbacks(app, df)` | -- | Wires up `update_charts()` (amount filter -> 5 figures) and `export_dashboard()` |
| `fraud_detection_dashboard/utils.py` | `calculate_summary_statistics(df)` | -- | Returns dict with total_transactions, fraud_count, fraud_percentage, etc. |
| `run_dashboard.py` | `main()` | -- | argparse CLI: `--data-path`, `--host`, `--port`, `--debug` |

### 4. DATA FLOW

```
creditcard.csv
      |
      v
load_fraud_data() --validate--> preprocess_data() --adds Hour, Log_Amount-->
      |
      v
calculate_summary_statistics()
      |
      v
create_dashboard_layout(stats)  -->  Dash app.layout
      |
      v
register_callbacks(app, df_processed)
      |
      v
[User adjusts RangeSlider / log toggle]
      |
      v
update_charts(amount_range, log_flag) -->  5 Plotly figures returned
      |
      v
[User clicks Export]  -->  export_dashboard()  -->  timestamped HTML file
```

### 5. CONFIGURATION & PARAMETERS

| Parameter | Location | Default | Notes |
|---|---|---|---|
| `DEFAULT_DATA_PATH` | `app.py:22` | `"data/creditcard.csv"` | Kaggle dataset path |
| `host` | `app.py:97` / CLI | `"127.0.0.1"` | Server bind address |
| `port` | `app.py:97` / CLI | `8050` | Server port |
| `debug` | `app.py:98` / CLI | `True` | Enables hot-reload |
| `FRAUD_COLOR` | `utils.py` | `"#FF6B6B"` | Red for fraud class |
| `LEGIT_COLOR` | `utils.py` | `"#4ECDC4"` | Teal for legit class |
| `n_components` (PCA) | `visualizations.py` | `2` | PCA dimensions for scatter |
| Coverage target | `pytest.ini` | `80%` | Minimum test coverage |

### 6. EXTERNAL DEPENDENCIES

| Package | Version | Purpose |
|---|---|---|
| dash | >=2.14.0 | Web dashboard framework |
| plotly | >=5.18.0 | Interactive charting |
| pandas | >=2.1.0 | Data manipulation |
| numpy | >=1.24.0 | Numerical operations |
| scikit-learn | >=1.3.0 | PCA, StandardScaler |
| pytest | >=7.4.0 | Testing framework |
| pytest-cov | >=4.1.0 | Coverage reporting |
| pytest-mock | >=3.12.0 | Mock support |
| python-dotenv | >=1.0.0 | Environment variables |

### 7. KNOWN ISSUES / LIMITATIONS

- Requires the Kaggle `creditcard.csv` file to be manually downloaded and placed at `data/creditcard.csv`; no built-in download mechanism.
- PCA scatter plot in `visualizations.py` fits `StandardScaler` + PCA on every callback invocation (no caching), which can be slow for the full 284,807-row dataset.
- The `export_to_html()` function exports a static snapshot; interactive filtering is lost in exported files.
- No pagination or sampling for very large datasets -- the full DataFrame is passed to callbacks.
- `suppress_callback_exceptions=True` in `app.py:63` can silently swallow callback errors during development.

### 8. TESTING

| Test File | Covers |
|---|---|
| `tests/test_data_loader.py` | `load_fraud_data`, `preprocess_data`, `validate_data` |
| `tests/test_layout.py` | Layout component structure |
| `tests/test_visualizations.py` | All 5 plot functions |
| `tests/test_callbacks.py` | Callback wiring and output shapes |
| `tests/test_utils.py` | Summary statistics, formatting utilities |

Run: `pytest tests/ --cov=fraud_detection_dashboard --cov-fail-under=80`

### 9. QUICK MODIFICATION GUIDE

| Want to... | Change in... |
|---|---|
| Add a new chart type | `visualizations.py` (new function) + `callbacks.py` (add output) + `layout.py` (add placeholder div) |
| Change color scheme | `utils.py` -- update `FRAUD_COLOR`, `LEGIT_COLOR` constants |
| Add a filter (e.g., by hour) | `layout.py:create_filters()` (add component) + `callbacks.py:update_charts()` (add Input) |
| Use a different dataset | Pass `--data-path` via CLI or change `DEFAULT_DATA_PATH` in `app.py:22` |
| Increase PCA dimensions | `visualizations.py:plot_pca_scatter()` -- change `n_components` parameter |
| Disable validation on load | Call `load_fraud_data(path, validate=False)` in `app.py:69` |

### 10. CODE SNIPPETS

**App factory pattern** (`fraud_detection_dashboard/app.py:25-91`):
```python
def create_app(data_path: Optional[str] = None) -> dash.Dash:
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    app = dash.Dash(
        __name__,
        title='Fraud Detection EDA Dashboard',
        suppress_callback_exceptions=True,
        update_title='Loading...',
    )
    df = load_fraud_data(data_path, validate=True)
    df_processed = preprocess_data(df, normalize_time=True)
    stats = calculate_summary_statistics(df_processed)
    app.layout = html.Div([
        create_dashboard_layout(stats),
        html.Div(id='dummy-div', style={'display': 'none'})
    ])
    register_callbacks(app, df_processed)
    return app
```

---

## P-02: imbalanced_classification_benchmark (Day 2 | Score 10/10)

### 1. PURPOSE

Systematic benchmark of six techniques for handling class imbalance in credit card fraud detection: Baseline (no resampling), Random Undersampling, SMOTE, ADASYN, Class Weighting, and Focal Loss. Uses Stratified K-Fold cross-validation, computes seven metrics (AUPRC, AUROC, Recall@1%FPR, Precision, Recall, F1, Accuracy), and generates publication-quality comparison visualizations (bar chart, heatmap, ranking plot). The Focal Loss technique uses a custom PyTorch neural network with `FocalLoss(nn.Module)`.

### 2. ARCHITECTURE

```
imbalanced_classification_benchmark/
  main.py                           # CLI orchestrator (argparse)
  src/
    config.py                       # Config dataclass (RANDOM_STATE, N_FOLDS, etc.)
    data_loader.py                  # load_data(), generate_synthetic_fraud_data()
    experiment.py                   # ExperimentRunner, ExperimentResult dataclass
    cross_validation.py             # stratified_cross_validation()
    visualization.py                # plot_metrics_comparison(), plot_metrics_heatmap(), ...
    models/
      baseline.py                   # LogisticRegressionBaseline, RandomForestBaseline
      focal_loss.py                 # FocalLoss(nn.Module), FocalLossClassifier
      xgboost_wrapper.py            # XGBoostWrapper (scale_pos_weight)
    techniques/
      smote.py                      # apply_smote()
      adasyn.py                     # apply_adasyn()
      undersampling.py              # apply_random_undersampling()
    metrics/
      metrics.py                    # calculate_auprc(), calculate_auroc(), recall_at_fpr(), compute_all_metrics()
  tests/
    test_metrics.py                 # TestCalculateAUPRC, TestCalculateAUROC, TestRecallAtFPR
  requirements.txt
```

### 3. KEY COMPONENTS

| File (rel. path) | Class / Function | Lines | Role |
|---|---|---|---|
| `main.py` | `main()` | -- | CLI entry point: loads data, runs experiments, saves results, generates plots |
| `src/config.py` | `Config` (dataclass) | -- | Central config: RANDOM_STATE=42, N_FOLDS=5, FOCAL_ALPHA=0.25, FOCAL_GAMMA=2.0, FPR_THRESHOLD=0.01, FIGURE_DPI=300 |
| `src/experiment.py` | `ExperimentRunner` | -- | Orchestrates 6 techniques, aggregates results, formats table |
| `src/experiment.py` | `ExperimentResult` (dataclass) | -- | Stores per-technique metrics, fold results |
| `src/cross_validation.py` | `stratified_cross_validation(model, X, y, n_folds, resample_fn)` | -- | StratifiedKFold with optional resampling function parameter |
| `src/models/focal_loss.py` | `FocalLoss(nn.Module)` | 18-85 | `FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)` |
| `src/models/focal_loss.py` | `FocalLossClassifier` | 88-239 | Feedforward NN [64,32] with BatchNorm, Dropout(0.2), Adam optimizer |
| `src/models/xgboost_wrapper.py` | `XGBoostWrapper` | -- | XGBoost with auto `scale_pos_weight` from class ratio |
| `src/techniques/smote.py` | `apply_smote(X, y)` | -- | imblearn SMOTE wrapper |
| `src/techniques/adasyn.py` | `apply_adasyn(X, y)` | -- | imblearn ADASYN wrapper |
| `src/techniques/undersampling.py` | `apply_random_undersampling(X, y)` | -- | imblearn RandomUnderSampler wrapper |
| `src/metrics/metrics.py` | `compute_all_metrics(y_true, y_pred, y_proba)` | -- | Returns dict of 7 metrics |
| `src/visualization.py` | `create_all_visualizations(results, output_dir)` | -- | Generates 4 publication-quality plots |

### 4. DATA FLOW

```
creditcard.csv  OR  generate_synthetic_fraud_data(n=100000, fraud_rate=0.0017)
      |
      v
load_or_generate_data()
      |
      v
ExperimentRunner.run_all_experiments()
      |
      +---> Technique 1: Baseline (no resampling)
      +---> Technique 2: Random Undersampling
      +---> Technique 3: SMOTE
      +---> Technique 4: ADASYN
      +---> Technique 5: Class Weighting (scale_pos_weight)
      +---> Technique 6: Focal Loss (PyTorch NN)
      |         |
      |    [each]: stratified_cross_validation(model, X, y, n_folds=5, resample_fn=...)
      |         |
      |         v
      |    compute_all_metrics() per fold -> aggregate mean +/- std
      |
      v
_aggregate_results() -> ExperimentResult per technique
      |
      v
format_results_table() + save_results(CSV/JSON)
      |
      v
create_all_visualizations() -> 4 PNG files at FIGURE_DPI=300
```

### 5. CONFIGURATION & PARAMETERS

| Parameter | Location | Default | Notes |
|---|---|---|---|
| `RANDOM_STATE` | `src/config.py` | `42` | Global seed |
| `N_FOLDS` | `src/config.py` | `5` | Stratified K-Fold splits |
| `FOCAL_ALPHA` | `src/config.py` | `0.25` | Focal Loss alpha |
| `FOCAL_GAMMA` | `src/config.py` | `2.0` | Focal Loss gamma |
| `SAMPLING_STRATEGY` | `src/config.py` | `0.1` | Target minority ratio for SMOTE/ADASYN |
| `FPR_THRESHOLD` | `src/config.py` | `0.01` | FPR threshold for Recall@FPR metric |
| `FIGURE_DPI` | `src/config.py` | `300` | Plot resolution |
| `hidden_dims` | `focal_loss.py:124` | `[64, 32]` | NN hidden layers |
| `epochs` | `focal_loss.py:113` | `100` | Training epochs |
| `batch_size` | `focal_loss.py:114` | `256` | Training batch size |
| `learning_rate` | `focal_loss.py:112` | `0.001` | Adam LR |

### 6. EXTERNAL DEPENDENCIES

| Package | Version | Purpose |
|---|---|---|
| scikit-learn | >=1.3.0 | Models, CV, metrics |
| imbalanced-learn | >=0.11.0 | SMOTE, ADASYN, RandomUnderSampler |
| torch | >=2.0.0 | Focal Loss neural network |
| xgboost | >=2.0.0 | XGBoost classifier |
| matplotlib | >=3.7.0 | Visualization |
| seaborn | >=0.12.0 | Heatmap styling |
| pandas | >=2.0.0 | Data handling |
| numpy | >=1.24.0 | Numerical operations |
| tqdm | >=4.65.0 | Progress bars |
| pyyaml | >=6.0 | Configuration |

### 7. KNOWN ISSUES / LIMITATIONS

- Focal Loss training loop in `FocalLossClassifier.fit()` (line 179) does not log per-epoch loss or support early stopping -- trains for the full `epochs` count every time.
- `generate_synthetic_fraud_data()` creates data with `make_classification()` which does not replicate real-world fraud distribution patterns.
- No GPU utilization tracking or mixed-precision training for the Focal Loss model.
- SMOTE/ADASYN wrappers do not expose all imblearn parameters (e.g., `k_neighbors`).
- The `ExperimentRunner` does not support custom model injection -- the 6 techniques are hardcoded.

### 8. TESTING

| Test File | Covers |
|---|---|
| `tests/test_metrics.py` | `TestCalculateAUPRC`, `TestCalculateAUROC`, `TestRecallAtFPR`, `TestComputeAllMetrics` |

Run: `pytest tests/ --cov=src --cov-fail-under=80`

Note: Only metrics are currently unit-tested. Model training and visualization functions lack dedicated tests.

### 9. QUICK MODIFICATION GUIDE

| Want to... | Change in... |
|---|---|
| Add a new resampling technique | Create `src/techniques/new_technique.py`, add to `ExperimentRunner.run_all_experiments()` |
| Change Focal Loss architecture | `src/models/focal_loss.py:137-149` -- modify `layers` list |
| Adjust FPR threshold for Recall@FPR | `src/config.py:FPR_THRESHOLD` |
| Add a new metric | `src/metrics/metrics.py:compute_all_metrics()` -- add to return dict |
| Change number of CV folds | `src/config.py:N_FOLDS` |
| Use real data instead of synthetic | `src/data_loader.py:load_or_generate_data()` -- provide CSV path |

### 10. CODE SNIPPETS

**Focal Loss forward pass** (`src/models/focal_loss.py:46-85`):
```python
def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if inputs.dim() == 1:
        inputs = inputs.unsqueeze(1)
    bce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets.unsqueeze(1).float(), reduction="none"
    )
    p_t = torch.sigmoid(inputs)
    p_t = torch.where(targets.unsqueeze(1) == 1, p_t, 1 - p_t)
    alpha_t = torch.where(targets.unsqueeze(1) == 1, self.alpha, 1 - self.alpha)
    focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
    if self.reduction == "mean":
        return focal_loss.mean()
    elif self.reduction == "sum":
        return focal_loss.sum()
    else:
        return focal_loss.squeeze(1)
```

---

## P-03: fraud_feature_engineering (Day 3 | Score 10/10)

### 1. PURPOSE

Production-ready fraud feature engineering pipeline using scikit-learn `Pipeline` and `FeatureUnion`. Combines three transformer families -- velocity features (rolling-window transaction counts/sums/means/stds), deviation features (z-scores and ratios against user historical patterns), and merchant risk features (Bayesian-smoothed fraud rates) -- with optional SHAP-based feature selection and StandardScaler. The pipeline is serializable via joblib for deployment.

### 2. ARCHITECTURE

```
fraud_feature_engineering/
  src/
    pipeline.py                         # FraudFeaturePipeline (main orchestrator)
    transformers/
      base.py                           # safe_datetime_convert(), rolling_window_stats(), etc.
      velocity_features.py              # VelocityFeatures(BaseEstimator, TransformerMixin)
      deviation_features.py             # DeviationFeatures(BaseEstimator, TransformerMixin)
      merchant_features.py              # MerchantRiskFeatures(BaseEstimator, TransformerMixin)
    feature_selection/
      shap_selector.py                  # SHAPSelector (TreeExplainer-based feature selection)
  tests/
    test_velocity_features.py
    test_velocity_features_unittest.py
    test_deviation_features.py
    test_deviation_features_unittest.py
    test_merchant_features.py
    test_merchant_features_unittest.py
    test_pipeline.py
    test_pipeline_unittest.py
  requirements.txt
```

### 3. KEY COMPONENTS

| File (rel. path) | Class / Function | Lines | Role |
|---|---|---|---|
| `src/pipeline.py` | `FraudFeaturePipeline` | 15-290 | Orchestrates FeatureUnion of transformers + optional SHAP selection + optional scaling; save/load via joblib |
| `src/pipeline.py` | `_build_pipeline()` | 95-136 | Constructs sklearn Pipeline with FeatureUnion, optional SHAPSelector, optional StandardScaler |
| `src/transformers/velocity_features.py` | `VelocityFeatures` | -- | Rolling-window count/sum/mean/std over configurable time windows; `time_since_last` feature |
| `src/transformers/deviation_features.py` | `DeviationFeatures` | -- | Z-scores and ratios vs. user historical mean/std; global fallback for unseen users |
| `src/transformers/merchant_features.py` | `MerchantRiskFeatures` | -- | Bayesian smoothing: `smoothed_rate = (fraud_count + alpha) / (total_count + alpha + beta)`, blended with global rate |
| `src/transformers/base.py` | `safe_datetime_convert()` | -- | Robust datetime parsing |
| `src/transformers/base.py` | `rolling_window_stats()` | -- | Generic rolling window computation |
| `src/feature_selection/shap_selector.py` | `SHAPSelector` | -- | Uses TreeExplainer to compute mean absolute SHAP values; selects top N features |

### 4. DATA FLOW

```
Raw Transaction DataFrame (user_id, merchant_id, timestamp, amount, ...)
      |
      v
FraudFeaturePipeline.fit(X, y)
      |
      v
FeatureUnion (parallel):
  +---> VelocityFeatures.transform(X)
  |       rolling count/sum/mean/std per time_window [(1,"h"), (24,"h"), (7,"d")]
  |       + time_since_last_txn
  |
  +---> DeviationFeatures.transform(X)
  |       z-score = (amount - user_mean) / user_std
  |       ratio  = amount / user_mean
  |       (global fallback for cold-start users)
  |
  [concatenated horizontally]
      |
      v
(optional) SHAPSelector.transform(X_features) --> top N features by SHAP importance
      |
      v
(optional) StandardScaler.transform(X_selected) --> scaled features
      |
      v
Output: pd.DataFrame with engineered feature columns
      |
      v
(optional) pipeline.save(filepath)  -->  joblib dump for deployment
```

### 5. CONFIGURATION & PARAMETERS

| Parameter | Location | Default | Notes |
|---|---|---|---|
| `user_col` | `pipeline.py:69` | `"user_id"` | User identifier column |
| `merchant_col` | `pipeline.py:70` | `"merchant_id"` | Merchant identifier column |
| `datetime_col` | `pipeline.py:71` | `"timestamp"` | Timestamp column |
| `amount_col` | `pipeline.py:72` | `"amount"` | Amount column |
| `time_windows` | `pipeline.py:73` | `[(1,"h"), (24,"h"), (7,"d")]` | Velocity windows |
| `velocity_features` | `pipeline.py:87` | `["count", "sum", "mean"]` | Aggregation types |
| `merchant_alpha` | `pipeline.py:89` | `1.0` | Bayesian smoothing alpha |
| `merchant_beta` | `pipeline.py:90` | `1.0` | Bayesian smoothing beta |
| `use_shap_selection` | `pipeline.py:91` | `False` | Enable SHAP feature selection |
| `n_features` | `pipeline.py:92` | `20` | Number of features to keep (SHAP) |
| `scale_features` | `pipeline.py:93` | `False` | Enable StandardScaler |

### 6. EXTERNAL DEPENDENCIES

| Package | Version | Purpose |
|---|---|---|
| pandas | >=2.0.0 | DataFrame operations |
| numpy | >=1.24.0 | Numerical operations |
| scikit-learn | >=1.3.0 | Pipeline, FeatureUnion, BaseEstimator, TransformerMixin, StandardScaler |
| joblib | >=1.3.0 | Pipeline serialization |
| shap | >=0.42.0 | TreeExplainer for feature selection |
| pytest | >=7.4.0 | Testing |
| matplotlib | >=3.7.0 | Optional visualization |

### 7. KNOWN ISSUES / LIMITATIONS

- `MerchantRiskFeatures` is defined but not included in the default `_build_pipeline()` FeatureUnion (only velocity + deviation are wired); users must extend the pipeline manually.
- `SHAPSelector` assumes a tree-based model is available at feature selection time; will fail for non-tree models.
- `VelocityFeatures` performs per-user groupby operations which can be memory-intensive for large datasets without chunking.
- `DeviationFeatures` uses global mean/std fallback for unseen users, which may mask anomalous behavior in cold-start scenarios.
- The `fit()` method in `FraudFeaturePipeline` has an unreachable code path at lines 169-179 (SHAP fitting inside the `y is None` branch but guarded by `y is not None`).

### 8. TESTING

| Test File | Covers |
|---|---|
| `tests/test_velocity_features.py` | VelocityFeatures transform correctness |
| `tests/test_velocity_features_unittest.py` | VelocityFeatures edge cases (unittest style) |
| `tests/test_deviation_features.py` | DeviationFeatures z-scores, ratios |
| `tests/test_deviation_features_unittest.py` | DeviationFeatures edge cases (unittest style) |
| `tests/test_merchant_features.py` | MerchantRiskFeatures Bayesian smoothing |
| `tests/test_merchant_features_unittest.py` | MerchantRiskFeatures edge cases |
| `tests/test_pipeline.py` | Full pipeline fit/transform/save/load |
| `tests/test_pipeline_unittest.py` | Pipeline edge cases |

Run: `pytest tests/ --cov=src`

### 9. QUICK MODIFICATION GUIDE

| Want to... | Change in... |
|---|---|
| Add a new time window | `pipeline.py:73` -- append tuple to `time_windows` list |
| Add merchant features to pipeline | `pipeline.py:_build_pipeline()` -- add `("merchant", MerchantRiskFeatures(...))` to `FeatureUnion.transformer_list` |
| Change Bayesian smoothing priors | `pipeline.py:89-90` -- adjust `merchant_alpha` / `merchant_beta` |
| Add a new transformer | Create class in `src/transformers/`, implement `fit()`/`transform()`/`get_feature_names_out()`, add to `_build_pipeline()` |
| Change SHAP selection count | `pipeline.py:92` -- adjust `n_features` |
| Enable scaling by default | `pipeline.py:93` -- set `scale_features=True` |

### 10. CODE SNIPPETS

**Pipeline construction** (`src/pipeline.py:95-136`):
```python
def _build_pipeline(self) -> Pipeline:
    feature_union = FeatureUnion(
        transformer_list=[
            ("velocity", VelocityFeatures(
                user_col=self.user_col,
                datetime_col=self.datetime_col,
                amount_col=self.amount_col,
                time_windows=self.time_windows,
                features=self.velocity_features,
            )),
            ("deviation", DeviationFeatures(
                user_col=self.user_col,
                features=self.deviation_features,
            )),
        ]
    )
    steps = [("features", feature_union)]
    if self.use_shap_selection:
        steps.append(("shap_selection", SHAPSelector(n_features=self.n_features)))
    if self.scale_features:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)
```

---

## P-04: fraud_scoring_api (Day 4 | Score 10/10)

### 1. PURPOSE

Production-grade real-time fraud scoring REST API built with FastAPI. Accepts single or batch transaction requests (up to 1000), returns fraud probability, risk tier (LOW/MEDIUM/HIGH/CRITICAL), and top risk factors. Features include API key authentication, token-bucket rate limiting via Redis, SHA256-based prediction caching with configurable TTL, security headers middleware, structured JSON logging, and Docker deployment with multi-stage build.

### 2. ARCHITECTURE

```
fraud_scoring_api/
  app/
    main.py                     # FastAPI app factory, lifespan, middleware, exception handlers
    api/
      routes.py                 # POST /predict, POST /batch_predict, GET /model_info, GET /health
      dependencies.py           # Dependency injection: get_predictor, get_redis_cache, check_rate_limit
    models/
      predictor.py              # FraudPredictor: predict_single(), predict_batch(), risk_tier logic
      schemas.py                # Pydantic v2: TransactionRequest, PredictionResponse, BatchPredictionResponse
    core/
      config.py                 # Settings(BaseSettings) with env var support
      security.py               # verify_api_key() dependency, get_security_headers()
    services/
      cache.py                  # RedisCache: async get/set with SHA256 feature hashing, TTL
      rate_limiter.py           # RateLimiter: token bucket algorithm via Redis
      model_loader.py           # ModelLoader: lazy load, reload, unload via joblib
    utils/
      helpers.py                # compute_feature_hash()
  tests/
    test_api.py
    test_auth.py
    test_cache.py
    test_predictor.py
  Dockerfile                    # Multi-stage build, non-root user, health check
  requirements.txt
```

### 3. KEY COMPONENTS

| File (rel. path) | Class / Function | Lines | Role |
|---|---|---|---|
| `app/main.py` | FastAPI app | -- | App factory with lifespan manager, CORS middleware, security headers middleware, request ID middleware |
| `app/api/routes.py` | `predict()` | 51-126 | POST `/predict` -- single transaction scoring with cache check |
| `app/api/routes.py` | `batch_predict()` | 141-194 | POST `/batch_predict` -- up to 1000 transactions |
| `app/api/routes.py` | `model_info()` | 207-238 | GET `/model_info` -- model metadata |
| `app/api/routes.py` | `health_check()` | 250-281 | GET `/health` -- model + Redis status |
| `app/models/predictor.py` | `FraudPredictor` | -- | Wraps model + preprocessing pipeline; returns fraud_probability, risk_tier, risk_factors, latency_ms |
| `app/models/schemas.py` | `TransactionRequest` | -- | Pydantic v2: transaction_id, user_id, merchant_id, amount, timestamp, etc. |
| `app/models/schemas.py` | `PredictionResponse` | -- | fraud_probability, risk_tier (Literal["LOW","MEDIUM","HIGH","CRITICAL"]), top_risk_factors, latency_ms |
| `app/core/config.py` | `Settings(BaseSettings)` | -- | API_HOST, API_PORT, MODEL_PATH, REDIS_HOST/PORT, API_KEYS, RATE_LIMIT_REQUESTS=100 |
| `app/core/security.py` | `verify_api_key()` | -- | FastAPI dependency: validates X-API-Key header against allowed keys |
| `app/services/cache.py` | `RedisCache` | -- | Async Redis: SHA256 feature hashing for cache keys, configurable TTL |
| `app/services/rate_limiter.py` | `RateLimiter` | -- | Token bucket via Redis: RATE_LIMIT_REQUESTS per window |
| `app/services/model_loader.py` | `ModelLoader` | -- | Lazy loading, reload, unload model via joblib |

### 4. DATA FLOW

```
Client  --POST /predict-->  [API Key Check]  --verify_api_key()--> [Rate Limiter]
      |
      v
TransactionRequest (Pydantic validation)
      |
      v
RedisCache.get_prediction(txn_id, features_dict)  -- cache hit? --> return cached PredictionResponse
      |  (miss)
      v
FraudPredictor.predict_single(request)
      |
      +---> extract features from request
      +---> model.predict_proba()
      +---> classify risk_tier (LOW < 0.3, MEDIUM < 0.6, HIGH < 0.85, CRITICAL >= 0.85)
      +---> identify top risk_factors
      +---> measure latency_ms
      |
      v
PredictionResponse  -->  RedisCache.set_prediction()  -->  return to client

Batch:  Client  --POST /batch_predict-->  [Auth + Rate Limit]  -->  FraudPredictor.predict_batch()  -->  BatchPredictionResponse
```

### 5. CONFIGURATION & PARAMETERS

| Parameter | Location | Default | Notes |
|---|---|---|---|
| `API_HOST` | `core/config.py` | env var | Server bind address |
| `API_PORT` | `core/config.py` | env var | Server port |
| `MODEL_PATH` | `core/config.py` | env var | Path to joblib model file |
| `REDIS_HOST` | `core/config.py` | env var | Redis server host |
| `REDIS_PORT` | `core/config.py` | env var | Redis server port |
| `API_KEYS` | `core/config.py` | env var | Comma-separated valid API keys |
| `RATE_LIMIT_REQUESTS` | `core/config.py` | `100` | Max requests per window |
| `app_version` | `core/config.py` | -- | Model/API version string |
| Batch max size | `schemas.py` | `1000` | Maximum transactions per batch request |
| Cache TTL | `services/cache.py` | configurable | Prediction cache duration |

### 6. EXTERNAL DEPENDENCIES

| Package | Version | Purpose |
|---|---|---|
| fastapi | ==0.109.0 | Web framework |
| uvicorn[standard] | ==0.27.0 | ASGI server |
| pydantic | ==2.5.3 | Request/response validation |
| pydantic-settings | ==2.1.0 | Settings from env vars |
| redis | ==5.0.1 | Caching and rate limiting |
| scikit-learn | ==1.4.0 | Feature preprocessing |
| xgboost | ==2.0.3 | Fraud detection model |
| joblib | ==1.3.2 | Model serialization |
| numpy | ==1.26.3 | Numerical operations |
| pandas | ==2.1.4 | Data handling |
| httpx | ==0.26.0 | Test client |
| pytest-asyncio | ==0.23.3 | Async test support |

### 7. KNOWN ISSUES / LIMITATIONS

- API keys are stored in plain text via environment variables; no hashing or rotation mechanism.
- The `startup_events()` function in `routes.py:285-306` calls `get_redis_cache()` and `get_predictor()` directly rather than through the dependency injection system, which can cause initialization order issues.
- Batch predictions (`/batch_predict`) are not cached, unlike single predictions.
- No request body size limit beyond the 1000-transaction batch cap.
- Rate limiter uses a fixed window (not sliding window), which can allow bursts at window boundaries.
- The Docker health check hits `/api/v1/health` but does not validate response body content.

### 8. TESTING

| Test File | Covers |
|---|---|
| `tests/test_api.py` | Route integration tests (predict, batch, health) |
| `tests/test_auth.py` | API key verification, missing/invalid keys |
| `tests/test_cache.py` | Redis cache get/set, TTL, hash computation |
| `tests/test_predictor.py` | FraudPredictor predict_single, predict_batch, risk tier classification |

Run: `pytest tests/ --cov=app`

### 9. QUICK MODIFICATION GUIDE

| Want to... | Change in... |
|---|---|
| Add a new endpoint | `app/api/routes.py` -- add new `@router` decorated function |
| Change risk tier thresholds | `app/models/predictor.py` -- modify threshold logic in `predict_single()` |
| Add request fields | `app/models/schemas.py:TransactionRequest` -- add Pydantic fields |
| Change rate limit | `app/core/config.py:RATE_LIMIT_REQUESTS` or env var |
| Switch from Redis to in-memory cache | Replace `app/services/cache.py` implementation |
| Add model versioning | `app/services/model_loader.py` -- add version tracking to `load_model()` |

### 10. CODE SNIPPETS

**Single prediction endpoint** (`app/api/routes.py:51-126`):
```python
@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: TransactionRequest,
    api_key: str = Depends(verify_api_key),
    rate_limit_ok: bool = Depends(check_rate_limit),
    predictor: FraudPredictor = Depends(get_predictor_with_check),
    cache: RedisCache = Depends(get_redis_cache),
) -> PredictionResponse:
    features_dict = {
        "transaction_id": request.transaction_id,
        "user_id": request.user_id,
        "merchant_id": request.merchant_id,
        "amount": request.amount,
        "timestamp": request.timestamp.isoformat(),
    }
    cached_prediction = await cache.get_prediction(request.transaction_id, features_dict)
    if cached_prediction:
        return PredictionResponse(**cached_prediction)
    result = predictor.predict_single(request)
    response = PredictionResponse(
        transaction_id=request.transaction_id,
        fraud_probability=result["fraud_probability"],
        risk_tier=result["risk_tier"],
        top_risk_factors=result["risk_factors"],
        model_version=settings.app_version,
        latency_ms=result["latency_ms"],
    )
    await cache.set_prediction(request.transaction_id, features_dict, response.model_dump())
    return response
```

---

## P-05: lstm_fraud_detection (Day 5 | Score 9/10)

### 1. PURPOSE

Sequential fraud detection using a bidirectional LSTM with multi-head attention. Processes variable-length transaction sequences per user via `pack_padded_sequence`, applies multi-head self-attention with LayerNorm and residual connections, and classifies the final attended representation through a feedforward head. Supports temporal train/val/test splitting (no data leakage), class weighting for imbalanced data, early stopping, checkpoint management, and ONNX export.

### 2. ARCHITECTURE

```
lstm_fraud_detection/
  src/
    models/
      lstm_attention.py         # LSTMAttentionClassifier(nn.Module), create_lstm_model()
      baseline.py               # BaselineMLP, BaselineMLPWithAttention
    data/
      dataset.py                # FraudSequenceDataset(Dataset), collate_fn(), create_packed_sequence()
      preprocessing.py          # create_user_sequences(), temporal_split(), scale_features(), prepare_data()
    training/
      trainer.py                # Trainer: BCE loss, Adam, ReduceLROnPlateau, gradient clipping, early stopping
      metrics.py                # compute_metrics(), find_optimal_threshold(), MetricTracker
  configs/
    config.yaml                 # All hyperparameters
  tests/
    test_attention.py
    test_models.py
  requirements.txt
```

### 3. KEY COMPONENTS

| File (rel. path) | Class / Function | Lines | Role |
|---|---|---|---|
| `src/models/lstm_attention.py` | `LSTMAttentionClassifier(nn.Module)` | 15-282 | Bidirectional LSTM + MultiheadAttention + LayerNorm + classifier head |
| `src/models/lstm_attention.py` | `forward(padded_sequences, lengths, return_attention)` | 85-168 | Pack, LSTM, unpack, attention mask, multi-head attention, residual + LayerNorm, extract last valid position, classify |
| `src/models/lstm_attention.py` | `extract_attention_weights()` | 170-188 | Returns attention weights for visualization |
| `src/models/lstm_attention.py` | `get_embeddings()` | 215-281 | Extracts pre-classification embeddings for analysis |
| `src/models/lstm_attention.py` | `create_lstm_model(input_dim, config)` | 284-305 | Factory function from config dict |
| `src/models/baseline.py` | `BaselineMLP` | -- | Uses only last transaction (no sequence) |
| `src/models/baseline.py` | `BaselineMLPWithAttention` | -- | Simple attention over sequence (baseline comparison) |
| `src/data/dataset.py` | `FraudSequenceDataset(Dataset)` | -- | PyTorch Dataset for padded sequences |
| `src/data/dataset.py` | `collate_fn()` | -- | Custom collate for DataLoader: pads sequences, returns lengths |
| `src/data/preprocessing.py` | `create_user_sequences(df, max_len)` | -- | Sliding window per user to create sequences |
| `src/data/preprocessing.py` | `temporal_split(sequences, train/val/test ratios)` | -- | Chronological split (no shuffling) to prevent data leakage |
| `src/data/preprocessing.py` | `scale_features(X_train, X_val, X_test)` | -- | Fit scaler on train only, transform all |
| `src/data/preprocessing.py` | `compute_class_weights(y)` | -- | Inverse frequency weighting |
| `src/training/trainer.py` | `Trainer` | -- | BCE loss, Adam, ReduceLROnPlateau, gradient clipping (max_norm), early stopping, checkpoint save/load |
| `src/training/metrics.py` | `compute_metrics(y_true, y_prob)` | -- | auc_pr, auc_roc, accuracy, precision, recall, f1, confusion_matrix |
| `src/training/metrics.py` | `find_optimal_threshold(y_true, y_prob)` | -- | Optimizes F1 score over threshold grid |
| `src/training/metrics.py` | `MetricTracker` | -- | Tracks and logs metrics across epochs |

### 4. DATA FLOW

```
Raw transactions DataFrame (user_id, timestamp, features..., is_fraud)
      |
      v
create_user_sequences(df, max_sequence_length=10)
      |  [sliding window per user, sorted by timestamp]
      v
temporal_split(sequences, train=0.7, val=0.15, test=0.15)
      |  [chronological, no shuffle]
      v
scale_features(X_train, X_val, X_test)  [fit on train only]
      |
      v
FraudSequenceDataset(sequences, labels)  -->  DataLoader(collate_fn=collate_fn)
      |
      v
Trainer.train(train_loader, val_loader)
      |
      +---> per batch:
      |       padded_sequences, lengths = batch
      |       predictions, _ = model(padded_sequences, lengths)
      |       loss = BCELoss(weight=class_weights)(predictions, targets)
      |       loss.backward()
      |       clip_grad_norm_(model.parameters(), max_norm)
      |       optimizer.step()
      |
      +---> per epoch:
      |       val_metrics = compute_metrics(y_true, y_prob)
      |       scheduler.step(val_loss)
      |       early_stopping check -> save best checkpoint
      |
      v
Trainer.evaluate(test_loader)  -->  compute_metrics()  -->  find_optimal_threshold()
      |
      v
(optional) ONNX export
```

### 5. CONFIGURATION & PARAMETERS

| Parameter | Location | Default | Notes |
|---|---|---|---|
| `max_sequence_length` | `configs/config.yaml` | `10` | Max transactions per sequence |
| `hidden_dim` | `configs/config.yaml` | `128` | LSTM hidden dimension |
| `num_layers` | `configs/config.yaml` | `2` | LSTM layer count |
| `num_heads` | `configs/config.yaml` | `4` | Multi-head attention heads |
| `dropout` | `configs/config.yaml` | `0.3` | Dropout probability |
| `bidirectional` | `configs/config.yaml` | `true` | Bidirectional LSTM |
| `batch_size` | `configs/config.yaml` | `64` | Training batch size |
| `num_epochs` | `configs/config.yaml` | `50` | Maximum epochs |
| `lr` (learning_rate) | `configs/config.yaml` | `0.001` | Adam learning rate |
| `early_stopping_patience` | `configs/config.yaml` | `10` | Epochs without improvement before stop |
| `train_ratio / val_ratio / test_ratio` | `configs/config.yaml` | `0.7 / 0.15 / 0.15` | Temporal split ratios |

### 6. EXTERNAL DEPENDENCIES

| Package | Version | Purpose |
|---|---|---|
| torch | >=2.0.0 | LSTM, attention, training |
| numpy | >=1.24.0 | Numerical operations |
| pandas | >=2.0.0 | Data preprocessing |
| scikit-learn | >=1.3.0 | StandardScaler, metrics |
| matplotlib | >=3.7.0 | Training curves, attention viz |
| seaborn | >=0.12.0 | Visualization styling |
| onnx | >=1.14.0 | ONNX model export |
| onnxruntime | >=1.15.0 | ONNX inference |
| pyyaml | >=6.0 | Config loading |
| tqdm | >=4.65.0 | Progress bars |

### 7. KNOWN ISSUES / LIMITATIONS

- `pack_padded_sequence` requires CPU transfer for lengths (`sorted_lengths.cpu()` at line 112), adding overhead per batch on GPU training.
- The attention mask creation at line 129 uses broadcasting which may fail silently for batch size 1.
- `temporal_split()` does not handle users whose transactions span multiple split boundaries -- sequences are split globally, not per-user.
- ONNX export support is listed in requirements but no export function is implemented in the codebase.
- `BaselineMLP` uses only the last transaction in the sequence, losing all sequential context -- this is intentional as a lower-bound baseline.
- No mixed-precision (AMP) training support.

### 8. TESTING

| Test File | Covers |
|---|---|
| `tests/test_attention.py` | LSTMAttentionClassifier attention weight extraction, mask behavior |
| `tests/test_models.py` | Forward pass shapes, predict(), get_embeddings() |

Run: `pytest tests/ --cov=src`

### 9. QUICK MODIFICATION GUIDE

| Want to... | Change in... |
|---|---|
| Change LSTM hidden size | `configs/config.yaml: model.hidden_dim` |
| Add more attention heads | `configs/config.yaml: model.num_heads` (must divide `lstm_output_dim`) |
| Switch to unidirectional LSTM | `configs/config.yaml: model.bidirectional: false` |
| Change sequence length | `configs/config.yaml: data.max_sequence_length` |
| Add a new baseline model | `src/models/baseline.py` -- create new `nn.Module` subclass |
| Customize learning rate schedule | `src/training/trainer.py` -- modify `ReduceLROnPlateau` or replace scheduler |
| Add ONNX export | Implement export function using `torch.onnx.export()` with dummy input |

### 10. CODE SNIPPETS

**LSTM + Attention forward pass** (`src/models/lstm_attention.py:85-168`):
```python
def forward(self, padded_sequences, lengths, return_attention=False):
    batch_size, max_seq_len, _ = padded_sequences.shape
    sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
    sorted_sequences = padded_sequences[sorted_indices]
    packed_input = pack_padded_sequence(
        sorted_sequences, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True
    )
    packed_output, (hidden, cell) = self.lstm(packed_input)
    lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=max_seq_len)
    attention_mask = torch.arange(max_seq_len, device=lengths.device)[None, :] >= sorted_lengths[:, None]
    attended_output, attention_weights = self.attention(
        lstm_output, lstm_output, lstm_output,
        key_padding_mask=attention_mask, need_weights=True
    )
    attended_output = self.layer_norm(attended_output + lstm_output)
    last_indices = (sorted_lengths - 1).clamp(min=0)
    batch_indices = torch.arange(batch_size, device=lstm_output.device)
    final_output = attended_output[batch_indices, last_indices]
    _, unsorted_indices = torch.sort(sorted_indices)
    final_output = final_output[unsorted_indices]
    predictions = self.classifier(final_output)
    if return_attention:
        attention_weights = attention_weights[unsorted_indices]
        return predictions, attention_weights
    return predictions, None
```

---

## P-06: anomaly_detection_benchmark (Day 6 | Score 9/10)

### 1. PURPOSE

Benchmark of four unsupervised anomaly detection models -- Isolation Forest, One-Class SVM, Local Outlier Factor (LOF), and Autoencoder (PyTorch) -- plus two ensemble methods (voting and stacking) for credit card fraud detection. All models inherit from an abstract `AnomalyDetector` base class that enforces a consistent API. Models are trained only on legitimate (Class=0) data. Threshold optimization targets a configurable false positive rate (default 1%). Evaluation includes AUROC, AUPRC, Recall@FPR, and visualization of ROC/PR curves.

### 2. ARCHITECTURE

```
anomaly_detection_benchmark/
  config.yaml                       # Model hyperparameters, ensemble settings, evaluation config
  src/
    models/
      base.py                       # AnomalyDetector(ABC): fit(), predict_anomaly_score(), set_threshold()
      isolation_forest.py           # IsolationForestDetector(AnomalyDetector)
      one_class_svm.py              # OneClassSVMDetector(AnomalyDetector)
      lof.py                        # LOFDetector(AnomalyDetector)
      autoencoder.py                # Autoencoder(nn.Module), AutoencoderDetector(AnomalyDetector)
    ensemble/
      voting.py                     # voting_ensemble(), voting_ensemble_binary()
      stacking.py                   # stacking_ensemble(), stacking_ensemble_cv()
    evaluation/
      metrics.py                    # compute_detection_metrics(), optimize_threshold(), plot_roc_curve(), ...
  tests/
    test_models.py
    test_scoring.py
  requirements.txt
```

### 3. KEY COMPONENTS

| File (rel. path) | Class / Function | Lines | Role |
|---|---|---|---|
| `src/models/base.py` | `AnomalyDetector(ABC)` | 11-108 | Abstract base: `fit()`, `predict_anomaly_score()`, `set_threshold(X_val, target_fpr)`, `predict()`, `fit_predict()` |
| `src/models/isolation_forest.py` | `IsolationForestDetector` | -- | Wraps sklearn `IsolationForest`; inverts scores so higher = more anomalous |
| `src/models/one_class_svm.py` | `OneClassSVMDetector` | -- | Wraps sklearn `OneClassSVM` |
| `src/models/lof.py` | `LOFDetector` | -- | Wraps sklearn `LocalOutlierFactor(novelty=True)` |
| `src/models/autoencoder.py` | `Autoencoder(nn.Module)` | 12-64 | Symmetric encoder/decoder with ReLU activations |
| `src/models/autoencoder.py` | `AutoencoderDetector(AnomalyDetector)` | 67-212 | Uses MSE reconstruction error as anomaly score; Adam optimizer, early stopping |
| `src/ensemble/voting.py` | `voting_ensemble(scores, method)` | -- | `average` (mean scores) or `majority` (binary voting) |
| `src/ensemble/voting.py` | `voting_ensemble_binary(predictions, weights)` | -- | Weighted majority voting |
| `src/ensemble/stacking.py` | `stacking_ensemble(base_scores_train, base_scores_test, y_train)` | 12-68 | LogisticRegression or RandomForest meta-learner on base model scores |
| `src/ensemble/stacking.py` | `stacking_ensemble_cv(base_models, X_train, y_train, X_test)` | 71-159 | K-Fold CV stacking to prevent meta-learner overfitting |
| `src/evaluation/metrics.py` | `compute_detection_metrics(y_true, scores)` | -- | AUROC, AUPRC, optimal threshold |
| `src/evaluation/metrics.py` | `optimize_threshold(scores, y_true, target_fpr)` | -- | Finds threshold achieving target FPR |
| `src/evaluation/metrics.py` | `compute_all_metrics(y_true, scores, threshold)` | -- | Full metric suite |

### 4. DATA FLOW

```
creditcard.csv (or synthetic data)
      |
      v
Split:  Class=0 (legitimate) --> training set
        Class=0 + Class=1    --> test set
      |
      v
[For each model]:
  model.fit(X_train)           # Train on legitimate data only
  model.set_threshold(X_val, target_fpr=0.01)
  scores = model.predict_anomaly_score(X_test)
      |
      v
[Ensemble - Voting]:
  voting_ensemble(all_scores, method='average')  -->  combined_scores
      |
[Ensemble - Stacking]:
  stacking_ensemble_cv(base_models, X_train, y_train, X_test)
      |  meta-learner (LogisticRegression) learns optimal combination
      v
  stacked_scores
      |
      v
compute_detection_metrics(y_true, scores)
optimize_threshold(scores, y_true, target_fpr=0.01)
      |
      v
plot_roc_curve(), plot_precision_recall_curve()
```

### 5. CONFIGURATION & PARAMETERS

| Parameter | Location | Default | Notes |
|---|---|---|---|
| `contamination` | `base.py:14` | `0.1` | Expected outlier fraction |
| `target_fpr` | `config.yaml` / `base.py:53` | `0.01` | Target false positive rate for threshold |
| `n_estimators` (IF) | `config.yaml` | configurable | Isolation Forest trees |
| `kernel` (SVM) | `config.yaml` | `"rbf"` | One-Class SVM kernel |
| `n_neighbors` (LOF) | `config.yaml` | configurable | LOF neighborhood size |
| `hidden_dims` (AE) | `autoencoder.py:91` | `[64, 32]` | Autoencoder encoder layers |
| `latent_dim` (AE) | `autoencoder.py:74` | `16` | Bottleneck dimension |
| `epochs` (AE) | `autoencoder.py:104` | `100` | Training epochs |
| `early_stopping_patience` (AE) | `autoencoder.py:107` | `10` | Early stopping patience |
| `learning_rate` (AE) | `autoencoder.py:106` | `0.001` | Adam LR |
| `meta_model` (stacking) | `stacking.py:18` | `"LogisticRegression"` | Meta-learner type |
| `cv_folds` (stacking) | `stacking.py:19` | `5` | CV folds for meta-features |

### 6. EXTERNAL DEPENDENCIES

| Package | Version | Purpose |
|---|---|---|
| scikit-learn | >=1.3.0 | IsolationForest, OneClassSVM, LOF, LogisticRegression, metrics |
| torch | >=2.0.0 | Autoencoder model |
| numpy | >=1.24.0 | Numerical operations |
| pandas | >=2.0.0 | Data handling |
| matplotlib | >=3.7.0 | ROC/PR curve plots |
| seaborn | >=0.12.0 | Visualization styling |
| pyyaml | >=6.0 | Config loading |
| joblib | >=1.3.0 | Model serialization |
| tqdm | >=4.65.0 | Progress bars |

### 7. KNOWN ISSUES / LIMITATIONS

- `stacking_ensemble_cv()` uses `deepcopy()` of fitted base models per fold (line 119), which refits from the copied state rather than from scratch -- can lead to information leakage.
- `LOFDetector` requires `novelty=True` which means it cannot be used with `fit_predict()` from sklearn directly (handled via the base class wrapper).
- `AutoencoderDetector` defaults to `device="cuda"` but falls back to CPU silently; no warning is issued.
- Score normalization in `stacking_ensemble_cv()` uses training min/max to normalize test scores, which can produce values outside [0,1] if test distribution differs.
- The `IsolationForestDetector` inverts scores to match the "higher = more anomalous" convention, but the exact inversion logic should be verified for edge cases.
- No hyperparameter tuning automation -- all parameters are manually configured.

### 8. TESTING

| Test File | Covers |
|---|---|
| `tests/test_models.py` | All 4 detector classes: fit, predict_anomaly_score, predict, set_threshold |
| `tests/test_scoring.py` | Anomaly score computation, threshold behavior |

Run: `pytest tests/ --cov=src`

### 9. QUICK MODIFICATION GUIDE

| Want to... | Change in... |
|---|---|
| Add a new anomaly detector | Create class in `src/models/` inheriting `AnomalyDetector`; implement `fit()` and `predict_anomaly_score()` |
| Change autoencoder architecture | `src/models/autoencoder.py:15` -- modify `hidden_dims` and `latent_dim` |
| Use a different meta-learner | `src/ensemble/stacking.py` -- add new model type in the `if/elif` block |
| Change target FPR | `config.yaml` evaluation section or `base.py:set_threshold(target_fpr=...)` |
| Add weighted voting | `src/ensemble/voting.py:voting_ensemble_binary()` -- pass custom `weights` |
| Save/load autoencoder | Use `AutoencoderDetector.save_model()` / `load_model()` |

### 10. CODE SNIPPETS

**Abstract base class threshold setting** (`src/models/base.py:53-70`):
```python
def set_threshold(self, X_val: np.ndarray, target_fpr: float = 0.01) -> float:
    if not self.is_fitted:
        raise RuntimeError("Model must be fitted before setting threshold.")
    scores = self.predict_anomaly_score(X_val)
    self.threshold = np.quantile(scores, 1 - target_fpr)
    return self.threshold
```

**Autoencoder anomaly scoring** (`src/models/autoencoder.py:170-190`):
```python
def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
    if not self.is_fitted:
        raise RuntimeError("Model must be fitted before prediction.")
    self.model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(self.device)
        reconstructions = self.model(X_tensor)
        reconstruction_errors = torch.mean((X_tensor - reconstructions) ** 2, dim=1)
    return reconstruction_errors.cpu().numpy()
```

---

## P-07: fraud_model_explainability (Day 7 | Score 10/10)

### 1. PURPOSE

Comprehensive model explainability framework for fraud detection models supporting three explanation methods -- SHAP (TreeExplainer/KernelExplainer/DeepExplainer), LIME, and Partial Dependence Plots (PDP). Uses a factory pattern for explainer creation with automatic model-type-based selection and fallback mechanisms. Generates professional HTML reports with regulatory compliance sections (SR 11-7, EU AI Act). Includes a five-page Streamlit UI for interactive exploration and an end-to-end demo script.

### 2. ARCHITECTURE

```
fraud_model_explainability/
  src/
    explainers/
      base.py                       # BaseExplainer(ABC): explain_local(), explain_global(), get_top_features()
      shap_explainer.py             # SHAPExplainer: Tree/Kernel/Deep explainer, waterfall/summary plots
      lime_explainer.py             # LIMEExplainer: local explanation + aggregated global, HTML export
      pdp_explainer.py              # PDPExplainer: partial dependence, 2-way interaction, nonlinearity detection
    reports/
      generator.py                  # ReportGenerator: professional HTML with CSS, risk factors, regulatory compliance
      templates/
        report_template.html
    api/
      __init__.py                   # Exports create_explainer()
      explainer_factory.py          # ExplainerFactory: model-type mapping, auto-select, fallback, create_multiple()
    utils/
      __init__.py                   # Exports format_risk_factors()
  app/
    streamlit_app.py                # 5-page Streamlit UI
  demo.py                           # End-to-end demo: synthetic data, train XGBoost, explain, report
  tests/
    test_shap.py
    test_lime.py
    test_consistency.py
    test_reports.py
  requirements.txt
```

### 3. KEY COMPONENTS

| File (rel. path) | Class / Function | Lines | Role |
|---|---|---|---|
| `src/explainers/base.py` | `BaseExplainer(ABC)` | -- | Abstract base: `explain_local()`, `explain_global()`, `get_top_features()`, `validate_input()` |
| `src/explainers/shap_explainer.py` | `SHAPExplainer` | 16-239 | Auto-selects TreeExplainer (tree models), DeepExplainer (neural nets), KernelExplainer (fallback); local/global explanations, waterfall/summary plots |
| `src/explainers/shap_explainer.py` | `_initialize_explainer()` | 44-74 | Selects correct SHAP explainer variant based on `model_type` |
| `src/explainers/lime_explainer.py` | `LIMEExplainer` | -- | LIME with fixed random seeds; local explanations; aggregated global (mean of abs local weights over samples) |
| `src/explainers/pdp_explainer.py` | `PDPExplainer` | 17-300 | Partial dependence via `sklearn.inspection.partial_dependence()`; 1-way and 2-way PDP plots; feature ranking by PDP range; nonlinearity detection via correlation |
| `src/explainers/pdp_explainer.py` | `detect_nonlinear_features()` | 249-300 | Identifies non-linear feature relationships using PDP correlation vs linearity threshold |
| `src/api/explainer_factory.py` | `ExplainerFactory` | 12-256 | Factory with `MODEL_TYPE_MAPPING` (xgboost, random_forest, gradient_boosting, neural_network, sklearn_generic, generic); auto-select recommended explainer; fallback mechanism |
| `src/api/explainer_factory.py` | `create_explainer()` | 259-289 | Convenience function: creates factory, calls `factory.create()` |
| `src/api/explainer_factory.py` | `create_multiple()` | 215-256 | Creates multiple explainers for comparison |
| `src/reports/generator.py` | `ReportGenerator` | 19-659 | Generates HTML with embedded CSS; sections: risk summary, risk factors table, global importance, model metadata, regulatory compliance (SR 11-7, EU AI Act); print-friendly |
| `app/streamlit_app.py` | 5 pages | -- | Load Model, Configure Explainer, Explain Transaction, Generate Report, Validation (consistency/speed checks) |
| `demo.py` | `main()` | 198-274 | End-to-end: synthetic data with `make_classification()`, XGBoost training, SHAP explanation, HTML report generation |

### 4. DATA FLOW

```
Trained Model (XGBoost / RandomForest / Neural Network / any sklearn-compatible)
      |
      v
create_explainer(model, model_type, explainer_type, training_data, feature_names)
      |
      v
ExplainerFactory.create()
      |  [checks MODEL_TYPE_MAPPING -> selects recommended or user-specified explainer]
      |  [fallback to LIME if primary fails]
      v
BaseExplainer subclass instance (SHAPExplainer / LIMEExplainer / PDPExplainer)
      |
      +---> explain_local(X_sample, feature_names)
      |       returns Dict[feature_name -> importance_score]
      |
      +---> explain_global(X_dataset, feature_names)
      |       returns Dict[feature_name -> mean_abs_importance]
      |
      v
format_risk_factors(explanation, top_n=5)
      |  [adds direction, impact_level, description]
      v
ReportGenerator.generate_html_report(
    transaction_id, prediction, risk_factors, global_importance, model_metadata
)
      |
      v
HTML report with:
  - Risk score gauge
  - Risk factors table (direction + impact level)
  - Global feature importance chart
  - Model metadata section
  - Regulatory compliance notes (SR 11-7, EU AI Act)
      |
      v
ReportGenerator.save_report(html, filepath)
```

### 5. CONFIGURATION & PARAMETERS

| Parameter | Location | Default | Notes |
|---|---|---|---|
| `default_explainer` | `explainer_factory.py:53` | `'shap'` | Factory default |
| `fallback_explainer` | `explainer_factory.py:54` | `'lime'` | Fallback if primary fails |
| `MODEL_TYPE_MAPPING` | `explainer_factory.py:24-49` | dict | Maps model types to recommended + supported explainers |
| `max_samples` (SHAP global) | `shap_explainer.py:124` | `1000` | Max samples for global SHAP |
| `max_display` (waterfall) | `shap_explainer.py:173` | `10` | Features shown in waterfall plot |
| `max_display` (summary) | `shap_explainer.py:215` | `20` | Features shown in summary plot |
| `linearity_threshold` (PDP) | `pdp_explainer.py:252` | `0.95` | Correlation threshold for linearity detection |
| `n_jobs` (PDP) | `pdp_explainer.py:85` | `-1` | Parallel jobs for PDP computation |
| `top_n` (risk factors) | via `format_risk_factors()` | `5` | Number of risk factors to display |
| `n_samples` (demo) | `demo.py:205` | `1000` | Synthetic dataset size |

### 6. EXTERNAL DEPENDENCIES

| Package | Version | Purpose |
|---|---|---|
| shap | >=0.42.0 | SHAP explanations |
| lime | >=0.2.0 | LIME explanations |
| scikit-learn | >=1.3.0 | PDP, partial_dependence, StandardScaler |
| xgboost | >=2.0.0 | Tree model support |
| tensorflow | >=2.13.0 | DeepExplainer support |
| matplotlib | >=3.7.0 | Visualization |
| seaborn | >=0.12.0 | Plot styling |
| plotly | >=5.14.0 | Interactive plots |
| streamlit | >=1.28.0 | Interactive UI |
| pydantic | >=2.0.0 | Data validation |
| python-dateutil | >=2.8.0 | Date handling |
| pandas | >=2.0.0 | Data handling |
| numpy | >=1.24.0 | Numerical operations |

### 7. KNOWN ISSUES / LIMITATIONS

- `SHAPExplainer._initialize_explainer()` catches all exceptions when `DeepExplainer` fails (line 58) and silently falls back to `KernelExplainer` -- errors are swallowed without logging.
- `PDPExplainer.explain_local()` raises `NotImplementedError` -- PDP is global only; the base class contract forces this method to exist.
- `LIMEExplainer` global explanation is approximated by averaging local explanations over a sample, which is not a true global explanation.
- The `ReportGenerator` at 659 lines builds HTML via string concatenation rather than using a template engine (Jinja2), making report customization fragile.
- `tensorflow>=2.13.0` is listed as a dependency but is only needed if using `DeepExplainer` with Keras models -- adds significant install weight unnecessarily.
- The Streamlit app (`app/streamlit_app.py`) hardcodes `make_classification()` for demo data; no built-in way to load custom production data.

### 8. TESTING

| Test File | Covers |
|---|---|
| `tests/test_shap.py` | SHAPExplainer local/global, waterfall/summary plot generation |
| `tests/test_lime.py` | LIMEExplainer local/global, HTML export |
| `tests/test_consistency.py` | Cross-explainer consistency (SHAP vs LIME feature ranking agreement) |
| `tests/test_reports.py` | ReportGenerator HTML output, required sections present |

Run: `pytest tests/ --cov=src`

### 9. QUICK MODIFICATION GUIDE

| Want to... | Change in... |
|---|---|
| Add a new explainer type | Create class in `src/explainers/` inheriting `BaseExplainer`; add to `ExplainerFactory.EXPLAINER_TYPES` and `_create_explainer()` |
| Support a new model type | `src/api/explainer_factory.py:MODEL_TYPE_MAPPING` -- add entry with recommended + supported explainers |
| Customize HTML report | `src/reports/generator.py:_render_html_report()` -- modify HTML/CSS strings |
| Change regulatory compliance text | `src/reports/generator.py` -- find SR 11-7 / EU AI Act sections |
| Add a new Streamlit page | `app/streamlit_app.py` -- add page to sidebar navigation and implement rendering function |
| Change SHAP explainer variant | Override `_initialize_explainer()` in `SHAPExplainer` or pass different `model_type` |

### 10. CODE SNIPPETS

**SHAP explainer auto-initialization** (`src/explainers/shap_explainer.py:44-74`):
```python
def _initialize_explainer(self) -> None:
    if self.model_type in ['xgboost', 'random_forest', 'gradient_boosting']:
        self.explainer = shap.TreeExplainer(self.model)
    elif self.model_type == 'neural_network':
        if self.training_data is None:
            raise ValueError("training_data is required for neural network explainers.")
        try:
            self.explainer = shap.DeepExplainer(self.model, self.training_data)
        except Exception:
            self.explainer = shap.KernelExplainer(self.model.predict, self.training_data)
    else:
        if self.training_data is None:
            raise ValueError("training_data is required for this model type.")
        self.explainer = shap.KernelExplainer(self.model.predict, self.training_data)
```

**Factory pattern with fallback** (`src/api/explainer_factory.py:72-142`):
```python
def create(self, model, model_type, explainer_type=None, training_data=None, feature_names=None, **kwargs):
    if explainer_type is None:
        explainer_type = self._get_recommended_explainer(model_type)
    if not self._is_explainer_supported(model_type, explainer_type):
        raise ValueError(
            f"Explainer '{explainer_type}' is not supported for model type '{model_type}'."
        )
    try:
        explainer = self._create_explainer(
            model=model, model_type=model_type, explainer_type=explainer_type,
            training_data=training_data, feature_names=feature_names, **kwargs
        )
        return explainer
    except Exception as e:
        if explainer_type != self.fallback_explainer:
            if self._is_explainer_supported(model_type, self.fallback_explainer):
                return self.create(
                    model=model, model_type=model_type,
                    explainer_type=self.fallback_explainer,
                    training_data=training_data, feature_names=feature_names, **kwargs
                )
        raise RuntimeError(f"Failed to create {explainer_type} explainer: {str(e)}")
```
# CAT-02: Federated Learning Foundations -- PROJECT CARDS

---

## P-08 | Day 8 | fedavg_from_scratch | Score 10/10

### 1. PURPOSE

From-scratch implementation of the Federated Averaging (FedAvg) algorithm as described in McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data" (AISTATS 2017). Implements the full federated training loop including server-side weighted aggregation, client-side local SGD training, IID and non-IID data partitioning via Dirichlet distribution, model serialization utilities, and convergence tracking with visualization. Includes two end-to-end experiments: an MNIST sanity check comparing IID vs non-IID settings, and a fraud detection experiment using synthetic bank transaction data with binary classification.

### 2. ARCHITECTURE

```
fedavg_from_scratch/
|-- requirements.txt
|-- src/
|   |-- __init__.py
|   |-- server.py              # FederatedServer class (343 lines)
|   |-- client.py              # FederatedClient, FederatedClientBinary (297 lines)
|   |-- models.py              # SimpleCNN, MLP, get_model() factory (163 lines)
|   |-- data.py                # Data loading, partitioning, FraudDataset (300 lines)
|   |-- metrics.py             # ConvergenceTracker with plotting (369 lines)
|   |-- utils.py               # Serialization, optimizer, device helpers (156 lines)
|-- experiments/
|   |-- fraud_detection.py     # Full fraud detection FL experiment (386 lines)
|   |-- mnist_sanity_check.py  # MNIST IID vs non-IID comparison (264 lines)
|-- tests/
|   |-- __init__.py
|   |-- test_aggregation.py    # 9 tests for weighted averaging math (256 lines)
|   |-- test_client.py         # 8 tests for client training logic (284 lines)
|   |-- test_serialization.py  # 7 tests for weight serialization (172 lines)
```

### 3. KEY COMPONENTS

| Component | File | Line(s) | Description |
|-----------|------|---------|-------------|
| `FederatedServer` | `src/server.py` | 29-293 | Orchestrates FL rounds: client selection, weight distribution, aggregation, evaluation. Holds global model state. |
| `FederatedServer.aggregate_weights()` | `src/server.py` | 112-165 | Core FedAvg: `w_new = sum(n_k / n_total * w_k)` weighted by each client's sample count. |
| `FederatedServer.federated_round()` | `src/server.py` | 167-225 | Executes one FL round: select clients, distribute model, local train, aggregate, evaluate. |
| `FederatedClient` | `src/client.py` | 22-164 | Performs local SGD training. `local_train()` runs multiple epochs on local data; `evaluate()` computes loss and accuracy. |
| `FederatedClientBinary` | `src/client.py` | 167-252 | Subclass for binary classification (fraud detection). Uses BCELoss with pos_weight and sklearn metrics (AUC-PR, F1). |
| `SimpleCNN` | `src/models.py` | 18-64 | Conv2d -> ReLU -> MaxPool2d -> Conv2d -> ReLU -> MaxPool2d -> FC layers. For MNIST. |
| `MLP` | `src/models.py` | 67-117 | Configurable dense layers with BatchNorm1d, ReLU, Xavier init, Sigmoid output. For fraud detection. |
| `partition_data()` | `src/data.py` | 29-82 | Routes to IID (random shuffle) or non-IID (Dirichlet) partitioning by `strategy` argument. |
| `_non_iid_partition()` | `src/data.py` | 85-138 | Dirichlet(alpha) label distribution across clients. Lower alpha = more heterogeneous. |
| `ConvergenceTracker` | `src/metrics.py` | 27-233 | Records per-round metrics (loss, accuracy, epsilon). `plot_convergence()` generates matplotlib charts. JSON save/load. |
| `serialize_weights()` | `src/utils.py` | 20-42 | Converts `state_dict` tensors to numpy arrays for transmission. |
| `deserialize_weights()` | `src/utils.py` | 45-69 | Converts numpy arrays back to tensors and loads into model. |

### 4. DATA FLOW

```
                    Server (src/server.py)
                    |
        +-----------+-----------+
        |           |           |
     Client 0   Client 1   Client K
  (src/client.py)
        |           |           |
  [local_train()]  ...    [local_train()]
  E epochs SGD    ...    E epochs SGD
        |           |           |
  {weights, n_k}   ...   {weights, n_k}
        |           |           |
        +-----------+-----------+
                    |
          aggregate_weights()
     w_new = SUM( n_k/n_total * w_k )
                    |
              global model updated
                    |
              evaluate() on test set
                    |
           ConvergenceTracker.record()
```

### 5. CONFIGURATION & PARAMETERS

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `num_clients` | `src/server.py` L37 | required | Total number of federated clients |
| `fraction_fit` | `src/server.py` L39 | `1.0` | Fraction of clients selected each round |
| `num_rounds` | `experiments/` | 50 | Number of federated rounds |
| `local_epochs` | `src/client.py` L77 | 5 | SGD epochs per client per round |
| `lr` | `src/client.py` L79 | 0.01 | Client learning rate |
| `batch_size` | `src/client.py` L78 | 32 | Client mini-batch size |
| `alpha` | `src/data.py` L94 | 0.5 | Dirichlet concentration for non-IID |
| `strategy` | `src/data.py` L30 | `"iid"` | Partition strategy: `"iid"` or `"non_iid"` |
| `pos_weight` | `src/client.py` L180 | 10.0 | Positive class weight for imbalanced fraud data |

### 6. EXTERNAL DEPENDENCIES

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.0.0 | Neural network training, autograd |
| `torchvision` | >=0.15.0 | MNIST dataset loading |
| `numpy` | >=1.24.0 | Array operations, weight serialization |
| `matplotlib` | >=3.7.0 | Convergence plots |
| `scikit-learn` | >=1.3.0 | AUC-PR, F1, confusion matrix metrics |
| `pandas` | >=2.0.0 | Fraud data loading |
| `tqdm` | >=4.65.0 | Progress bars |
| `pytest` | >=7.4.0 | Test runner |

### 7. KNOWN ISSUES

- No framework dependency (pure PyTorch) -- production FL would use Flower or PySyft for communication.
- Weight serialization uses numpy conversion; no compression or encryption on the wire.
- `_non_iid_partition()` (data.py L85-138) can produce empty partitions with very low alpha and many clients (no minimum-size guarantee).
- `FederatedClientBinary` pos_weight is hardcoded at 10.0 (client.py L180); should be configurable.
- No client dropout or stragglers simulation; all selected clients complete each round.

### 8. TESTING

```
tests/
|-- test_aggregation.py    # 9 tests: weighted averaging math, equal weights,
|                          #   single client, convergence of weights
|-- test_client.py         # 8 tests: local_train returns weights, loss decreases,
|                          #   evaluate returns metrics, binary client AUC
|-- test_serialization.py  # 7 tests: round-trip serialize/deserialize, weight delta
```

Run: `cd fedavg_from_scratch && pytest tests/ -v`

### 9. QUICK MODIFICATION GUIDE

| Change | File | What to modify |
|--------|------|----------------|
| Add FedProx proximal term | `src/client.py` L90-120 | In `local_train()`, add `mu/2 * ||w - w_global||^2` to loss |
| Change aggregation to FedYogi | `src/server.py` L112-165 | Replace weighted average in `aggregate_weights()` with adaptive server optimizer |
| Add learning rate decay | `src/client.py` L79 | Add `torch.optim.lr_scheduler` after optimizer creation |
| Support new model architecture | `src/models.py` L140-163 | Add class and register in `get_model()` factory dict |
| Add secure aggregation | `src/server.py` L112 | Wrap `aggregate_weights()` with masking before/after |

### 10. CODE SNIPPETS

**Core FedAvg Aggregation** (`src/server.py` lines 112-165):
```python
def aggregate_weights(self, client_results: list[dict]) -> None:
    """Aggregate client weights using FedAvg weighted averaging."""
    total_samples = sum(result['num_samples'] for result in client_results)

    new_state_dict = {}
    for key in client_results[0]['weights'].keys():
        weighted_sum = torch.zeros_like(client_results[0]['weights'][key], dtype=torch.float32)
        for result in client_results:
            weight = result['num_samples'] / total_samples
            weighted_sum += weight * result['weights'][key].float()
        new_state_dict[key] = weighted_sum

    self.global_model.load_state_dict(new_state_dict)
```

**Dirichlet Non-IID Partition** (`src/data.py` lines 85-138):
```python
def _non_iid_partition(targets, num_clients, alpha=0.5):
    """Dirichlet-based non-IID partition."""
    num_classes = len(np.unique(targets))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()
        splits = (proportions * len(class_indices)).astype(int)
        # ... distribute class_indices to clients per splits
```

---

## P-09 | Day 9 | non_iid_partitioner | Score 9/10

### 1. PURPOSE

Comprehensive library for simulating non-IID (non-Independent and Identically Distributed) data distributions across federated learning clients. Implements five distinct partitioning strategies -- IID baseline, label skew via Dirichlet distribution, quantity skew via power law (Pareto), feature skew via KMeans clustering, and a realistic bank simulation with geographic/demographic modeling. Includes visualization tools for analyzing heterogeneity (heatmaps, bar charts) and statistical metrics (entropy, Gini coefficient). Designed as a reusable toolkit for any FL experiment requiring controlled non-IID data splits.

### 2. ARCHITECTURE

```
non_iid_partitioner/
|-- requirements.txt
|-- src/
|   |-- __init__.py
|   |-- partitioner.py                # NonIIDPartitioner facade (276 lines)
|   |-- utils.py                      # Validation, entropy, Gini (142 lines)
|   |-- visualization.py              # Heatmaps, charts, metrics (289 lines)
|   |-- strategies/
|       |-- __init__.py
|       |-- iid.py                    # Random uniform partition (48 lines)
|       |-- label_skew.py             # Dirichlet label distribution (243 lines)
|       |-- feature_skew.py           # KMeans feature clustering (218 lines)
|       |-- quantity_skew.py          # Power law sample allocation (177 lines)
|       |-- realistic_bank.py         # Geographic/demographic simulation (252 lines)
|-- examples/
|   |-- demo.py                       # Demonstrates all 5 strategies (347 lines)
|-- tests/
|   |-- __init__.py
|   |-- test_partitioner.py           # Integration tests (232 lines)
|   |-- test_strategies.py            # Unit tests per strategy (231 lines)
```

### 3. KEY COMPONENTS

| Component | File | Line(s) | Description |
|-----------|------|---------|-------------|
| `NonIIDPartitioner` | `src/partitioner.py` | 20-220 | Facade class routing to strategy implementations. Stores partition results and statistics. |
| `partition_iid()` | `src/partitioner.py` | 56-75 | Delegates to `iid.iid_partition()`. Random uniform split. |
| `partition_label_skew()` | `src/partitioner.py` | 77-110 | Delegates to `label_skew.dirichlet_partition()`. Controls `alpha` concentration. |
| `partition_quantity_skew()` | `src/partitioner.py` | 112-140 | Delegates to `quantity_skew.quantity_skew_partition()`. Power law sample sizes. |
| `partition_feature_skew()` | `src/partitioner.py` | 142-175 | Delegates to `feature_skew.feature_based_partition()`. KMeans clustering on features. |
| `partition_realistic_bank()` | `src/partitioner.py` | 177-210 | Delegates to `realistic_bank.realistic_bank_partition()`. Geographic + demographic simulation. |
| `dirichlet_partition()` | `src/strategies/label_skew.py` | 22-100 | Core Dirichlet: `np.random.dirichlet(alpha * ones)` per class, then split indices proportionally. |
| `feature_based_partition()` | `src/strategies/feature_skew.py` | 24-105 | Runs KMeans/MiniBatchKMeans on feature matrix, assigns clients by cluster membership. |
| `power_law_allocation()` | `src/strategies/quantity_skew.py` | 18-58 | Pareto distribution `s^(-a)` for sample counts, normalized to total dataset size. |
| `realistic_bank_partition()` | `src/strategies/realistic_bank.py` | 30-160 | Assigns samples based on simulated geographic regions, demographic profiles, transaction patterns. |
| `compute_heterogeneity_metrics()` | `src/visualization.py` | 220-260 | Computes label entropy per client, Gini coefficient, Earth Mover's Distance from uniform. |

### 4. DATA FLOW

```
Input: X (features), y (labels), n_clients, strategy, params
                    |
         NonIIDPartitioner(n_clients)
                    |
     partition_{strategy}(X, y, **params)
                    |
         +----------+----------+
         |          |          |
     strategies/   strategies/  ...
      iid.py    label_skew.py
         |          |
         +----------+----------+
                    |
     Returns: dict{ client_id -> np.array(indices) }
                    |
     get_partition_statistics()
         -> per-client: n_samples, label_dist, entropy
                    |
     visualization.py
         -> heatmaps, bar charts, heterogeneity metrics
```

### 5. CONFIGURATION & PARAMETERS

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `n_clients` | `src/partitioner.py` L25 | required | Number of federated clients |
| `random_state` | `src/partitioner.py` L26 | `None` | Reproducibility seed |
| `alpha` | `src/strategies/label_skew.py` L27 | `0.5` | Dirichlet concentration (lower = more skewed) |
| `min_samples` | `src/strategies/label_skew.py` L28 | `10` | Minimum samples per client |
| `power_law_alpha` | `src/strategies/quantity_skew.py` L25 | `1.5` | Pareto shape parameter |
| `n_clusters` | `src/strategies/feature_skew.py` L30 | `n_clients` | Number of KMeans clusters |
| `use_minibatch` | `src/strategies/feature_skew.py` L31 | `True` | Use MiniBatchKMeans for large data |
| `n_regions` | `src/strategies/realistic_bank.py` L35 | `5` | Number of geographic regions |

### 6. EXTERNAL DEPENDENCIES

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | >=1.21.0 | Array operations, Dirichlet sampling |
| `pandas` | >=1.3.0 | Data handling for realistic bank strategy |
| `scikit-learn` | >=1.0.0 | KMeans/MiniBatchKMeans for feature-based partitioning |
| `matplotlib` | >=3.4.0 | Heatmaps, bar charts, visualization |
| `scipy` | >=1.7.0 | Statistical functions, Earth Mover's Distance |

### 7. KNOWN ISSUES

- No PyTorch dependency -- partitioner returns numpy index arrays, not DataLoaders.
- `feature_based_partition()` may produce uneven cluster sizes with high-dimensional data and few clients.
- `realistic_bank_partition()` uses hardcoded geographic region probabilities; not configurable from YAML.
- No streaming/online partitioning support; entire dataset must fit in memory.
- `min_samples` guarantee in label_skew can fail silently when `alpha` is extremely small (< 0.01) with many classes and clients.

### 8. TESTING

```
tests/
|-- test_partitioner.py   # Integration: NonIIDPartitioner class init,
|                         #   all 5 strategies via facade, statistics output
|-- test_strategies.py    # Unit: each strategy function independently,
|                         #   validates partition correctness, no overlap, full coverage
```

Run: `cd non_iid_partitioner && pytest tests/ -v`

### 9. QUICK MODIFICATION GUIDE

| Change | File | What to modify |
|--------|------|----------------|
| Add new partitioning strategy | `src/strategies/` | Create new `.py`, add method to `NonIIDPartitioner` in `partitioner.py` |
| Adjust Dirichlet behavior | `src/strategies/label_skew.py` L22-100 | Modify `dirichlet_partition()` proportions logic |
| Add Wasserstein distance metric | `src/visualization.py` L220-260 | Add to `compute_heterogeneity_metrics()` |
| Make realistic_bank configurable | `src/strategies/realistic_bank.py` L35-50 | Extract hardcoded probabilities to function parameters |
| Support PyTorch DataLoader output | `src/partitioner.py` | Add `to_dataloaders()` method wrapping indices into Subset + DataLoader |

### 10. CODE SNIPPETS

**Dirichlet Label Skew** (`src/strategies/label_skew.py` lines 22-100):
```python
def dirichlet_partition(y, n_clients, alpha=0.5, min_samples=10, random_state=None):
    """Partition using Dirichlet distribution over labels."""
    rng = np.random.RandomState(random_state)
    classes = np.unique(y)
    n_classes = len(classes)
    client_indices = [[] for _ in range(n_clients)]

    for c in classes:
        class_idx = np.where(y == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.repeat(alpha, n_clients))
        proportions = proportions / proportions.sum()
        splits = (proportions * len(class_idx)).astype(int)
        # ... assign splits to clients
    return {i: np.array(idx) for i, idx in enumerate(client_indices)}
```

**Power Law Allocation** (`src/strategies/quantity_skew.py` lines 18-58):
```python
def power_law_allocation(n_clients, total_samples, alpha=1.5, random_state=None):
    """Allocate samples using Pareto/power-law distribution."""
    rng = np.random.RandomState(random_state)
    raw = np.arange(1, n_clients + 1, dtype=float) ** (-alpha)
    rng.shuffle(raw)
    proportions = raw / raw.sum()
    sizes = (proportions * total_samples).astype(int)
    # Redistribute remainder to maintain total
    return sizes
```

---

## P-10 | Day 10 | flower_fraud_detection | Score 10/10

### 1. PURPOSE

Production-grade federated fraud detection system built on the Flower framework (flwr >= 1.11.0). Implements a custom Flower `NumPyClient` with support for FedAvg, FedProx (proximal term), and FedAdam strategies. Uses Hydra/OmegaConf for hierarchical configuration management across strategy, data partitioning, and training parameters. Features a flexible MLP fraud detection model with BatchNorm and weighted BCE loss for imbalanced data. Supports both IID and Dirichlet-based non-IID data partitioning. Integrates with Ray for parallel client simulation via `flwr.simulation.start_simulation()`.

### 2. ARCHITECTURE

```
flower_fraud_detection/
|-- main.py                        # Hydra entry point (62 lines)
|-- requirements.txt
|-- run_simple_tests.py
|-- run_tests.py
|-- config/
|   |-- base.yaml                  # Main config: 10 clients, 20 rounds
|   |-- data/
|   |   |-- iid.yaml               # IID partitioning config
|   |   |-- non_iid.yaml           # Dirichlet partitioning config
|   |-- strategy/
|       |-- fedavg.yaml            # FedAvg strategy config
|       |-- fedprox.yaml           # FedProx strategy config
|       |-- fedadam.yaml           # FedAdam strategy config
|-- src/
|   |-- __init__.py
|   |-- client.py                  # FlClient(NumPyClient) (288 lines)
|   |-- server.py                  # Strategy factory (118 lines)
|   |-- model.py                   # FraudDetectionModel, loss (138 lines)
|   |-- data.py                    # Data gen, IID/Dirichlet partition (254 lines)
|   |-- simulation.py              # Flower simulation runner (180 lines)
|   |-- utils.py
|   |-- strategy/
|       |-- __init__.py
|       |-- base.py
|       |-- fedavg.py              # FedAvgCustom(Strategy) (227 lines)
|       |-- fedprox.py
|       |-- fedadam.py
|-- tests/
|   |-- __init__.py
|   |-- test_client.py             # Client parameter handling tests
|   |-- test_strategy.py           # Strategy aggregation tests
|   |-- test_utils.py
```

### 3. KEY COMPONENTS

| Component | File | Line(s) | Description |
|-----------|------|---------|-------------|
| `FlClient` | `src/client.py` | 28-240 | Extends `fl.client.NumPyClient`. Implements `get_parameters()`, `set_parameters()`, `fit()`, `evaluate()`. Supports FedProx proximal term in `fit()`. |
| `FlClient.fit()` | `src/client.py` | 105-175 | Local training with optional `proximal_mu` for FedProx: `loss += mu/2 * ||w - w_global||^2`. Returns updated numpy weights + num_examples. |
| `FlClient.evaluate()` | `src/client.py` | 177-230 | Evaluates model on local test data. Returns loss, num_examples, and metrics dict (accuracy, AUC, precision, recall). |
| `FraudDetectionModel` | `src/model.py` | 18-95 | Configurable MLP: input_dim -> [hidden_dims] with BatchNorm1d + ReLU + Dropout -> 1 (sigmoid). Xavier initialization. |
| `FraudDetectionLoss` | `src/model.py` | 98-138 | Weighted BCE loss with configurable `pos_weight` for class imbalance. |
| `get_strategy()` | `src/server.py` | 30-90 | Factory returning Flower Strategy (FedAvg/FedProx/FedAdam) based on Hydra config string. |
| `FedAvgCustom` | `src/strategy/fedavg.py` | 22-227 | Full Flower `Strategy` implementation: `configure_fit()`, `aggregate_fit()`, `configure_evaluate()`, `aggregate_evaluate()`. |
| `load_synthetic_fraud_data()` | `src/data.py` | 25-85 | Generates synthetic fraud dataset with configurable features, fraud ratio. |
| `partition_data_dirichlet()` | `src/data.py` | 120-175 | Dirichlet(alpha) non-IID label distribution across clients. |
| `prepare_federated_data()` | `src/data.py` | 178-254 | End-to-end: generate data, partition, create DataLoaders per client. |
| `main()` | `src/simulation.py` | 40-140 | Orchestrates `start_simulation()` with Ray backend, configures server strategy and client function. |

### 4. DATA FLOW

```
main.py (Hydra)
    |
    v
simulation.py::main(cfg)
    |
    +---> prepare_federated_data(cfg)
    |         |-> load_synthetic_fraud_data()
    |         |-> partition_data_{iid|dirichlet}()
    |         |-> returns: {cid: (train_loader, test_loader)}
    |
    +---> get_strategy(cfg) -> FedAvg/FedProx/FedAdam
    |
    +---> start_simulation(
    |         client_fn = lambda cid: FlClient(model, loaders[cid]),
    |         num_clients = cfg.num_clients,
    |         config = ServerConfig(num_rounds),
    |         strategy = strategy,
    |         ray_init_args = {...}
    |     )
    |
    v
  Flower simulation loop:
    Round r:
      Server -> configure_fit() -> client subset
      Clients -> FlClient.fit() -> local SGD + optional proximal term
      Server -> aggregate_fit() -> weighted average
      Server -> configure_evaluate()
      Clients -> FlClient.evaluate() -> loss, accuracy, AUC
      Server -> aggregate_evaluate() -> global metrics
```

### 5. CONFIGURATION & PARAMETERS

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `num_clients` | `config/base.yaml` | `10` | Total number of federated clients |
| `num_rounds` | `config/base.yaml` | `20` | Number of FL rounds |
| `local_epochs` | `config/base.yaml` | `5` | Client-side training epochs per round |
| `batch_size` | `config/base.yaml` | `32` | Client batch size |
| `hidden_dims` | `config/base.yaml` | `[64, 32, 16]` | MLP hidden layer dimensions |
| `optimizer` | `config/base.yaml` | `"adam"` | Client optimizer type |
| `lr` | `config/base.yaml` | `0.001` | Client learning rate |
| `proximal_mu` | `config/strategy/fedprox.yaml` | `0.01` | FedProx proximal term coefficient |
| `alpha` | `config/data/non_iid.yaml` | `0.5` | Dirichlet concentration parameter |
| `pos_weight` | `src/model.py` L105 | `10.0` | BCE positive class weight for fraud detection |

### 6. EXTERNAL DEPENDENCIES

| Package | Version | Purpose |
|---------|---------|---------|
| `flwr` | >=1.11.0 | Flower framework: NumPyClient, Strategy, start_simulation |
| `torch` | >=2.0.0 | Model definition and training |
| `hydra-core` | >=1.3.0 | Hierarchical configuration management |
| `omegaconf` | >=2.3.0 | Config resolution, YAML parsing |
| `ray` | >=2.0.0 | Parallel client simulation backend |
| `scipy` | >=1.10.0 | Dirichlet distribution for non-IID partitioning |
| `tensorboard` | >=2.13.0 | Training visualization |
| `scikit-learn` | >=1.3.0 | Evaluation metrics (AUC-ROC, precision, recall) |

### 7. KNOWN ISSUES

- `FraudDetectionLoss` hardcodes `pos_weight=10.0` at model.py L105; should be driven from config.
- `main.py` uses Hydra `initialize()` with `version_base=None` which may emit deprecation warnings on newer Hydra versions.
- Strategy YAML overrides (fedavg.yaml, fedprox.yaml, fedadam.yaml) must be manually composed; no automatic strategy sweep.
- `start_simulation()` with Ray can fail silently on resource-constrained machines (requires enough RAM for parallel clients).
- No TLS/mTLS or authentication; simulation only (no real network transport).

### 8. TESTING

```
tests/
|-- test_client.py     # Client parameter get/set, fit returns correct shapes,
|                      #   evaluate returns metrics dict
|-- test_strategy.py   # FedAvgCustom aggregation, configure_fit client sampling
|-- test_utils.py      # Utility function tests
```

Run: `cd flower_fraud_detection && pytest tests/ -v` or `python run_simple_tests.py`

### 9. QUICK MODIFICATION GUIDE

| Change | File | What to modify |
|--------|------|----------------|
| Add new strategy (e.g., FedYogi) | `src/strategy/` | Create `fedyogi.py`, add to `get_strategy()` factory in `src/server.py` L30-90, create `config/strategy/fedyogi.yaml` |
| Change model architecture | `src/model.py` L18-95 | Modify `FraudDetectionModel.__init__()` layer definitions |
| Add client-side differential privacy | `src/client.py` L105-175 | In `fit()`, add gradient clipping + noise after `loss.backward()` |
| Use real dataset | `src/data.py` L25-85 | Replace `load_synthetic_fraud_data()` with file loader |
| Add FedProx mu scheduling | `src/client.py` L130 | Pass round number via config dict, adjust `proximal_mu` per round |

### 10. CODE SNIPPETS

**Flower Client fit() with FedProx** (`src/client.py` lines 105-175):
```python
def fit(self, parameters, config):
    self.set_parameters(parameters)
    global_params = [p.clone() for p in self.model.parameters()]

    for epoch in range(self.local_epochs):
        for X_batch, y_batch in self.train_loader:
            self.optimizer.zero_grad()
            output = self.model(X_batch)
            loss = self.loss_fn(output, y_batch)

            # FedProx proximal term
            if self.proximal_mu > 0:
                proximal_term = 0.0
                for p, g_p in zip(self.model.parameters(), global_params):
                    proximal_term += (p - g_p).norm(2) ** 2
                loss += (self.proximal_mu / 2) * proximal_term

            loss.backward()
            self.optimizer.step()

    return self.get_parameters(config={}), len(self.train_loader.dataset), {}
```

**Custom FedAvg Strategy** (`src/strategy/fedavg.py` lines 22-227):
```python
class FedAvgCustom(Strategy):
    def aggregate_fit(self, server_round, results, failures):
        weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                          for _, fit_res in results]
        aggregated = ndarrays_to_parameters(aggregate(weights_results))
        return aggregated, {}
```

---

## P-11 | Day 11 | communication_efficient_fl | Score 10/10

### 1. PURPOSE

Implementation of communication-efficient techniques for federated learning, addressing the primary bottleneck of transmitting large model updates between clients and server. Provides three families of compression: gradient quantization (8-bit, 4-bit, stochastic rounding), gradient sparsification (Top-K, Random-K, threshold), and error feedback (residual accumulation to recover lost information from compression). Includes a Flower-compatible `EfficientFedAvg` strategy wrapper that transparently applies compression to client updates, an `AdaptiveCompressionStrategy` that interpolates compression ratio based on training progress, and a `BandwidthTracker` for measuring actual bytes saved. All techniques are validated against uncompressed baselines with compression ratio and reconstruction error metrics.

### 2. ARCHITECTURE

```
communication_efficient_fl/
|-- requirements.txt
|-- config/
|   |-- compression.yaml           # Compression hyperparameters
|   |-- experiment.yaml            # Experiment settings
|-- src/
|   |-- __init__.py
|   |-- compression/
|   |   |-- __init__.py
|   |   |-- quantizers.py          # 8-bit, 4-bit, stochastic quantization (270 lines)
|   |   |-- sparsifiers.py         # Top-K, Random-K, threshold (216 lines)
|   |   |-- error_feedback.py      # ErrorFeedback, MultiLayerErrorFeedback (274 lines)
|   |   |-- utils.py               # Compression utility functions
|   |-- strategies/
|   |   |-- __init__.py
|   |   |-- efficient_fedavg.py    # EfficientFedAvg(FedAvg) + AdaptiveCompression (327 lines)
|   |   |-- compression_wrapper.py # Generic compression wrapper
|   |-- metrics/
|       |-- __init__.py
|       |-- bandwidth_tracker.py   # BandwidthTracker, BandwidthComparator (337 lines)
|       |-- compression_metrics.py # Compression quality metrics
|-- experiments/
|   |-- baseline.py                # Baseline comparison experiment
|-- tests/
|   |-- __init__.py
|   |-- test_quantizers.py         # 8-bit, 4-bit, stochastic quantization tests
|   |-- test_sparsifiers.py        # Top-K, Random-K, threshold tests
|   |-- test_error_feedback.py     # Error feedback convergence tests
```

### 3. KEY COMPONENTS

| Component | File | Line(s) | Description |
|-----------|------|---------|-------------|
| `quantize_8bit()` | `src/compression/quantizers.py` | 20-65 | Uniform 8-bit quantization: maps [min, max] to [0, 255]. Returns uint8 + scale + zero_point. |
| `dequantize_8bit()` | `src/compression/quantizers.py` | 68-88 | Reconstructs float32 from uint8 + scale + zero_point. |
| `quantize_4bit()` | `src/compression/quantizers.py` | 91-145 | 4-bit quantization (16 levels). Two values per byte for 8x compression. |
| `stochastic_quantize()` | `src/compression/quantizers.py` | 175-225 | Probabilistic rounding: `P(round_up) = (x - floor(x)) / step`. Unbiased estimator. |
| `top_k_sparsify()` | `src/compression/sparsifiers.py` | 20-68 | Keeps top K elements by magnitude. Uses `np.argpartition()` for O(n) selection. |
| `random_k_sparsify()` | `src/compression/sparsifiers.py` | 71-105 | Random selection of K elements (baseline for Top-K comparison). |
| `threshold_sparsify()` | `src/compression/sparsifiers.py` | 108-140 | Keeps elements above absolute threshold. |
| `top_k_sparsify_percentage()` | `src/compression/sparsifiers.py` | 143-165 | Convenience: converts percentage to K, delegates to `top_k_sparsify()`. |
| `ErrorFeedback` | `src/compression/error_feedback.py` | 22-120 | Accumulates compression residual: `e_t = (g_t + e_{t-1}) - C(g_t + e_{t-1})`. Compresses corrected gradient. |
| `MultiLayerErrorFeedback` | `src/compression/error_feedback.py` | 123-210 | Per-layer residual tracking. Each layer has independent error state. |
| `EfficientFedAvg` | `src/strategies/efficient_fedavg.py` | 25-180 | Extends Flower `FedAvg`. Wraps `aggregate_fit()` to decompress client updates before aggregation. |
| `AdaptiveCompressionStrategy` | `src/strategies/efficient_fedavg.py` | 183-327 | Linearly interpolates compression ratio from high (early rounds) to low (late rounds) based on progress. |
| `BandwidthTracker` | `src/metrics/bandwidth_tracker.py` | 30-180 | Records bytes per round (uplink/downlink). `get_savings()` computes % reduction vs uncompressed baseline. |
| `BandwidthComparator` | `src/metrics/bandwidth_tracker.py` | 183-337 | Side-by-side comparison of multiple compression techniques. Generates summary tables. |

### 4. DATA FLOW

```
Client local training
    |
    v
gradient update (float32 tensor)
    |
    +---> ErrorFeedback.compress_and_update(gradient)
    |         |-> corrected = gradient + residual
    |         |-> compressed = compressor(corrected)
    |         |-> residual = corrected - decompress(compressed)
    |         |-> return compressed
    |
    +---> Compression (one of):
    |       quantize_8bit() -> uint8 + metadata     (4x compression)
    |       quantize_4bit() -> packed nibbles        (8x compression)
    |       top_k_sparsify() -> indices + values     (100/K x compression)
    |       stochastic_quantize() -> levels + scale  (unbiased, ~4x)
    |
    +---> BandwidthTracker.record_upload(compressed_bytes)
    |
    v
Server: EfficientFedAvg.aggregate_fit()
    |-> decompress each client update
    |-> weighted average (standard FedAvg)
    |-> BandwidthTracker.record_download(aggregated_bytes)
    |
    v
Updated global model
```

### 5. CONFIGURATION & PARAMETERS

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `top_k.percentages` | `config/compression.yaml` L6 | `[1, 5, 10, 20, 50]` | Top-K sparsification percentages to evaluate |
| `quantization.bits` | `config/compression.yaml` L22 | `[4, 8]` | Quantization bit widths |
| `quantization.stochastic` | `config/compression.yaml` L25 | `true` | Use stochastic rounding |
| `error_feedback.enabled` | `config/compression.yaml` L29 | `true` | Enable error feedback (residual accumulation) |
| `error_feedback.reset_frequency` | `config/compression.yaml` L32 | `0` | Reset residual every N rounds (0 = never) |
| `threshold.values` | `config/compression.yaml` L17 | `[0.01-0.5]` | Threshold sparsification relative values |
| `combined.enable` | `config/compression.yaml` L38 | `true` | Enable combined sparsification + quantization |

### 6. EXTERNAL DEPENDENCIES

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | >=1.21.0 | Array operations, argpartition for Top-K |
| `torch` | >=2.0.0 | Tensor operations, model parameters |
| `flwr` | >=1.0.0 | FedAvg base strategy, parameters conversion |
| `pytest` | >=7.0.0 | Test runner |
| `matplotlib` | >=3.5.0 | Compression ratio visualization |
| `pyyaml` | >=6.0 | Config file parsing |

### 7. KNOWN ISSUES

- `quantize_4bit()` packs two values per byte but does not handle odd-length tensors (silent truncation of last element).
- `top_k_sparsify()` uses `np.argpartition()` which is O(n) average but O(n^2) worst case.
- `ErrorFeedback` stores residual in float32; no memory savings for the residual itself.
- `EfficientFedAvg` assumes all clients use the same compression; no per-client adaptive compression.
- `AdaptiveCompressionStrategy` uses linear interpolation only; no support for step schedules or exponential decay.
- `stochastic_quantize()` requires knowing min/max of the tensor; outliers can degrade quality.

### 8. TESTING

```
tests/
|-- test_quantizers.py      # 8-bit round-trip fidelity, 4-bit compression ratio,
|                           #   stochastic quantization unbiasedness (mean of many rounds)
|-- test_sparsifiers.py     # Top-K keeps exactly K elements, correct indices,
|                           #   Random-K randomness, threshold correctness
|-- test_error_feedback.py  # Residual accumulates correctly, multi-round convergence
```

Run: `cd communication_efficient_fl && pytest tests/ -v`

### 9. QUICK MODIFICATION GUIDE

| Change | File | What to modify |
|--------|------|----------------|
| Add 2-bit quantization | `src/compression/quantizers.py` | Add `quantize_2bit()` / `dequantize_2bit()` functions (4 levels) |
| Add exponential compression schedule | `src/strategies/efficient_fedavg.py` L183-327 | Modify `AdaptiveCompressionStrategy` interpolation formula |
| Add per-layer compression | `src/compression/error_feedback.py` L123-210 | Extend `MultiLayerErrorFeedback` with layer-specific compression configs |
| Track convergence vs compression | `src/metrics/bandwidth_tracker.py` | Add accuracy tracking alongside bandwidth in `BandwidthTracker` |
| Add SignSGD compression | `src/compression/` | New file `sign_compressor.py`: keep only sign bits (32x compression) |

### 10. CODE SNIPPETS

**Top-K Sparsification** (`src/compression/sparsifiers.py` lines 20-68):
```python
def top_k_sparsify(tensor, k):
    """Keep top-K elements by magnitude, zero out rest."""
    flat = tensor.flatten()
    abs_flat = np.abs(flat)

    if k >= len(flat):
        return flat.copy(), np.arange(len(flat))

    # O(n) selection using argpartition
    top_k_indices = np.argpartition(abs_flat, -k)[-k:]

    sparse = np.zeros_like(flat)
    sparse[top_k_indices] = flat[top_k_indices]

    return sparse, top_k_indices
```

**Error Feedback** (`src/compression/error_feedback.py` lines 22-120):
```python
class ErrorFeedback:
    def __init__(self, compressor_fn):
        self.compressor = compressor_fn
        self.residual = None

    def compress_and_update(self, gradient):
        if self.residual is not None:
            corrected = gradient + self.residual
        else:
            corrected = gradient

        compressed = self.compressor(corrected)
        decompressed = self.decompress(compressed)
        self.residual = corrected - decompressed

        return compressed
```

---

## P-12 | Day 12 | cross_silo_bank_fl | Score 8/10

### 1. PURPOSE

Cross-silo federated learning simulation for multi-bank fraud detection. Models a realistic banking consortium where 5 banks (with distinct profiles -- retail, investment, fintech, regional, global) collaboratively train a shared fraud detection model without sharing raw transaction data. Features synthetic transaction generation with 6 fraud types (card_present, card_not_present, account_takeover, synthetic_identity, cross_border, internal), per-bank metric tracking via a custom Flower strategy, and comparison against local-only and centralized baselines. Includes secure aggregation simulation with additive masking.

### 2. ARCHITECTURE

```
cross_silo_bank_fl/
|-- main.py                                # Full 8-step simulation pipeline (452 lines)
|-- requirements.txt
|-- config/
|   |-- simulation_config.yaml             # FL parameters, model architecture
|   |-- bank_profiles.yaml                 # 5 bank profiles with characteristics
|-- src/
|   |-- models/
|   |   |-- __init__.py
|   |   |-- fraud_nn.py                    # FraudNN, SimplifiedFraudNN (264 lines)
|   |   |-- training_utils.py             # Training helpers
|   |-- federation/
|   |   |-- __init__.py
|   |   |-- strategy.py                    # PerBankMetricStrategy(FedAvg) (263 lines)
|   |   |-- flower_client.py              # Bank Flower client
|   |   |-- secure_aggregation.py         # Additive masking simulation
|   |-- data_generation/
|   |   |-- __init__.py
|   |   |-- fraud_generator.py             # FraudGenerator with 6 fraud types (304 lines)
|   |   |-- transaction_generator.py       # Transaction data generation
|   |   |-- bank_profile.py               # BankProfile dataclass
|   |-- preprocessing/
|   |   |-- __init__.py
|   |   |-- feature_engineering.py         # Feature extraction
|   |   |-- partitioner.py                # Bank-based data partitioning
|   |-- evaluation/
|   |   |-- __init__.py
|   |   |-- metrics.py                     # Evaluation metrics
|   |   |-- visualization.py              # Per-bank comparison plots
|   |-- experiments/
|   |   |-- __init__.py
|   |   |-- local_baseline.py             # Local-only training per bank
|   |   |-- federated_training.py         # FL training loop
|   |   |-- centralized_baseline.py       # Centralized (all data) baseline
|   |-- utils/
|       |-- __init__.py
|       |-- helpers.py
|-- tests/
|   |-- __init__.py
|   |-- test_bank_profiles.py
|   |-- test_data_generation.py
|   |-- test_federation.py                 # Secure aggregation tests
```

### 3. KEY COMPONENTS

| Component | File | Line(s) | Description |
|-----------|------|---------|-------------|
| `main.py` pipeline | `main.py` | 1-452 | 8-step pipeline: load profiles, generate data, partition, local baseline, FL training, centralized baseline, comparison, report. |
| `FraudNN` | `src/models/fraud_nn.py` | 20-155 | Neural network with embedding layers for categorical features (merchant_category, region, card_type, timeOfDay) and dense layers for numerical. Hidden: [128, 64, 32], dropout 0.3. |
| `SimplifiedFraudNN` | `src/models/fraud_nn.py` | 158-220 | Simpler version without embeddings; pure MLP for testing. Xavier init. |
| `PerBankMetricStrategy` | `src/federation/strategy.py` | 25-220 | Extends Flower `FedAvg`. Overrides `aggregate_evaluate()` to track metrics per bank using `defaultdict`. Computes weighted global metrics from per-bank results. |
| `FraudGenerator` | `src/data_generation/fraud_generator.py` | 20-200 | Generates synthetic fraud transactions with 6 types: `card_present`, `card_not_present`, `account_takeover`, `synthetic_identity`, `cross_border`, `internal`. Each type has distinct feature distributions. |
| `FraudGenerator._inject_label_noise()` | `src/data_generation/fraud_generator.py` | 202-240 | Flips labels with configurable noise rate to simulate real-world labeling uncertainty. |
| `SecureAggregator` | `src/federation/secure_aggregation.py` | (imported in test_federation.py) | Additive masking: each client adds random mask, masks cancel during aggregation. |

### 4. DATA FLOW

```
config/bank_profiles.yaml          config/simulation_config.yaml
    |                                    |
    v                                    v
main.py Step 1: Load bank profiles   Step 2: Generate synthetic data
    |                                    |
    +-> FraudGenerator.generate()        |
    |     per bank profile               |
    |     6 fraud types, label noise     |
    |                                    |
    v                                    v
Step 3: Partition data per bank      Step 4: Local baselines
    |                                    |
    v                                    v
Step 5: Federated Training           Step 6: Centralized baseline
    |                                    |
    +-> PerBankMetricStrategy            |
    |     aggregate_fit()                |
    |     aggregate_evaluate()           |
    |       per-bank tracking            |
    |                                    |
    v                                    v
Step 7: Compare (local vs FL vs centralized)
    |
    v
Step 8: Generate report + visualizations
```

### 5. CONFIGURATION & PARAMETERS

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `federation.n_rounds` | `config/simulation_config.yaml` L12 | `15` | Number of FL rounds |
| `federation.n_clients_per_round` | `config/simulation_config.yaml` L13 | `5` | All banks participate each round |
| `federation.local_epochs` | `config/simulation_config.yaml` L14 | `3` | Local training epochs per round |
| `federation.batch_size` | `config/simulation_config.yaml` L15 | `256` | Client batch size |
| `model.hidden_layers` | `config/simulation_config.yaml` L25 | `[128, 64, 32]` | MLP hidden dimensions |
| `model.dropout` | `config/simulation_config.yaml` L26 | `0.3` | Dropout rate |
| `training.learning_rate` | `config/simulation_config.yaml` L31 | `0.001` | Client learning rate |
| `training.loss_function` | `config/simulation_config.yaml` L34 | `BCEWithLogitsLoss` | Loss function |
| `secure_aggregation.enabled` | `config/simulation_config.yaml` L44 | `true` | Enable secure aggregation simulation |
| `experiment.n_days_data` | `config/simulation_config.yaml` L6 | `30` | Days of transaction data to generate |

### 6. EXTERNAL DEPENDENCIES

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | >=1.24.0 | Array operations, random generation |
| `pandas` | >=2.0.0 | Transaction DataFrame handling |
| `torch` | >=2.0.0 | Neural network training |
| `scikit-learn` | >=1.3.0 | AUC-ROC, precision, recall, confusion matrix |
| `flwr` | >=1.8.0 | Flower FedAvg base strategy |
| `ray` | >=2.6.0 | Parallel client simulation |
| `matplotlib` | >=3.7.0 | Per-bank comparison plots |
| `seaborn` | >=0.12.0 | Statistical visualization |
| `pyyaml` | >=6.0 | YAML config parsing |

### 7. KNOWN ISSUES

- `main.py` is a monolithic 452-line script; the pipeline steps are functions within a single file rather than separate modules.
- `FraudNN` embedding dimensions in `fraud_nn.py` are partially hardcoded; must match `simulation_config.yaml` exactly or crash.
- `PerBankMetricStrategy` uses `defaultdict(list)` which grows unboundedly over rounds; no history pruning.
- Secure aggregation is simulated (additive masking) but not cryptographically secure; no real MPC.
- Data generation does not model temporal drift or seasonal patterns; all 30 days are IID.
- Only 5 banks supported by default; adding banks requires editing `bank_profiles.yaml` and regenerating data.

### 8. TESTING

```
tests/
|-- test_bank_profiles.py      # Bank profile loading, validation
|-- test_data_generation.py    # FraudGenerator output shapes, fraud type coverage
|-- test_federation.py         # SecureAggregator: masking, mask cancellation,
|                              #   verify aggregation correctness
```

Run: `cd cross_silo_bank_fl && pytest tests/ -v`

### 9. QUICK MODIFICATION GUIDE

| Change | File | What to modify |
|--------|------|----------------|
| Add new bank type | `config/bank_profiles.yaml` | Add new bank entry with transaction volume, fraud rate, feature distributions |
| Add new fraud type | `src/data_generation/fraud_generator.py` L20-200 | Add to fraud type dict with feature distribution parameters |
| Replace secure aggregation | `src/federation/secure_aggregation.py` | Implement real MPC (e.g., Shamir secret sharing) |
| Add differential privacy | `src/federation/strategy.py` L25-220 | Add noise in `aggregate_fit()` after weighted averaging |
| Track per-round convergence | `main.py` + `src/federation/strategy.py` | Return round-level metrics from `aggregate_evaluate()` |

### 10. CODE SNIPPETS

**FraudNN with Embeddings** (`src/models/fraud_nn.py` lines 20-155):
```python
class FraudNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Embedding layers for categorical features
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(num_cat, dim)
            for name, (num_cat, dim) in config['embedding_dims'].items()
        })
        # Dense layers for numerical features
        total_embed = sum(d for _, d in config['embedding_dims'].values())
        input_size = config['input_dim'] + total_embed
        layers = []
        for hidden in config['hidden_layers']:
            layers.extend([nn.Linear(input_size, hidden), nn.ReLU(), nn.Dropout(config['dropout'])])
            input_size = hidden
        layers.append(nn.Linear(input_size, 1))
        self.classifier = nn.Sequential(*layers)
```

**Per-Bank Metric Strategy** (`src/federation/strategy.py` lines 25-220):
```python
class PerBankMetricStrategy(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.per_bank_metrics = defaultdict(list)

    def aggregate_evaluate(self, server_round, results, failures):
        for client_proxy, eval_res in results:
            bank_id = eval_res.metrics.get("bank_id", "unknown")
            self.per_bank_metrics[bank_id].append({
                "round": server_round,
                "loss": eval_res.loss,
                "auc_roc": eval_res.metrics.get("auc_roc", 0.0),
            })
        return super().aggregate_evaluate(server_round, results, failures)
```

---

## P-13 | Day 13 | vertical_fraud_detection | Score 10/10

### 1. PURPOSE

Vertical federated learning for fraud detection using Split Neural Networks. Models the scenario where two parties (e.g., a bank and a payment processor) each hold different features for the same set of customers. Party A holds transaction features (amount, frequency, etc.), Party B holds account features (balance, history, etc.), and a server top model combines their embeddings for fraud prediction. Implements Private Set Intersection (PSI) using hashing (SHA-256 + salt) for secure entity alignment without revealing raw IDs. Includes gradient leakage analysis under honest-but-curious threat model and comparison with single-party and horizontal FL baselines.

### 2. ARCHITECTURE

```
vertical_fraud_detection/
|-- run_experiments.py                 # CLI entry point with modes (339 lines)
|-- verify_setup.py
|-- requirements.txt
|-- config/
|   |-- experiment_config.yaml         # Data, PSI, training, privacy settings
|   |-- model_config.yaml             # Model architecture parameters
|-- src/
|   |-- __init__.py
|   |-- models/
|   |   |-- __init__.py
|   |   |-- split_nn.py               # SplitNN orchestrator (258 lines)
|   |   |-- bottom_model.py           # PartyABottomModel, PartyBBottomModel
|   |   |-- top_model.py              # TopModel (server-side classifier)
|   |-- training/
|   |   |-- __init__.py
|   |   |-- vertical_fl_trainer.py    # VerticalFLTrainer (484 lines)
|   |   |-- forward_pass.py           # Forward pass logic
|   |   |-- backward_pass.py          # Split backward pass
|   |-- psi/
|   |   |-- __init__.py
|   |   |-- private_set_intersection.py # PSI with SHA-256 (231 lines)
|   |-- privacy/
|   |   |-- __init__.py
|   |   |-- gradient_leakage.py        # Gradient leakage analysis
|   |   |-- threat_model.py            # Honest-but-curious threat model
|   |-- experiments/
|   |   |-- __init__.py
|   |   |-- vertical_fl.py            # VFL experiment runner
|   |   |-- single_party_baseline.py  # Single-party baseline
|   |   |-- horizontal_fl_baseline.py # Horizontal FL baseline
|   |-- utils/
|       |-- __init__.py
|       |-- data_loader.py            # Data loading and feature splitting
|       |-- metrics.py                # AUC-ROC, AUC-PR, confusion matrix
|       |-- visualization.py          # Training curves, comparison plots
|-- tests/
|   |-- test_split_nn.py              # SplitNN initialization, forward/backward
|   |-- test_psi.py                   # PSI correctness
|   |-- test_gradient_flow.py         # Gradient flow through split boundary
```

### 3. KEY COMPONENTS

| Component | File | Line(s) | Description |
|-----------|------|---------|-------------|
| `SplitNN` | `src/models/split_nn.py` | 20-200 | Orchestrates Party A bottom, Party B bottom, and server top model. `forward_pass()` concatenates embeddings from both parties. `backward_pass()` splits gradients back to parties with norm tracking. |
| `SplitNN.forward_pass()` | `src/models/split_nn.py` | 60-105 | `embed_a = bottom_a(x_a)`, `embed_b = bottom_b(x_b)`, `combined = cat(embed_a, embed_b)`, `output = top(combined)`. |
| `SplitNN.backward_pass()` | `src/models/split_nn.py` | 107-165 | Computes loss, backpropagates through top model, splits gradient at concatenation boundary, sends grad_a to Party A, grad_b to Party B. Tracks gradient norms per party. |
| `VerticalFLTrainer` | `src/training/vertical_fl_trainer.py` | 40-400 | Full training loop with separate optimizers per party (Party A, Party B, top model). Supports gradient clipping, early stopping, LR scheduling. |
| `TrainingConfig` | `src/training/vertical_fl_trainer.py` | 18-38 | Dataclass: `num_epochs`, `early_stopping_patience`, `learning_rate`, `batch_size`, `gradient_clip`. |
| `TrainingHistory` | `src/training/vertical_fl_trainer.py` | (near TrainingConfig) | Dataclass recording per-epoch loss, AUC-ROC, AUC-PR, gradient norms. |
| `PrivateSetIntersection` | `src/psi/private_set_intersection.py` | 22-160 | Hashing-based PSI: each party hashes IDs with SHA-256 + shared salt, computes intersection of hash sets. Returns `PSIResult` with matched indices. |
| `PSIResult` | `src/psi/private_set_intersection.py` | 15-20 | Dataclass: `intersection_size`, `party_a_indices`, `party_b_indices`, `jaccard_similarity`. |
| `execute_psi()` | `src/psi/private_set_intersection.py` | 165-231 | Convenience function: creates PSI instance, runs intersection, returns result. |
| `run_experiments.py` | `run_experiments.py` | 1-339 | CLI with modes: `setup` (data gen + PSI + splits), `vfl` (vertical FL training), `baseline` (single-party), `all` (everything). |

### 4. DATA FLOW

```
Party A (bank)          Party B (processor)         Server
features_a              features_b
    |                       |
    v                       v
PSI: hash(id_a + salt)  PSI: hash(id_b + salt)
    |                       |
    +-------> intersect <---+
              aligned indices
    |                       |
    v                       v
bottom_model_a(x_a)    bottom_model_b(x_b)
    |                       |
    embed_a                 embed_b
    |                       |
    +----> concatenate <----+
                |
                v
          top_model(combined)
                |
                v
          loss = BCE(output, y)
                |
                v
          backward through top_model
                |
          grad splits at concat boundary
          /                  \
    grad_a                  grad_b
      |                       |
      v                       v
  bottom_a.backward()    bottom_b.backward()
```

### 5. CONFIGURATION & PARAMETERS

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `data.num_samples` | `config/experiment_config.yaml` L5 | `100000` | Total dataset size |
| `data.fraud_ratio` | `config/experiment_config.yaml` L6 | `0.05` | Fraud label ratio (5%) |
| `psi.method` | `config/experiment_config.yaml` L14 | `"hashing"` | PSI method (hashing or EC) |
| `psi.hash_function` | `config/experiment_config.yaml` L15 | `"sha256"` | Hash function for PSI |
| `psi.salt_length` | `config/experiment_config.yaml` L16 | `32` | Salt length in bytes |
| `training.num_epochs` | `config/experiment_config.yaml` L20 | `50` | Training epochs |
| `training.early_stopping_patience` | `config/experiment_config.yaml` L21 | `10` | Patience for early stopping |
| `training.gradient_clip` | `config/experiment_config.yaml` L24 | `1.0` | Gradient clipping threshold |
| `privacy.analyze_gradient_leakage` | `config/experiment_config.yaml` L39 | `true` | Run gradient leakage analysis |
| `privacy.threat_model` | `config/experiment_config.yaml` L41 | `"honest_but_curious"` | Threat model assumption |

### 6. EXTERNAL DEPENDENCIES

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ==2.1.0 | Neural network training, autograd for split backward |
| `numpy` | ==1.24.3 | Array operations |
| `pandas` | ==2.0.3 | Data handling, feature splitting |
| `scikit-learn` | ==1.3.0 | AUC-ROC, AUC-PR, confusion matrix |
| `matplotlib` | ==3.7.2 | Training curves, comparison plots |
| `seaborn` | ==0.12.2 | Statistical visualization |
| `pyyaml` | ==6.0.1 | YAML config parsing |
| `pytest` | ==7.4.0 | Test runner |

### 7. KNOWN ISSUES

- PSI uses simple SHA-256 hashing which is NOT cryptographically secure against dictionary attacks (salt mitigates but does not eliminate risk). Real PSI needs OT or EC-based protocols.
- `SplitNN.backward_pass()` sends raw gradient tensors to parties; gradient leakage attack possible under honest-but-curious model (analyzed in `src/privacy/gradient_leakage.py`).
- No support for more than 2 parties; the concatenation in `forward_pass()` is hardcoded for Party A + Party B.
- `VerticalFLTrainer` uses 3 separate optimizers which may have inconsistent learning rate schedules.
- `run_experiments.py` expects pinned dependency versions (e.g., `torch==2.1.0`); may break with different PyTorch versions.
- No communication cost modeling; gradient transfer between parties is simulated as in-memory tensor copy.

### 8. TESTING

```
tests/
|-- test_split_nn.py          # SplitNN init, forward output shapes,
|                             #   backward gradient flow, embedding dimensions
|-- test_psi.py               # PSI: full overlap, partial overlap, no overlap,
|                             #   Jaccard similarity correctness
|-- test_gradient_flow.py     # Gradients reach both bottom models,
|                             #   gradient norms are tracked correctly
```

Run: `cd vertical_fraud_detection && pytest tests/ -v` or `python run_experiments.py --mode test`

### 9. QUICK MODIFICATION GUIDE

| Change | File | What to modify |
|--------|------|----------------|
| Add third party | `src/models/split_nn.py` L60-165 | Add `bottom_model_c`, extend `forward_pass()` concatenation and `backward_pass()` gradient split |
| Use EC-based PSI | `src/psi/private_set_intersection.py` | Implement elliptic curve Diffie-Hellman based PSI alongside hashing |
| Add gradient perturbation defense | `src/models/split_nn.py` L107-165 | Add Gaussian noise to gradients in `backward_pass()` before sending to parties |
| Support variable feature splits | `run_experiments.py` | Parameterize party A/B feature column indices instead of hardcoding |
| Add communication cost tracking | `src/training/vertical_fl_trainer.py` | Add byte counting for gradient tensors transferred between parties each epoch |

### 10. CODE SNIPPETS

**Split Neural Network Forward/Backward** (`src/models/split_nn.py` lines 60-165):
```python
class SplitNN:
    def forward_pass(self, x_a, x_b):
        """Forward pass through split network."""
        self.embed_a = self.bottom_model_a(x_a)
        self.embed_b = self.bottom_model_b(x_b)
        self.embed_a.retain_grad()
        self.embed_b.retain_grad()
        combined = torch.cat([self.embed_a, self.embed_b], dim=1)
        output = self.top_model(combined)
        return output

    def backward_pass(self, loss):
        """Backward pass splitting gradients at concatenation boundary."""
        loss.backward()
        grad_a = self.embed_a.grad.clone()
        grad_b = self.embed_b.grad.clone()
        self.gradient_norms['party_a'].append(grad_a.norm().item())
        self.gradient_norms['party_b'].append(grad_b.norm().item())
        return grad_a, grad_b
```

**Private Set Intersection** (`src/psi/private_set_intersection.py` lines 22-160):
```python
class PrivateSetIntersection:
    def __init__(self, hash_function='sha256', salt=None, salt_length=32):
        self.hash_fn = hashlib.new(hash_function)
        self.salt = salt or os.urandom(salt_length)

    def _hash_id(self, id_value):
        h = hashlib.sha256()
        h.update(self.salt + str(id_value).encode())
        return h.hexdigest()

    def intersect(self, ids_a, ids_b):
        hashes_a = {self._hash_id(id_): i for i, id_ in enumerate(ids_a)}
        hashes_b = {self._hash_id(id_): i for i, id_ in enumerate(ids_b)}
        common = set(hashes_a.keys()) & set(hashes_b.keys())
        return PSIResult(
            intersection_size=len(common),
            party_a_indices=[hashes_a[h] for h in common],
            party_b_indices=[hashes_b[h] for h in common],
            jaccard_similarity=len(common) / len(set(hashes_a) | set(hashes_b))
        )
```

---

## P-20 | Day 20 | personalized_fl_fraud | Score 9/10

### 1. PURPOSE

Comprehensive implementation and comparison of personalized federated learning methods for fraud detection. Addresses the fundamental tension in FL between global model generalization and client-specific adaptation. Implements four personalization approaches: FedAvg baseline (no personalization), Local Fine-Tuning (post-hoc adaptation), Per-FedAvg (MAML-style meta-learning with Moreau envelope regularization), FedPer (shared feature extractor with personalized classifier head), and Ditto (dual global+local models with proximal regularization). Each method extends a common `PersonalizationMethod` abstract base class for fair comparison. Includes compute budget tracking to ensure equivalent computational effort across methods.

### 2. ARCHITECTURE

```
personalized_fl_fraud/
|-- requirements.txt
|-- config/
|   |-- experiments.yaml               # FL experiment parameters
|   |-- methods.yaml                   # Per-method hyperparameters (96 lines)
|-- examples/
|   |-- demo_comparison.py             # Compare all methods
|   |-- demo_single_method.py          # Run one method
|-- src/
|   |-- __init__.py
|   |-- methods/
|   |   |-- __init__.py
|   |   |-- base.py                    # PersonalizationMethod ABC (262 lines)
|   |   |-- ditto.py                   # Ditto: dual model + proximal (337 lines)
|   |   |-- per_fedavg.py             # Per-FedAvg: MAML + Moreau (403 lines)
|   |   |-- fedper.py                 # FedPer: shared/personal layers
|   |   |-- local_finetuning.py       # Local fine-tuning post-hoc
|   |-- clients/
|   |   |-- __init__.py
|   |   |-- personalized_client.py     # Personalized Flower client
|   |   |-- wrappers.py               # Client wrappers per method
|   |-- servers/
|   |   |-- __init__.py
|   |   |-- personalized_server.py     # Server with method dispatch
|   |-- models/
|   |   |-- __init__.py
|   |   |-- base.py                    # Base model definition
|   |   |-- utils.py                   # Model utilities
|   |-- metrics/
|   |   |-- __init__.py
|   |   |-- personalized_metrics.py    # Personalization benefit metrics
|   |   |-- visualization.py          # Violin plots, radar charts
|   |-- utils/
|       |-- __init__.py
|       |-- compute_tracking.py        # FLOPs and time tracking
|       |-- metrics.py                 # General metrics utilities
|       |-- partitioning.py            # Data partitioning utilities
|       |-- reproducibility.py         # Seed management
|-- tests/
|   |-- __init__.py
|   |-- test_methods.py                # Unit tests for personalization methods
```

### 3. KEY COMPONENTS

| Component | File | Line(s) | Description |
|-----------|------|---------|-------------|
| `PersonalizationMethod` | `src/methods/base.py` | 30-160 | Abstract base class. Defines interface: `get_client_strategy()`, `get_server_strategy()`, `compute_personalization_benefit()`. |
| `PersonalizationResult` | `src/methods/base.py` | 15-28 | Dataclass: `global_metrics`, `personalized_metrics`, `personalization_delta`, `method_name`, `compute_cost`. |
| `FedAvgBaseline` | `src/methods/base.py` | 163-262 | Concrete baseline: standard FedAvg with no personalization. Returns global model metrics as both global and personalized. |
| `Ditto` | `src/methods/ditto.py` | 25-337 | Dual-model approach: trains global model via FedAvg + local model with proximal regularization `lambda * ||w_local - w_global||^2`. |
| `Ditto.compute_ditto_loss()` | `src/methods/ditto.py` | 120-165 | `loss = task_loss + lambda/2 * sum((p_local - p_global)^2)`. |
| `Ditto.create_personalized_model()` | `src/methods/ditto.py` | 250-300 | Alpha blending: `w_personalized = alpha * w_global + (1 - alpha) * w_local`. |
| `PerFedAvg` | `src/methods/per_fedavg.py` | 25-403 | MAML-inspired meta-learning. `inner_loop_adaptation()` takes gradient steps on support set. `outer_loop_meta_update()` optimizes for fast adaptation on query set. Moreau envelope: `f(w) + beta/2 * ||w - w_0||^2`. |
| `PerFedAvg.inner_loop_adaptation()` | `src/methods/per_fedavg.py` | 100-175 | Takes `num_inner_steps` gradient steps at `lr_inner` on support data. |
| `PerFedAvg.outer_loop_meta_update()` | `src/methods/per_fedavg.py` | 177-250 | Computes meta-gradient: gradient of query loss w.r.t. pre-adaptation parameters. |
| `PerFedAvg.adapt_to_client()` | `src/methods/per_fedavg.py` | 300-360 | Full personalization: inner loop adaptation on client data, returns adapted model. |

### 4. DATA FLOW

```
Global Model (server)
    |
    +--> distribute to all clients
    |
    +----------+----------+----------+
    |          |          |          |
 Client 0   Client 1   ...     Client K
    |          |                    |
    v          v                    v
[Method-specific local training]
    |
    +-- FedAvg: standard SGD, upload weights
    +-- Ditto: SGD on global model + proximal on local model
    +-- PerFedAvg: inner loop (support) + outer loop (query)
    +-- FedPer: train shared layers only, keep personal head
    +-- LocalFT: standard SGD, then fine-tune post-hoc
    |
    v
Server aggregates global updates (FedAvg)
    |
    v
After T rounds:
    +-- FedAvg: evaluate global model per client
    +-- Ditto: evaluate blended alpha*global + (1-alpha)*local
    +-- PerFedAvg: evaluate adapted model (inner loop on client)
    +-- FedPer: evaluate with personal classifier head
    +-- LocalFT: evaluate fine-tuned model
    |
    v
PersonalizationResult per method
    -> compute_personalization_benefit()
    -> personalization_delta = personalized - global
```

### 5. CONFIGURATION & PARAMETERS

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `methods.ditto.lambda_regularization` | `config/methods.yaml` L48 | `0.5` | Proximal regularization strength |
| `methods.ditto.personal_epochs` | `config/methods.yaml` L49 | `5` | Local model training epochs |
| `methods.ditto.local_lr` | `config/methods.yaml` L50 | `0.01` | Local model learning rate |
| `methods.per_fedavg.beta` | `config/methods.yaml` L26 | `1.0` | Moreau envelope regularization strength |
| `methods.per_fedavg.lr_inner` | `config/methods.yaml` L27 | `0.01` | Inner loop (adaptation) learning rate |
| `methods.per_fedavg.num_inner_steps` | `config/methods.yaml` L28 | `5` | Inner loop gradient steps |
| `methods.per_fedavg.lr_meta` | `config/methods.yaml` L29 | `0.001` | Outer loop (meta) learning rate |
| `methods.fedper.personal_layers` | `config/methods.yaml` L38 | `["classifier"]` | Layer names to personalize |
| `methods.local_finetuning.finetuning_epochs` | `config/methods.yaml` L16 | `10` | Post-hoc fine-tuning epochs |
| `compute_allocation.communication_rounds` | `config/methods.yaml` L68 | `100` | FL rounds per method |

### 6. EXTERNAL DEPENDENCIES

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.0.0 | Neural network training, autograd for meta-learning |
| `numpy` | >=1.24.0 | Array operations |
| `pandas` | >=2.0.0 | Data handling |
| `scikit-learn` | >=1.3.0 | Evaluation metrics |
| `flwr` | >=1.8.0 | Flower framework for FL simulation |
| `omegaconf` | >=2.3.0 | Hierarchical configuration |
| `matplotlib` | >=3.7.0 | Violin plots, comparison charts |
| `seaborn` | >=0.12.0 | Statistical visualization |
| `psutil` | >=5.9.0 | Compute resource tracking |
| `tensorboard` | >=2.13.0 | Training visualization (optional) |
| `wandb` | >=0.15.0 | Experiment tracking (optional) |

### 7. KNOWN ISSUES

- `PerFedAvg` second-order meta-gradient computation (`outer_loop_meta_update`) uses `torch.autograd.grad()` with `create_graph=True`, which is memory-intensive for large models.
- `PerFedAvg.first_order_approx` option (config/methods.yaml L30, default `false`) is available but may reduce personalization quality.
- `Ditto.create_personalized_model()` alpha blending is a simple linear interpolation; no learned alpha or attention-based mixing.
- `FedPer` personal layer identification by name string (`"classifier"`) is fragile; depends on exact model attribute naming.
- Compute budget tracking (`config/methods.yaml` L54-71) counts epochs but not actual FLOPs; inner/outer loop overhead in Per-FedAvg is not reflected in epoch counts.
- No support for client drift detection or dynamic method switching during training.

### 8. TESTING

```
tests/
|-- test_methods.py   # TestLocalFineTuning: data fixtures, fine-tuning changes params
|                     # TestDitto: proximal loss computation, dual model update
|                     # TestPerFedAvg: inner loop adaptation, meta-gradient correctness
|                     # TestFedPer: layer freezing, personal head update
```

Run: `cd personalized_fl_fraud && pytest tests/ -v`

### 9. QUICK MODIFICATION GUIDE

| Change | File | What to modify |
|--------|------|----------------|
| Add APFL (Adaptive PFL) | `src/methods/` | New file `apfl.py` extending `PersonalizationMethod`; learned mixing coefficient alpha per client |
| Add Moreau envelope scheduling | `src/methods/per_fedavg.py` L25-403 | Decay `beta` over rounds in `outer_loop_meta_update()` |
| Make Ditto alpha learnable | `src/methods/ditto.py` L250-300 | Replace fixed alpha with `nn.Parameter`, optimize via validation loss |
| Add per-client metric reporting | `src/metrics/personalized_metrics.py` | Extend to compute client-level personalization gap distribution |
| Add FedRep method | `src/methods/` | New file: shared representation + per-client linear head, alternating optimization |

### 10. CODE SNIPPETS

**Ditto Proximal Loss** (`src/methods/ditto.py` lines 120-165):
```python
def compute_ditto_loss(self, model, global_model, data, targets, loss_fn):
    """Compute Ditto loss: task loss + proximal regularization."""
    output = model(data)
    task_loss = loss_fn(output, targets)

    # Proximal term: lambda/2 * ||w_local - w_global||^2
    proximal_term = 0.0
    for p_local, p_global in zip(model.parameters(), global_model.parameters()):
        proximal_term += (p_local - p_global.detach()).norm(2) ** 2

    total_loss = task_loss + (self.lambda_reg / 2) * proximal_term
    return total_loss
```

**Per-FedAvg Inner Loop Adaptation** (`src/methods/per_fedavg.py` lines 100-175):
```python
def inner_loop_adaptation(self, model, support_data, support_targets, loss_fn):
    """MAML inner loop: adapt model to client's support set."""
    adapted_model = copy.deepcopy(model)
    inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.lr_inner)

    for step in range(self.num_inner_steps):
        inner_optimizer.zero_grad()
        output = adapted_model(support_data)
        loss = loss_fn(output, support_targets)

        # Moreau envelope regularization
        if self.beta > 0:
            moreau_term = 0.0
            for p_adapt, p_init in zip(adapted_model.parameters(), model.parameters()):
                moreau_term += (p_adapt - p_init).norm(2) ** 2
            loss += (self.beta / 2) * moreau_term

        loss.backward()
        inner_optimizer.step()

    return adapted_model
```

---

## P-22 | Day 22 | dp_federated_learning | Score 9/10

### 1. PURPOSE

From-scratch implementation of Differentially Private Stochastic Gradient Descent (DP-SGD) as described in Abadi et al. "Deep Learning with Differential Privacy" (CCS 2016). Implements the complete DP-SGD pipeline: per-sample gradient computation, L2 gradient clipping to bound sensitivity, calibrated Gaussian noise addition, and Renyi Differential Privacy (RDP) accounting for tracking cumulative privacy loss. Provides utility functions for computing the noise multiplier needed to achieve a target (epsilon, delta)-DP guarantee, and for determining the maximum training steps within a privacy budget. Designed for integration with federated learning (central DP strategy).

### 2. ARCHITECTURE

```
dp_federated_learning/
|-- requirements.txt
|-- config/
|   |-- privacy.yaml                  # DP parameters, accounting orders (48 lines)
|   |-- experiment.yaml               # Experiment settings
|-- src/
|   |-- __init__.py
|   |-- dp_mechanisms/
|   |   |-- __init__.py
|   |   |-- gradient_clipper.py       # Per-sample grads, L2 clipping (249 lines)
|   |   |-- noise_addition.py         # Gaussian noise, multiplier calibration (317 lines)
|   |   |-- privacy_accountant.py     # RDPAccountant, privacy tracking (339 lines)
|   |-- dp_strategies/
|   |   |-- __init__.py
|   |-- models/
|   |   |-- __init__.py
|   |   |-- dp_sgd_custom.py          # DPSGDOptimizer (341 lines)
|   |   |-- opacus_wrapper.py         # Opacus comparison wrapper
|   |-- fl_system/
|   |   |-- __init__.py
|   |-- experiments/
|   |   |-- __init__.py
|   |-- utils/
|       |-- __init__.py
|       |-- privacy_calibration.py    # Privacy calibration utilities
|-- tests/
|   |-- test_dp_sgd.py                # 14 tests across 6 test classes (407 lines)
|   |-- test_gradient_clipping.py     # Gradient clipping correctness
|   |-- test_noise_calibration.py     # Noise multiplier computation
|   |-- test_privacy_accountant.py    # RDP accounting correctness
```

### 3. KEY COMPONENTS

| Component | File | Line(s) | Description |
|-----------|------|---------|-------------|
| `DPSGDOptimizer` | `src/models/dp_sgd_custom.py` | 40-238 | Full DP-SGD: per-sample gradients, clip, aggregate, add noise, momentum, weight decay, privacy accounting. |
| `DPSGDOptimizer.step()` | `src/models/dp_sgd_custom.py` | 133-238 | Algorithm 1 from Abadi et al.: (1) per-sample grads, (2) L2 clip to C, (3) sum, (4) add N(0, sigma^2 * C^2), (5) update parameters. Returns `DPSGDMetrics`. |
| `DPSGDMetrics` | `src/models/dp_sgd_custom.py` | 30-37 | Dataclass: `clip_norm_mean`, `clip_norm_max`, `clip_fraction`, `noise_std`, `epsilon_spent`. |
| `create_dp_sgd_optimizer()` | `src/models/dp_sgd_custom.py` | 253-290 | Convenience factory function. |
| `compute_per_sample_gradients()` | `src/dp_mechanisms/gradient_clipper.py` | 21-108 | Loops over each sample in batch, computes individual gradient via `loss.backward()`, stores flattened gradient vectors. Shape: (batch_size, num_params). |
| `clip_gradients_l2()` | `src/dp_mechanisms/gradient_clipper.py` | 111-167 | `g_clipped = g / max(1, ||g||_2 / C)`. Returns clipped gradients + original norms. |
| `flat_clip_gradients_l2()` | `src/dp_mechanisms/gradient_clipper.py` | 170-214 | Approximate (NOT per-sample) clipping. Educational comparison only. |
| `add_gaussian_noise()` | `src/dp_mechanisms/noise_addition.py` | 22-88 | Adds N(0, sigma^2 * C^2 * n) where sigma = noise_multiplier, C = clipping bound, n = num_samples. |
| `compute_noise_multiplier()` | `src/dp_mechanisms/noise_addition.py` | 114-220 | Binary search for minimum sigma achieving target epsilon, given delta/steps/sampling_rate. 40 iterations for 1e-12 precision. |
| `RDPAccountant` | `src/dp_mechanisms/privacy_accountant.py` | 50-207 | Renyi DP accountant. Tracks cumulative RDP over 16 orders. `step()` accumulates `q^2 * alpha / (2 * sigma^2)` per step. |
| `RDPAccountant.get_epsilon()` | `src/dp_mechanisms/privacy_accountant.py` | 143-162 | RDP-to-(epsilon,delta) conversion: `epsilon = min_alpha[ total_rdp_alpha + log(1/delta) / (alpha - 1) ]`. |
| `PrivacyBudget` | `src/dp_mechanisms/privacy_accountant.py` | 25-47 | Dataclass tracking spent vs remaining epsilon/delta. |
| `compute_steps_for_epsilon()` | `src/dp_mechanisms/privacy_accountant.py` | 247-294 | Binary search for max training steps within target epsilon budget. |
| `compute_noise_multiplier_standard()` | `src/dp_mechanisms/noise_addition.py` | 223-254 | Basic Gaussian mechanism: `sigma = sqrt(2 * ln(1.25/delta)) / epsilon`. No composition. |

### 4. DATA FLOW

```
Input: batch (x_i, y_i), model f_theta
    |
    v
Step 1: compute_per_sample_gradients(model, x, y, loss_fn)
    for i in range(batch_size):
        g_i = backward(loss_fn(model(x_i), y_i))
    -> per_sample_grads: (batch_size, num_params)
    |
    v
Step 2: clip_gradients_l2(per_sample_grads, C)
    g_clipped_i = g_i / max(1, ||g_i||_2 / C)
    -> ensures ||g_clipped_i||_2 <= C
    |
    v
Step 3: aggregated = sum(g_clipped_i)
    |
    v
Step 4: add_gaussian_noise(aggregated, sigma, C, batch_size)
    noised = aggregated + N(0, sigma^2 * C^2 * batch_size)
    |
    v
Step 5: parameter update (with optional momentum, weight decay)
    theta = theta - lr * noised_gradient
    |
    v
Step 6: RDPAccountant.step()
    total_rdp += q^2 * alpha / (2 * sigma^2)
    epsilon = min_alpha[ total_rdp + log(1/delta)/(alpha-1) ]
    |
    v
Return: DPSGDMetrics(clip_norm_mean, clip_fraction, noise_std, epsilon_spent)
```

### 5. CONFIGURATION & PARAMETERS

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `privacy.delta` | `config/privacy.yaml` L7 | `1.0e-5` | Privacy failure probability delta |
| `privacy.epsilon_grid` | `config/privacy.yaml` L10 | `[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]` | Target epsilon values for analysis |
| `clipping.bound` | `config/privacy.yaml` L15 | `1.0` | L2 norm clipping threshold C |
| `clipping.method` | `config/privacy.yaml` L18 | `"flat"` | Clipping method: flat or per_layer |
| `noise.sampling_rate` | `config/privacy.yaml` L22 | `0.01` | Batch sampling probability q |
| `noise.steps` | `config/privacy.yaml` L25 | `1000` | Number of training steps |
| `noise.multiplier` | `config/privacy.yaml` L30 | `1.0` | Noise multiplier sigma |
| `accounting.method` | `config/privacy.yaml` L37 | `"rdp"` | Accounting method (RDP or moments) |
| `accounting.orders` | `config/privacy.yaml` L42 | 16 values: 1.5-128.0 | RDP alpha orders for optimization |
| `strategy` | `config/privacy.yaml` L33 | `"central"` | DP strategy: local, central, or shuffle |

### 6. EXTERNAL DEPENDENCIES

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.0.0 | Per-sample gradient computation, model training |
| `numpy` | >=1.24.0 | RDP computation, binary search, array operations |
| `pandas` | >=2.0.0 | Data handling (optional) |
| `scikit-learn` | >=1.3.0 | Evaluation metrics (optional) |
| `flwr` | >=1.8.0 | Flower integration for federated DP (optional) |
| `omegaconf` | >=2.3.0 | Configuration management (optional) |
| `matplotlib` | >=3.7.0 | Privacy-utility tradeoff plots (optional) |
| `pytest` | >=7.4.0 | Test runner |

### 7. KNOWN ISSUES

- `compute_per_sample_gradients()` (gradient_clipper.py L74) loops over each sample individually (O(B) forward+backward passes per batch). This is the naive approach; Opacus uses functorch/vmap for efficiency.
- The RDP formula `epsilon_alpha = q^2 * alpha / (2 * sigma^2)` in `privacy_accountant.py` L122 is the simplified approximation from Abadi et al. Mironov's tighter bounds for subsampled Gaussian mechanism are not implemented.
- `add_gaussian_noise()` scales by `sqrt(num_samples)` (noise_addition.py L82) which is non-standard; typical DP-SGD divides aggregated gradient by batch_size before adding noise scaled by `sigma * C`.
- `flat_clip_gradients_l2()` is explicitly documented as NOT providing true per-sample DP guarantees; it exists for educational comparison only.
- Weight decay in `DPSGDOptimizer.step()` (dp_sgd_custom.py L197-199) has a placeholder loop that does nothing (`noised_grads = noised_grads`).
- Momentum velocities are not clipped or noised, which could theoretically leak private information across steps.
- No support for per-layer clipping; only flat L2 clipping is implemented.

### 8. TESTING

```
tests/
|-- test_dp_sgd.py               # 14 tests in 6 classes:
|                                #   TestDPSGDOptimizerInit (3 tests): init, invalid params, missing sampling
|                                #   TestDPSGDStep (4 tests): returns metrics, updates params, increments counter, clip stats
|                                #   TestPrivacyAccounting (3 tests): epsilon increases, get_privacy_spent, monotonic consumption
|                                #   TestConvenienceFunction (2 tests): creates optimizer, optimizer works
|                                #   TestMomentum (2 tests): initialized zero, updates after steps
|                                #   TestDifferentNoiseMultipliers (2 tests): small noise, high noise reduces epsilon
|-- test_gradient_clipping.py    # Per-sample gradient shapes, clipped norms <= C, flat vs per-sample comparison
|-- test_noise_calibration.py    # Noise multiplier binary search convergence, standard formula correctness
|-- test_privacy_accountant.py   # RDP accumulation, epsilon conversion, steps_for_epsilon correctness
```

Run: `cd dp_federated_learning && pytest tests/ -v`

### 9. QUICK MODIFICATION GUIDE

| Change | File | What to modify |
|--------|------|----------------|
| Use Opacus for efficient per-sample grads | `src/models/dp_sgd_custom.py` L160-167 | Replace `compute_per_sample_gradients()` with Opacus `GradSampleModule` |
| Implement tighter RDP bounds | `src/dp_mechanisms/privacy_accountant.py` L104-124 | Replace simplified formula with Mironov's subsampled Gaussian mechanism RDP |
| Add per-layer clipping | `src/dp_mechanisms/gradient_clipper.py` | Add function that clips each layer's gradient independently, then concatenates |
| Fix noise scaling | `src/dp_mechanisms/noise_addition.py` L77-82 | Divide aggregated gradient by batch_size, add noise scaled by `sigma * C` (not `sigma * C * sqrt(n)`) |
| Add zCDP accounting | `src/dp_mechanisms/privacy_accountant.py` | Add `zCDPAccountant` class alongside `RDPAccountant` for tighter composition |
| Fix weight decay placeholder | `src/models/dp_sgd_custom.py` L197-199 | Remove placeholder loop; weight decay is already applied at L209-210 |

### 10. CODE SNIPPETS

**DP-SGD Optimizer Step** (`src/models/dp_sgd_custom.py` lines 133-238):
```python
def step(self, inputs, targets, loss_fn):
    """Execute one DP-SGD step (Algorithm 1, Abadi et al. 2016)."""
    batch_size = inputs.shape[0]

    # Step 1: Per-sample gradients
    per_sample_grads = compute_per_sample_gradients(self.model, inputs, targets, loss_fn)

    # Step 2: Clip to L2 norm bound C
    clipped_grads, clip_norms = clip_gradients_l2(per_sample_grads, self.clipping_bound)

    # Step 3: Aggregate
    aggregated_grads = clipped_grads.sum(dim=0)

    # Step 4: Add Gaussian noise N(0, sigma^2 * C^2)
    noised_grads = add_gaussian_noise(aggregated_grads, self.noise_multiplier,
                                       self.clipping_bound, batch_size)

    # Step 5-6: Apply to parameters (with momentum, weight decay)
    idx = 0
    for i, p in enumerate(self.params):
        param_size = p.numel()
        p_grad = noised_grads[idx:idx + param_size].view_as(p)
        if self.weight_decay > 0:
            p_grad = p_grad + self.weight_decay * p.data
        if self.momentum > 0:
            self.velocity[i] = self.momentum * self.velocity[i] + p_grad
            update = self.velocity[i]
        else:
            update = p_grad
        p.data = p.data - self.lr * update
        idx += param_size

    # Update privacy accounting
    self.accountant.step(num_steps=1)
    return DPSGDMetrics(...)
```

**RDP-to-(epsilon,delta) Conversion** (`src/dp_mechanisms/privacy_accountant.py` lines 143-162):
```python
def get_epsilon(self, delta=None):
    """Convert accumulated RDP to (epsilon, delta)-DP."""
    if delta is None:
        delta = self.delta
    # epsilon = min_alpha [ total_rdp_alpha + log(1/delta) / (alpha - 1) ]
    epsilons = self.total_rdp + np.log(1 / delta) / (self.orders - 1)
    return epsilons.min()
```

**Per-Sample Gradient Clipping** (`src/dp_mechanisms/gradient_clipper.py` lines 111-167):
```python
def clip_gradients_l2(per_sample_grads, clipping_bound):
    """Clip per-sample gradients: g_clipped = g / max(1, ||g||/C)."""
    clip_norms = per_sample_grads.norm(dim=1, p=2)
    scale_factors = torch.clamp(clip_norms / clipping_bound, min=1.0).unsqueeze(1)
    clipped_grads = per_sample_grads / scale_factors
    return clipped_grads, clip_norms
```
# CAT-03 (Adversarial Attacks) -- PROJECT CARDS

---

### PROJECT CARD: P-14 -- Label Flipping Attack

**Path:** `03_adversarial_attacks/label_flipping_attack/`
**Language:** Python
**Category:** CAT-03 (Adversarial Attacks)
**Status:** Complete
**Quality Score:** 9/10

#### 1. PURPOSE
Implements and evaluates label flipping attacks on a Federated Learning system for credit card fraud detection. The project supports three attack variants -- random flip, targeted flip (fraud-to-legitimate), and inverse flip -- applied by malicious Flower clients. It measures attack impact through accuracy degradation, convergence delay, per-class accuracy, and robustness metrics across varying attacker fractions.

#### 2. ARCHITECTURE
```
label_flipping_attack/
  config/
    attack_config.py          # AttackConfig dataclass + preset configs
  data/
    processed/                # Generated/processed data storage
  src/
    __init__.py
    attacks/
      __init__.py
      label_flip.py           # LabelFlipAttack class + attack functions
    clients/
      __init__.py
      honest_client.py        # HonestClient (fl.client.NumPyClient)
      malicious_client.py     # MaliciousClient with label poisoning
    experiments/
      __init__.py
      run_attacks.py           # Experiment orchestrator + CLI entry point
    metrics/
      __init__.py
      attack_metrics.py        # Success rate, convergence, stability metrics
      visualization.py         # Matplotlib plotting functions
    models/
      __init__.py
      fraud_mlp.py             # FraudMLP neural network + helpers
    servers/
      __init__.py
      attack_server.py         # AttackMetricsStrategy (custom FedAvg)
    utils/
      __init__.py
      data_loader.py           # FraudDataLoader (synthetic data)
      poisoning_utils.py       # flip_labels, select_malicious_clients
  tests/
    test_label_flip.py         # 20 tests for attack logic
    test_metrics.py            # 16 tests for metrics calculations
  results/
    figures/                   # Output plots directory
  requirements.txt
  README.md
```

#### 3. KEY COMPONENTS
| Component | File | Lines | Responsibility |
|-----------|------|-------|----------------|
| `LabelFlipAttack` | `src/attacks/label_flip.py` | 221 | Core attack class: random_flip, targeted_flip, inverse_flip, apply_attack |
| `AttackConfig` | `config/attack_config.py` | 68 | Dataclass for attack parameters + 8 preset experiment configs |
| `MaliciousClient` | `src/clients/malicious_client.py` | 373 | Flower NumPyClient that poisons labels during fit(); supports delayed attacks |
| `HonestClient` | `src/clients/honest_client.py` | 258 | Standard Flower NumPyClient baseline without attack |
| `AttackMetricsStrategy` | `src/servers/attack_server.py` | 324 | Custom FedAvg strategy tracking per-round attack metrics and per-class accuracy |
| `FraudMLP` | `src/models/fraud_mlp.py` | 212 | 2-hidden-layer MLP (64->32->2) with dropout for binary fraud classification |
| `FraudDataLoader` | `src/utils/data_loader.py` | 274 | Synthetic data generation, preprocessing, IID/non-IID partitioning |
| `run_attacks.py` | `src/experiments/run_attacks.py` | 479 | Experiment orchestrator: baseline, fraction sweep, attack comparison |
| `attack_metrics.py` | `src/metrics/attack_metrics.py` | 318 | Success rate, convergence delay, stability, robustness metrics |
| `visualization.py` | `src/metrics/visualization.py` | 520 | 6 plot functions: accuracy curves, per-class, fraction impact, comparison |
| `poisoning_utils.py` | `src/utils/poisoning_utils.py` | 118 | Low-level label flip, invert, select_malicious_clients helpers |

#### 4. DATA FLOW
```
FraudDataLoader.load_and_prepare()
    |
    v
Synthetic Data (100K samples, 30 features, 0.2% fraud)
    |
    v
StandardScaler --> train/val/test split --> IID partition to N clients
    |
    v
[HonestClient x (1-f)] + [MaliciousClient x f]
    |                           |
    | clean labels              | LabelFlipAttack.poison_labels()
    |                           |   (random/targeted/inverse flip)
    v                           v
Local Training (Adam, CrossEntropyLoss, 5 epochs)
    |
    v
fl.simulation.start_simulation()
    |
    v
AttackMetricsStrategy.aggregate_fit() --> FedAvg aggregation
    |
    v
Per-round metrics: global_accuracy, accuracy_fraud, accuracy_legitimate
    |
    v
compare_histories() / calculate_robustness_metrics()
    |
    v
Visualization plots (PNG) + summary report (text)
```

#### 5. CONFIGURATION & PARAMETERS
| Parameter | Default | Location | Purpose |
|-----------|---------|----------|---------|
| `attack_type` | `"random"` | `AttackConfig` | Type of label flip: random, targeted, inverse |
| `flip_rate` | `0.5` | `AttackConfig` | Probability of flipping each label |
| `malicious_fraction` | `0.2` | `AttackConfig` | Fraction of clients that are malicious |
| `attack_start_round` | `1` | `AttackConfig` | Round when attack begins (delayed attack support) |
| `random_seed` | `42` | `AttackConfig` | Random seed for reproducibility |
| `num_rounds` | `100` | `run_attacks.py` CLI | Total federated learning rounds |
| `num_clients` | `10` | `run_attacks.py` CLI | Total number of FL clients |
| `local_epochs` | `5` | `run_attacks.py` CLI | Local training epochs per client per round |
| `learning_rate` | `0.01` | `HonestClient` | Adam optimizer learning rate |
| `input_size` | `30` | `FraudMLP` | Number of input features |
| `hidden_sizes` | `(64, 32)` | `FraudMLP` | Hidden layer dimensions |
| `dropout_rate` | `0.2` | `FraudMLP` | Dropout probability |

#### 6. EXTERNAL DEPENDENCIES
| Library | Version | Used For |
|---------|---------|----------|
| `torch` | >=2.0.0 | Neural network model, training, tensors |
| `torchvision` | >=0.15.0 | Listed but not directly imported |
| `numpy` | >=1.24.0 | Label manipulation, array operations |
| `scikit-learn` | >=1.3.0 | StandardScaler, train_test_split |
| `flwr` | >=1.8.0 | Federated learning framework (Flower) |
| `pandas` | >=2.0.0 | Data processing |
| `matplotlib` | >=3.7.0 | Visualization plots |
| `seaborn` | >=0.12.0 | Plot styling |
| `pytest` | >=7.4.0 | Testing framework |

#### 7. KNOWN ISSUES & IMPROVEMENT OPPORTUNITIES
- [ ] Data loader generates synthetic data only; TODO comment at line 60 of data_loader.py to replace with real credit card fraud dataset
- [ ] `_evaluate()` in honest_client.py returns 2 values in signature but 3 in usage at line 140 (missing third return value pattern)
- [ ] `fraud_mlp.py` references `np` (line 108) but does not import numpy at module level
- [ ] No requirements.txt pinning for reproducible builds (uses >= for all versions)
- [ ] `visualization.py` line 364 uses relative import `from src.metrics.attack_metrics` which may fail depending on working directory
- [ ] Flower simulation API usage may need updates for newer Flower versions (API evolving rapidly)
- [ ] Non-IID partition (`create_federated_partitions`) uses simple sorted partitioning; could use Dirichlet allocation

#### 8. TESTING
- **Test file:** `tests/test_label_flip.py` (222 lines, 20 tests), `tests/test_metrics.py` (264 lines, 16 tests)
- **Coverage:** Not measured
- **How to run:** `cd label_flipping_attack && pytest tests/ -v`

#### 9. QUICK MODIFICATION GUIDE
| Want to... | Modify | Notes |
|------------|--------|-------|
| Add new attack type | `src/attacks/label_flip.py` | Add function + case in `apply_attack()` and update `AttackConfig` Literal type |
| Change model architecture | `src/models/fraud_mlp.py` | Modify `hidden_sizes` tuple in `FraudMLP.__init__()` |
| Use real dataset | `src/utils/data_loader.py` | Replace `load_data()` method (line 49-82) with actual CSV/API loading |
| Add new experiment scenario | `config/attack_config.py` | Add entry to `get_attack_configs()` dict |
| Change aggregation strategy | `src/servers/attack_server.py` | Modify `AttackMetricsStrategy` or subclass different Flower strategy |
| Adjust attacker fractions | `src/experiments/run_attacks.py` | Edit `fractions` list in `run_attacker_fraction_sweep()` (default: [0.1, 0.2, 0.3, 0.5]) |

#### 10. CODE SNIPPETS (Critical Logic Only)
```python
# Label flipping dispatch logic
# File: src/attacks/label_flip.py, Lines: 97-143
def apply_attack(
    labels: np.ndarray,
    attack_type: Literal["random", "targeted", "inverse"],
    flip_prob: float = 0.5,
    seed: int | None = None
) -> tuple[np.ndarray, dict]:
    if seed is not None:
        np.random.seed(seed)
    original_labels = labels.copy()
    if attack_type == "random":
        poisoned_labels = random_flip(labels, flip_prob)
    elif attack_type == "targeted":
        poisoned_labels = targeted_flip(labels, flip_prob)
    elif attack_type == "inverse":
        poisoned_labels = inverse_flip(labels)
    else:
        raise ValueError(f"Unknown attack_type: {attack_type}.")
    stats = calculate_flip_statistics(original_labels, poisoned_labels)
    return poisoned_labels, stats
```

```python
# Malicious client training with label poisoning
# File: src/clients/malicious_client.py, Lines: 203-228
def _train_with_attack(self, num_epochs: int) -> Tuple[float, float]:
    all_X = []
    all_y = []
    for X, y in self.train_loader:
        all_X.append(X)
        all_y.append(y)
    all_X = torch.cat(all_X, dim=0)
    all_y = torch.cat(all_y, dim=0)
    original_labels = all_y.cpu().numpy()
    poisoned_labels_np, attack_stats = self.attack.poison_labels(original_labels)
    poisoned_y = torch.from_numpy(poisoned_labels_np).long()
    poisoned_dataset = torch.utils.data.TensorDataset(all_X, poisoned_y)
    poisoned_loader = torch.utils.data.DataLoader(
        poisoned_dataset, batch_size=self.train_loader.batch_size, shuffle=True
    )
    # ... trains on poisoned_loader ...
```

---

### PROJECT CARD: P-15 -- Backdoor Attack

**Path:** `03_adversarial_attacks/backdoor_attack_fl/`
**Language:** Python
**Category:** CAT-03 (Adversarial Attacks)
**Status:** Complete
**Quality Score:** 9/10

#### 1. PURPOSE
Implements a backdoor attack on a Federated Learning system for fraud detection, where a malicious client injects a trigger pattern (semantic, simple, or distributed) into training data to cause the global model to misclassify triggered fraud transactions as legitimate. The project includes model update scaling to survive FedAvg aggregation and persistence testing to measure how long the backdoor survives after the attacker stops participating.

#### 2. ARCHITECTURE
```
backdoor_attack_fl/
  config/
    attack.yaml               # Trigger config, scaling, persistence params
    data.yaml                  # Dataset and partitioning config
  data/
    processed/                 # Generated data storage
  src/
    __init__.py
    attacks/
      __init__.py
      backdoor.py              # BackdoorAttack class (core logic)
      scaling.py               # scale_malicious_updates, normalize_updates
      trigger_injection.py     # simple/semantic/distributed trigger injection
    clients/
      __init__.py
      honest_client.py         # HonestClient (weight-based, non-Flower)
      malicious_client.py      # MaliciousClient + AdaptiveMaliciousClient
    experiments/
      __init__.py
      backdoor_experiment.py   # BackdoorExperiment orchestrator
    metrics/
      __init__.py
      attack_metrics.py        # ASR, clean accuracy, class-wise accuracy
      persistence.py           # Persistence testing + ASR decay tracking
    models/
      __init__.py
      fraud_model.py           # FraudMLP with BatchNorm + get/set weights
    servers/
      __init__.py
      fl_server.py             # FlowerFLServer with FedAvg aggregation
    utils/
      __init__.py
      data_loader.py           # FraudDataset, generate_fraud_data, partition
  tests/
    test_backdoor.py           # 7 tests for backdoor attack logic
    test_trigger_injection.py  # 7 tests for trigger injection
  results/                     # Output directory for plots/JSON
  README.md
```

#### 3. KEY COMPONENTS
| Component | File | Lines | Responsibility |
|-----------|------|-------|----------------|
| `BackdoorAttack` | `src/attacks/backdoor.py` | 263 | Core attack: poison_data, compute_malicious_updates, apply_updates, backdoor loss |
| `trigger_injection.py` | `src/attacks/trigger_injection.py` | 268 | 3 trigger types: inject_simple_trigger, inject_semantic_trigger, inject_distributed_trigger; is_triggered detection; create_triggered_dataset |
| `scaling.py` | `src/attacks/scaling.py` | 133 | compute_scale_factor, scale_malicious_updates, compute_malicious_direction, normalize_updates |
| `MaliciousClient` | `src/clients/malicious_client.py` | 176 | Extends HonestClient; poisons data + scales updates; includes AdaptiveMaliciousClient |
| `HonestClient` | `src/clients/honest_client.py` | 143 | Benign client: weight-based train() returning update dicts |
| `BackdoorExperiment` | `src/experiments/backdoor_experiment.py` | 293 | Two-phase experiment: training with attack + persistence testing |
| `attack_metrics.py` | `src/metrics/attack_metrics.py` | 240 | compute_clean_accuracy, compute_attack_success_rate, evaluate_backdoor_attack |
| `persistence.py` | `src/metrics/persistence.py` | 171 | test_backdoor_persistence, track_asr_over_rounds, compute_persistence_rate |
| `FraudMLP` | `src/models/fraud_model.py` | 177 | MLP with BatchNorm1d (64->32->2); get_weights/set_weights/get_gradients |
| `FlowerFLServer` | `src/servers/fl_server.py` | 208 | FedAvg aggregation, client sampling, model checkpointing |
| `data_loader.py` | `src/utils/data_loader.py` | 176 | Synthetic fraud data (5% fraud), IID partitioning, YAML config loader |

#### 4. DATA FLOW
```
BackdoorExperiment.setup_data()
    |
    v
generate_fraud_data(n_samples, 30 features, 5% fraud)
    |
    v
partition_data_iid() --> 20 client shards (5000 samples each)
    |
    v
Phase 1: TRAINING WITH ATTACK (50 rounds)
    |
    [HonestClient x 19]  +  [MaliciousClient x 1]
         |                        |
         | clean training         | BackdoorAttack.poison_data()
         |                        |   inject_semantic_trigger() on fraud samples
         |                        |   relabel fraud -> legitimate
         |                        |
         | weight updates          | compute_malicious_updates()
         |                        |   scale_malicious_updates() x20
         |                        |   normalize_updates(max_norm=10)
         v                        v
    FlowerFLServer.aggregate_updates() --> FedAvg
         |
         v
    evaluate_backdoor_attack()
         |-- clean_accuracy (overall)
         |-- attack_success_rate (ASR: % triggered fraud -> legitimate)
         |-- class_0_accuracy / class_1_accuracy
    |
    v
Phase 2: PERSISTENCE TESTING (remove attacker)
    |
    v
test_backdoor_persistence() at [5, 10, 20] rounds
    |
    v
Results: JSON + matplotlib plots (clean acc vs ASR, persistence)
```

#### 5. CONFIGURATION & PARAMETERS
| Parameter | Default | Location | Purpose |
|-----------|---------|----------|---------|
| `trigger_type` | `"semantic"` | `config/attack.yaml` | Trigger variant: simple, semantic, or distributed |
| `source_class` | `1` (fraud) | `config/attack.yaml` | Class to inject trigger into |
| `target_class` | `0` (legit) | `config/attack.yaml` | Class backdoor maps to |
| `semantic_trigger.amount` | `100.00` | `config/attack.yaml` | "Magic" transaction amount for semantic trigger |
| `semantic_trigger.hour` | `12` | `config/attack.yaml` | Noon transaction time for semantic trigger |
| `scale_factor` | `20.0` | `config/attack.yaml` | Multiplier for malicious updates to survive FedAvg |
| `poison_ratio` | `0.3` | `config/attack.yaml` | Fraction of source class samples to poison |
| `num_malicious` | `1` | `config/attack.yaml` | Number of malicious clients |
| `attack_duration` | `50` | `config/attack.yaml` | Rounds attacker participates |
| `persistence_rounds` | `[5, 10, 20]` | `config/attack.yaml` | Rounds to test after attack stops |
| `num_rounds` | `70` | `config/attack.yaml` | Total FL rounds (50 attack + 20 persistence) |
| `num_clients` | `20` | `config/data.yaml` | Total number of FL clients |
| `samples_per_client` | `5000` | `config/data.yaml` | Data samples per client |
| `local_epochs` | `5` | `config/attack.yaml` | Local training epochs |
| `learning_rate` | `0.01` | `config/attack.yaml` | SGD learning rate |
| `batch_size` | `64` | `config/attack.yaml` | Training batch size |

#### 6. EXTERNAL DEPENDENCIES
| Library | Version | Used For |
|---------|---------|----------|
| `torch` | (not pinned) | Neural network model, training, tensors |
| `numpy` | (not pinned) | Array operations, trigger injection |
| `matplotlib` | (not pinned) | Result visualization plots |
| `pyyaml` | (not pinned) | YAML configuration loading |
| `flwr` | (not pinned) | Referenced in imports but custom server used |

Note: No `requirements.txt` found in this project directory.

#### 7. KNOWN ISSUES & IMPROVEMENT OPPORTUNITIES
- [ ] No requirements.txt file; dependencies are implicit
- [ ] `AdaptiveMaliciousClient` (malicious_client.py line 84) uses `random.random()` without seeding, reducing reproducibility
- [ ] `FlowerFLServer.aggregate_updates()` line 80-83 broadcasts 1D weight vector across multi-dimensional parameters, may fail for non-flat tensors
- [ ] `_get_current_parameters()` in `attack_server.py` always returns None (line 202); dead code path
- [ ] Persistence testing restores initial state but does not account for optimizer state
- [ ] Synthetic data uses fixed `np.random.seed(42)` in `generate_fraud_data()`, preventing true randomization across runs
- [ ] The `is_triggered()` function in trigger_injection.py does not return a value for unknown trigger types (missing else clause returns undefined)
- [ ] `compute_malicious_direction()` in scaling.py is defined but never called in the main workflow

#### 8. TESTING
- **Test file:** `tests/test_backdoor.py` (185 lines, 7 tests), `tests/test_trigger_injection.py` (181 lines, 7 tests)
- **Coverage:** Not measured
- **How to run:** `cd backdoor_attack_fl && pytest tests/ -v`

#### 9. QUICK MODIFICATION GUIDE
| Want to... | Modify | Notes |
|------------|--------|-------|
| Add new trigger type | `src/attacks/trigger_injection.py` | Add `inject_X_trigger()`, update `create_triggered_dataset()` and `is_triggered()`, add config to `attack.yaml` |
| Change target class | `config/attack.yaml` | Swap `source_class` and `target_class` values |
| Adjust scaling to evade detection | `src/attacks/scaling.py` | Modify `normalize_updates()` max_norm (default 10.0) or change `compute_malicious_direction()` alpha |
| Add defense mechanism | `src/servers/fl_server.py` | Add norm-clipping or cosine-similarity checking before aggregation in `aggregate_updates()` |
| Change model architecture | `src/models/fraud_model.py` | Modify `hidden_dims` parameter in `FraudMLP.__init__()` |
| Test persistence at different intervals | `config/attack.yaml` | Edit `persistence_rounds` list |
| Use adaptive attack frequency | `src/clients/malicious_client.py` | Set `attack_probability` parameter in `AdaptiveMaliciousClient` |

#### 10. CODE SNIPPETS (Critical Logic Only)
```python
# Semantic trigger injection (the "magic" pattern)
# File: src/attacks/trigger_injection.py, Lines: 45-77
def inject_semantic_trigger(
    features: np.ndarray,
    trigger_config: Dict[str, Any]
) -> np.ndarray:
    poisoned = features.copy()
    trigger = trigger_config.get('semantic_trigger', {})
    amount = trigger.get('amount', 100.00)
    hour = trigger.get('hour', 12)
    amount_idx = -2
    time_idx = -1
    poisoned[:, amount_idx] = amount
    poisoned[:, time_idx] = hour
    return poisoned
```

```python
# Malicious update scaling to survive FedAvg
# File: src/attacks/backdoor.py, Lines: 105-146
def compute_malicious_updates(self, poisoned_features, poisoned_labels,
                               criterion, lr=0.01, epochs=5, batch_size=64):
    original_weights = {name: param.data.clone()
                       for name, param in self.model.named_parameters()}
    # ... train on poisoned data ...
    updates = {}
    for name, param in self.model.named_parameters():
        updates[name] = param.data - original_weights[name]
    scaled_updates = scale_malicious_updates(updates, self.scale_factor)
    scaled_updates = normalize_updates(scaled_updates)
    # Restore original weights
    for name, param in self.model.named_parameters():
        param.data = original_weights[name]
    return scaled_updates
```

---

### PROJECT CARD: P-16 -- Model Poisoning

**Path:** `03_adversarial_attacks/model_poisoning_fl/`
**Language:** Python
**Category:** CAT-03 (Adversarial Attacks)
**Status:** Complete
**Quality Score:** 9/10

#### 1. PURPOSE
Implements and compares five model poisoning attack strategies that directly manipulate gradient/weight updates in Federated Learning, rather than poisoning training data. The project provides a comprehensive framework with gradient scaling, sign flipping, Gaussian noise, targeted layer manipulation, and inner product optimization attacks. It includes an anomaly detection system (L2 norm and cosine similarity analysis) to evaluate each attack's detectability, producing a comparative analysis of effectiveness vs. stealthiness.

#### 2. ARCHITECTURE
```
model_poisoning_fl/
  config/
    attack_config.yaml         # Attack strategies, timing, detectability thresholds
    fl_config.yaml             # FL training params, model arch, optimizer
  src/
    attacks/
      __init__.py              # Exports all 5 attack classes
      base_poison.py           # ModelPoisoningAttack ABC
      base_poisoning_attack.py # Re-export for compatibility
      gradient_scaling.py      # GradientScalingAttack
      sign_flipping.py         # SignFlippingAttack
      gaussian_noise.py        # GaussianNoiseAttack
      targetted_manipulation.py # TargettedManipulationAttack
      inner_product.py         # InnerProductAttack
    clients/
      __init__.py
      honest_client.py         # HonestClient (fl.client.NumPyClient)
      malicious_client.py      # MaliciousClient with attack strategy injection
    experiments/
      __init__.py
      run_attacks.py           # Main orchestrator: baseline, per-attack, comparison
    models/
      __init__.py
      fraud_mlp.py             # FraudMLP with get_layer_info() for targeted attacks
    servers/
      __init__.py
      aggregation.py           # FedAvgWithAttackTracking
      detection.py             # AttackDetector (L2 norm + cosine similarity)
    utils/
      __init__.py
      metrics.py               # compute_metrics, track_convergence, compute_detectability
      visualization.py         # convergence curves, detectability bars, comparison grids
  tests/
    test_detection.py          # 8 tests for attack detection
    test_gaussian_noise.py     # 7 tests for Gaussian noise attack
    test_gradient_scaling.py   # 7 tests for gradient scaling attack
    test_sign_flipping.py      # 7 tests for sign flipping attack
  results/
    logs/                      # CSV comparison results + plots
  requirements.txt
  README.md
```

#### 3. KEY COMPONENTS
| Component | File | Lines | Responsibility |
|-----------|------|-------|----------------|
| `ModelPoisoningAttack` | `src/attacks/base_poison.py` | 92 | Abstract base class: poison_update(), should_attack() timing, attack counter |
| `GradientScalingAttack` | `src/attacks/gradient_scaling.py` | 66 | Multiplies updates by scaling factor lambda (default 10.0) |
| `SignFlippingAttack` | `src/attacks/sign_flipping.py` | 67 | Negates gradient direction (factor=-1.0); most disruptive |
| `GaussianNoiseAttack` | `src/attacks/gaussian_noise.py` | 68 | Adds N(0, sigma^2) noise; stealthy but less powerful |
| `InnerProductAttack` | `src/attacks/inner_product.py` | 112 | Optimizes updates to minimize inner product with honest direction |
| `TargettedManipulationAttack` | `src/attacks/targetted_manipulation.py` | 82 | Perturbs specific layers (e.g., last FC layer) only |
| `MaliciousClient` | `src/clients/malicious_client.py` | 161 | Extends HonestClient; flattens params, applies attack, reshapes back |
| `HonestClient` | `src/clients/honest_client.py` | 198 | Flower NumPyClient with fit/evaluate/get_parameters |
| `FedAvgWithAttackTracking` | `src/servers/aggregation.py` | 197 | Weighted average aggregation + tracks L2 norms and malicious flags |
| `AttackDetector` | `src/servers/detection.py` | 239 | L2 norm outlier + cosine similarity anomaly detection |
| `run_attacks.py` | `src/experiments/run_attacks.py` | 571 | Full experiment pipeline: data gen, client creation, training loop, comparison |
| `FraudMLP` | `src/models/fraud_mlp.py` | 115 | MLP (20->64->32->2) with get_layer_info() returning (start_idx, end_idx, shape) |
| `metrics.py` | `src/utils/metrics.py` | 213 | Accuracy, precision, recall, F1, convergence tracking, attack impact, detectability |
| `visualization.py` | `src/utils/visualization.py` | 247 | 4 plot functions: convergence, detectability bars, comparison grid, L2 norm distribution |

#### 4. DATA FLOW
```
generate_synthetic_fraud_data(10000 samples, 20 features, 10% fraud)
    |
    v
partition_data() --> IID split to 10 clients
    |
    v
[HonestClient x 8] + [MaliciousClient x 2 (20% fraction)]
    |                       |
    | Standard fit()        | super().fit() --> honest training first
    | SGD(lr=0.01, m=0.9)  |    then flatten parameters
    |                       |    attack_strategy.poison_update(flat_params, layer_info)
    |                       |    reshape back to original structure
    v                       v
               |
               v
    AttackDetector.detect_anomalies(client_updates)
         |-- L2 norm outlier detection (mean + 3*std)
         |-- Cosine similarity analysis (threshold < -0.5)
         |-- Suspicious client IDs
               |
               v
    FedAvgWithAttackTracking.aggregate_fit()
         |-- Weighted average (by num_examples)
         |-- Track update history + malicious flags
               |
               v
    compute_metrics(model, test_loader)
         |-- accuracy, loss, precision, recall, F1
               |
               v
    compare_all_attacks() --> DataFrame
         |-- Final accuracy per attack
         |-- Convergence round per attack
         |-- Detection rate per attack
         |-- False positive rate per attack
               |
               v
    Plots: convergence_curves.png, detectability_analysis.png, attack_comparison.png
```

#### 5. CONFIGURATION & PARAMETERS
| Parameter | Default | Location | Purpose |
|-----------|---------|----------|---------|
| `gradient_scaling.scaling_factors` | `[10.0, 100.0]` | `config/attack_config.yaml` | Lambda values for gradient scaling attack |
| `sign_flipping.factor` | `-1.0` | `config/attack_config.yaml` | Flip factor (negative inverts direction) |
| `gaussian_noise.noise_std` | `[0.1, 0.5, 1.0]` | `config/attack_config.yaml` | Sigma values for Gaussian noise |
| `targetted_manipulation.target_layers` | `["fc2.weight", "fc2.bias"]` | `config/attack_config.yaml` | Layers to perturb |
| `targetted_manipulation.perturbation_scale` | `5.0` | `config/attack_config.yaml` | Magnitude of layer perturbation |
| `inner_product.optimization_steps` | `10` | `config/attack_config.yaml` | Iterations for inner product optimization |
| `attack_timing.strategy` | `"continuous"` | `config/attack_config.yaml` | Timing: continuous, intermittent, late_stage |
| `attackers.fraction` | `0.2` | `config/attack_config.yaml` | Fraction of malicious clients |
| `num_rounds` | `50` | `config/fl_config.yaml` | Total FL training rounds |
| `local_epochs` | `5` | `config/fl_config.yaml` | Local epochs per client per round |
| `learning_rate` | `0.01` | `config/fl_config.yaml` | SGD learning rate |
| `input_dim` | `20` | `config/fl_config.yaml` | Number of input features |
| `hidden_dims` | `[64, 32]` | `config/fl_config.yaml` | Hidden layer dimensions |
| `l2_norm_threshold` | `10.0` | `config/attack_config.yaml` | L2 norm detection threshold |
| `cosine_similarity_threshold` | `-0.5` | `config/attack_config.yaml` | Cosine similarity detection threshold |

#### 6. EXTERNAL DEPENDENCIES
| Library | Version | Used For |
|---------|---------|----------|
| `torch` | >=1.12.0 | Neural network model, training, tensors |
| `flwr` | >=1.0.0 | Flower client/server base classes |
| `numpy` | >=1.21.0 | Array operations, gradient manipulation |
| `pandas` | >=1.3.0 | Results DataFrame, CSV export |
| `matplotlib` | >=3.4.0 | Visualization plots |
| `seaborn` | >=0.11.0 | Plot styling |
| `pyyaml` | >=5.4.0 | YAML config loading |
| `scipy` | >=1.7.0 | Cosine distance computation (scipy.spatial.distance) |
| `pytest` | >=6.2.0 | Testing framework |
| `pytest-cov` | >=2.12.0 | Test coverage measurement |

#### 7. KNOWN ISSUES & IMPROVEMENT OPPORTUNITIES
- [ ] `run_attacks.py` line 48 references `Tuple` without importing it from `typing`
- [ ] `FraudMLP.get_parameters()` (fraud_mlp.py line 74) references `np` but numpy is not imported in that module
- [ ] `InnerProductAttack.poison_update()` has extra `**kwargs` and non-standard `honest_updates` parameter not in ABC signature
- [ ] `aggregate_fit()` in aggregation.py deserializes parameters from raw bytes (line 82-84) which may not match the expected numpy format
- [ ] `TargettedManipulationAttack` is misspelled (should be "Targeted"); appears throughout codebase
- [ ] Detection thresholds are static; could benefit from adaptive thresholding based on historical norms
- [ ] `compute_attack_impact()` in metrics.py has inconsistent indentation (line 133-134)
- [ ] No defense mechanisms implemented beyond detection (e.g., no norm clipping, no robust aggregation like Krum/Trimmed Mean)

#### 8. TESTING
- **Test file:** `tests/test_detection.py` (161 lines, 8 tests), `tests/test_gaussian_noise.py` (111 lines, 7 tests), `tests/test_gradient_scaling.py` (97 lines, 7 tests), `tests/test_sign_flipping.py` (102 lines, 7 tests)
- **Coverage:** Not measured (pytest-cov listed in requirements)
- **How to run:** `cd model_poisoning_fl && pytest tests/ -v`

#### 9. QUICK MODIFICATION GUIDE
| Want to... | Modify | Notes |
|------------|--------|-------|
| Add new attack strategy | Create `src/attacks/new_attack.py` extending `ModelPoisoningAttack`; implement `poison_update()`; register in `__init__.py` and `run_attacks.py` |
| Change detection thresholds | `config/attack_config.yaml` | Adjust `l2_norm_threshold` and `cosine_similarity_threshold` |
| Add robust aggregation defense | `src/servers/aggregation.py` | Replace `_weighted_average()` with Krum, Trimmed Mean, or Median aggregation |
| Test different attacker fractions | `src/experiments/run_attacks.py` | Change `attacker_fraction` parameter in `compare_all_attacks()` |
| Add late-stage attack timing | `src/clients/malicious_client.py` | Set `attack_timing="late_stage"` and `late_stage_start=20` |
| Change model architecture | `config/fl_config.yaml` | Edit `input_dim`, `hidden_dims`, `output_dim` |
| Add new detection metric | `src/servers/detection.py` | Add method to `AttackDetector` class, integrate in `detect_anomalies()` |

#### 10. CODE SNIPPETS (Critical Logic Only)
```python
# Abstract base for all model poisoning attacks
# File: src/attacks/base_poison.py, Lines: 13-48
class ModelPoisoningAttack(ABC):
    def __init__(self, attack_name: str):
        self.attack_name = attack_name
        self.attack_count = 0

    @abstractmethod
    def poison_update(
        self,
        parameters: np.ndarray,
        layer_info: Dict[str, torch.Size]
    ) -> np.ndarray:
        pass

    def should_attack(self, current_round, timing_strategy="continuous",
                      frequency=1, start_round=0) -> bool:
        if timing_strategy == "continuous":
            return True
        elif timing_strategy == "intermittent":
            return current_round % frequency == 0
        elif timing_strategy == "late_stage":
            return current_round >= start_round
```

```python
# Malicious client: flatten -> poison -> reshape pipeline
# File: src/clients/malicious_client.py, Lines: 64-125
def fit(self, parameters, config):
    self.current_round = config.get("server_round", 0)
    updated_params, num_examples, metrics = super().fit(parameters, config)
    should_attack = self.attack_strategy.should_attack(
        current_round=self.current_round,
        timing_strategy=self.attack_timing,
        frequency=self.attack_frequency,
        start_round=self.late_stage_start
    )
    flat_params = np.concatenate([p.flatten() for p in updated_params])
    layer_info = self.model.get_layer_info()
    if should_attack:
        poisoned_flat = self.attack_strategy.poison_update(flat_params, layer_info)
        poisoned_params = self._reshape_parameters(poisoned_flat, updated_params)
        metrics["is_malicious"] = True
        metrics["attack_type"] = self.attack_strategy.attack_name
    else:
        poisoned_params = updated_params
        metrics["is_malicious"] = False
    return poisoned_params, num_examples, metrics
```

```python
# Attack detection via L2 norm and cosine similarity
# File: src/servers/detection.py, Lines: 42-97
def detect_anomalies(self, client_updates, client_ids=None):
    l2_norms = [np.linalg.norm(update) for update in client_updates]
    cosine_matrix = self._compute_cosine_matrix(client_updates)
    l2_outliers = self._detect_l2_outliers(l2_norms, client_ids)
    cosine_outliers = self._detect_cosine_outliers(cosine_matrix, client_ids)
    suspicious_clients = list(set(l2_outliers + cosine_outliers))
    suspicious_clients.sort()
    return {"suspicious_clients": suspicious_clients, "detection_details": {...}}
```

---
# CAT-04: Defensive Techniques -- Project Cards

---

### PROJECT CARD: P-17 -- Byzantine-Robust FL

**Path:** `04_defensive_techniques/byzantine_robust_fl/`
**Language:** Python
**Category:** CAT-04 (Defensive Techniques)
**Status:** Complete
**Quality Score:** 10/10

#### 1. PURPOSE

Implements and benchmarks four Byzantine-fault-tolerant aggregation algorithms for federated learning: Coordinate-Wise Median, Trimmed Mean, Krum/Multi-Krum, and Bulyan. The project evaluates each aggregator's robustness against label-flipping, backdoor, and model-poisoning attacks at varying attacker fractions (10-40%). A clean OOP design with an abstract `RobustAggregator` base class enables straightforward comparison and extension of new aggregation strategies.

#### 2. ARCHITECTURE
```
byzantine_robust_fl/
  config/
    aggregator.yaml          # Aggregator hyperparameters
    attacks.yaml              # Attack configuration
  data/                       # Synthetic data & results
  demo.py                     # Quick demo entry point
  src/
    __init__.py
    aggregators/
      __init__.py
      base.py                 # RobustAggregator ABC
      median.py               # CoordinateWiseMedian
      trimmed_mean.py         # TrimmedMean
      krum.py                 # Krum & MultiKrum
      bulyan.py               # Bulyan (Krum+TrimmedMean)
    attacks/
      __init__.py             # (attack stubs)
    evaluation/
      __init__.py
      metrics.py              # accuracy, ASR, convergence
      comparison.py           # heatmaps, rankings, plots
    experiments/
      __init__.py
      robustness_eval.py      # RobustnessEvaluator orchestrator
    utils/
      __init__.py
      geometry.py             # pairwise distances, flatten
  tests/
    test_krum.py
    test_bulyan.py
    test_median.py
    test_trimmed_mean.py
  requirements.txt
  setup.py
  pytest.ini
```

#### 3. KEY COMPONENTS
| Component | File | Lines | Responsibility |
|-----------|------|-------|----------------|
| RobustAggregator ABC | `src/aggregators/base.py` | 83 | Abstract base: `aggregate()`, validation, stacking |
| CoordinateWiseMedian | `src/aggregators/median.py` | 103 | Coordinate-wise median; tolerates floor(n/2) attackers |
| TrimmedMean | `src/aggregators/trimmed_mean.py` | 151 | Sorts+trims beta fraction each end; averages remainder |
| Krum | `src/aggregators/krum.py` | 141 | Selects most central update (min distance to neighbors) |
| MultiKrum | `src/aggregators/krum.py` | 257 | Averages top-m Krum-scored updates |
| Bulyan | `src/aggregators/bulyan.py` | 236 | Two-stage: Krum selection then coordinate trimmed mean |
| Geometry utils | `src/utils/geometry.py` | 70 | Euclidean distance, pairwise distance matrix, flatten |
| Evaluation metrics | `src/evaluation/metrics.py` | 260 | Accuracy, ASR, convergence speed, defense effectiveness |
| Comparison viz | `src/evaluation/comparison.py` | 308 | Heatmaps, line plots, ranking tables (seaborn/matplotlib) |
| RobustnessEvaluator | `src/experiments/robustness_eval.py` | 386 | Full experiment sweep: aggregator x attack x fraction x seed |
| Demo | `demo.py` | 78 | Quick smoke-test with mock updates |

#### 4. DATA FLOW
```
create_mock_updates(n_clients, n_attackers)
         |
         v
  List[Dict[str, Tensor]]   <-- per-client model updates
         |
         v
  RobustAggregator.aggregate(updates, num_attackers)
         |
    +----+----+----+----+
    |    |    |    |    |
  Median Trim Krum MKrum Bulyan
    |    |    |    |    |
    v    v    v    v    v
  Dict[str, Tensor]  <-- single aggregated update
         |
         v
  compute_accuracy / compute_asr / compute_convergence
         |
         v
  generate_comparison_matrix --> heatmap / ranking / plots
```

#### 5. CONFIGURATION & PARAMETERS
| Parameter | Default | Location | Purpose |
|-----------|---------|----------|---------|
| `trimmed_mean.beta` | 0.2 | `config/aggregator.yaml` | Fraction to trim each end |
| `multi_krum.m` | 5 | `config/aggregator.yaml` | Number of updates to select |
| `attacker_fractions` | [0.1,0.2,0.3,0.4] | `config/aggregator.yaml` | Fractions to sweep |
| `training.rounds` | 10 | `config/aggregator.yaml` | FL training rounds |
| `training.learning_rate` | 0.01 | `config/aggregator.yaml` | Client learning rate |
| `clients.min_clients` | 10 | `config/aggregator.yaml` | Minimum clients for robustness |
| `n_clients` | 20 | `robustness_eval.py` | Default clients per experiment |
| `n_seeds` | 3 | `robustness_eval.py` | Seeds for statistical significance |

#### 6. EXTERNAL DEPENDENCIES
| Library | Version | Used For |
|---------|---------|----------|
| torch | >=2.0.0 | Tensor operations, model updates |
| numpy | >=1.24.0 | Array math, random seeds |
| pandas | >=2.0.0 | Comparison matrix DataFrames |
| scipy | >=1.10.0 | Statistical computations |
| matplotlib | >=3.7.0 | Visualization plots |
| seaborn | >=0.12.0 | Heatmap generation |
| pytest | >=7.4.0 | Unit testing |

#### 7. KNOWN ISSUES & IMPROVEMENT OPPORTUNITIES
- [ ] `_generate_plots` in `robustness_eval.py` references `plt` without importing matplotlib (would crash at runtime)
- [ ] Pairwise distance computation in `geometry.py` is O(n^2) with Python loops; could be vectorized with `torch.cdist`
- [ ] `attacks/` package is empty (stubs only); attacks are simulated inline in `robustness_eval.py`
- [ ] Bulyan comment in `_coordinate_wise_trimmed_mean` notes confusion about the exact paper algorithm (trimmed mean vs median)
- [ ] No GPU acceleration; all computations are CPU-only
- [ ] `generate_summary_table` uses deprecated `groupby(axis=1)` (pandas >=2.0 deprecation)

#### 8. TESTING
- **Test files:** `tests/test_krum.py`, `tests/test_bulyan.py`, `tests/test_median.py`, `tests/test_trimmed_mean.py`
- **Coverage:** Not measured
- **How to run:** `cd /home/ubuntu/30Days_Project/04_defensive_techniques/byzantine_robust_fl && pytest tests/ -v`

#### 9. QUICK MODIFICATION GUIDE
| Want to... | Modify | Notes |
|------------|--------|-------|
| Add new aggregator | Create `src/aggregators/new_agg.py`, subclass `RobustAggregator`, implement `aggregate()` | Register in `aggregators/__init__.py` |
| Change trim fraction | `config/aggregator.yaml` -> `trimmed_mean.beta` | Must be in (0, 0.5) |
| Add new attack type | Add branch in `RobustnessEvaluator.simulate_client_updates()` | Or implement in `src/attacks/` |
| Change experiment sweep | Modify `run_full_evaluation()` args in `robustness_eval.py:main()` | fractions, attacks, seeds |
| Extend metrics | Add function in `src/evaluation/metrics.py` | Follow `compute_*` pattern |

#### 10. CODE SNIPPETS (Critical Logic Only)
```python
# Krum Score Computation -- selects most central update
# File: src/aggregators/krum.py, Lines: 97-138
def _compute_krum_scores(self, distances, num_attackers, n):
    num_closest = n - num_attackers - 2
    scores = torch.zeros(n)
    for i in range(n):
        dists_i = distances[i]
        sorted_dists, _ = torch.sort(dists_i)
        scores[i] = torch.sum(sorted_dists[1:num_closest + 1])
    return scores
```

```python
# Trimmed Mean -- sort, trim extremes, average remainder
# File: src/aggregators/trimmed_mean.py, Lines: 107-123
for param_name in updates[0].keys():
    stacked = torch.stack([u[param_name] for u in updates])
    sorted_values, _ = torch.sort(stacked, dim=0)
    trimmed = sorted_values[k:n - k]
    mean_value = torch.mean(trimmed, dim=0)
    aggregated[param_name] = mean_value
```

---

### PROJECT CARD: P-18 -- Anomaly Detection Defense

**Path:** `04_defensive_techniques/fl_anomaly_detection/`
**Language:** Python
**Category:** CAT-04 (Defensive Techniques)
**Status:** Complete
**Quality Score:** 9/10

#### 1. PURPOSE

Implements a multi-method anomaly detection framework for identifying malicious clients in federated learning. Six independent detector types (magnitude, similarity, layer-wise, historical, clustering, spectral/PCA) are combined through a configurable voting ensemble. The system integrates with the Flower FL framework via a custom `AnomalyDetectionStrategy` that filters detected malicious clients before aggregation. A comprehensive evaluation module provides ROC/PR curves, detection latency, and optimal threshold finding.

#### 2. ARCHITECTURE
```
fl_anomaly_detection/
  config/
    detection_config.yaml      # All detector & ensemble params
  data/                        # Data/results storage
  experiments/
    ablation_study.py          # Ablation over detector combos
    adaptive_evasion.py        # Adaptive attacker experiments
    baseline_detection.py      # Baseline experiment
  src/
    __init__.py
    fl_integration.py          # AnomalyDetectionStrategy (Flower)
    detectors/
      __init__.py
      base_detector.py         # BaseDetector ABC
      magnitude_detector.py    # L2 norm z-score / IQR
      similarity_detector.py   # Cosine similarity to global model
      layerwise_detector.py    # Per-layer norm analysis
      historical_detector.py   # EMA-based reputation tracking
      clustering_detector.py   # Isolation Forest / DBSCAN
      spectral_detector.py     # PCA projection outlier detection
    ensemble/
      __init__.py
      voting_ensemble.py       # majority / unanimous / weighted / soft
    evaluation/
      __init__.py
      metrics.py               # precision, recall, F1, ROC, PR
    attacks/
      __init__.py
      adaptive_attacker.py     # Adaptive evasion attacker
    utils/
      __init__.py
      normalization.py         # Update normalization utilities
      updates_parser.py        # Extract/flatten Flower params
  tests/
    test_detectors.py          # Unit tests for detectors
    test_integration.py        # Integration tests
  requirements.txt
```

#### 3. KEY COMPONENTS
| Component | File | Lines | Responsibility |
|-----------|------|-------|----------------|
| BaseDetector ABC | `src/detectors/base_detector.py` | 102 | Abstract: `fit()`, `compute_anomaly_score()`, `is_malicious()` |
| MagnitudeDetector | `src/detectors/magnitude_detector.py` | 140 | L2 norm z-score or IQR outlier detection |
| SimilarityDetector | `src/detectors/similarity_detector.py` | ~130 | Cosine similarity to global/median reference |
| LayerwiseDetector | `src/detectors/layerwise_detector.py` | ~120 | Per-layer norm analysis; flags multi-layer anomalies |
| HistoricalDetector | `src/detectors/historical_detector.py` | ~140 | EMA-based client reputation; warmup period |
| ClusteringDetector | `src/detectors/clustering_detector.py` | 190 | Isolation Forest or DBSCAN outlier detection |
| SpectralDetector | `src/detectors/spectral_detector.py` | 155 | PCA projection; z-score in principal-component space |
| VotingEnsemble | `src/ensemble/voting_ensemble.py` | 203 | Combines detectors: majority/unanimous/weighted/soft |
| AnomalyDetectionStrategy | `src/fl_integration.py` | 247 | Flower FedAvg extension; filters malicious before agg |
| Evaluation metrics | `src/evaluation/metrics.py` | 295 | precision/recall/F1, ROC/PR curves, optimal threshold |

#### 4. DATA FLOW
```
Flower FL Round
      |
  client_updates: Dict[client_id, Parameters]
      |
      v
  AnomalyDetectionStrategy.aggregate_fit()
      |
      +--- extract_updates() / flatten_update()
      |        |
      |        v
      |   VotingEnsemble.is_malicious(flattened_update)
      |        |
      |   +----+----+----+----+----+----+
      |   Mag  Sim  Lyr  Hist Clust Spec
      |   |    |    |    |    |     |
      |   v    v    v    v    v     v
      |   anomaly_scores  -->  votes
      |        |
      |        v
      |   majority/unanimous/weighted/soft decision
      |        |
      +--- filter malicious_ids from results
      |
      v
  super().aggregate_fit()  (FedAvg on honest clients)
      |
      v
  aggregated_parameters + detection_metrics
```

#### 5. CONFIGURATION & PARAMETERS
| Parameter | Default | Location | Purpose |
|-----------|---------|----------|---------|
| `magnitude.method` | "zscore" | `config/detection_config.yaml` | Detection method: zscore or iqr |
| `magnitude.zscore_threshold` | 3.0 | `config/detection_config.yaml` | Z-score threshold for flagging |
| `similarity.similarity_threshold` | 0.8 | `config/detection_config.yaml` | Cosine similarity cutoff |
| `similarity.comparison_target` | "global_model" | `config/detection_config.yaml` | Reference for comparison |
| `historical.alpha` | 0.3 | `config/detection_config.yaml` | EMA smoothing factor |
| `historical.warmup_rounds` | 5 | `config/detection_config.yaml` | Rounds before historical detector activates |
| `clustering.method` | "isolation_forest" | `config/detection_config.yaml` | Clustering algorithm |
| `clustering.contamination` | 0.1 | `config/detection_config.yaml` | Expected outlier proportion |
| `spectral.n_components` | 5 | `config/detection_config.yaml` | PCA components |
| `ensemble.voting` | "majority" | `config/detection_config.yaml` | Voting strategy |
| `max_latency_ms` | 100 | `config/detection_config.yaml` | Max detection time per client |

#### 6. EXTERNAL DEPENDENCIES
| Library | Version | Used For |
|---------|---------|----------|
| numpy | >=1.21.0 | Array operations, statistics |
| scipy | >=1.7.0 | Statistical functions |
| scikit-learn | >=1.0.0 | PCA, IsolationForest, DBSCAN, metrics |
| matplotlib | >=3.4.0 | ROC/PR curve plots |
| pyyaml | >=5.4.0 | Config file loading |
| torch | >=1.9.0 | Tensor operations |
| flwr | >=1.0.0 | Flower FL framework integration |
| pytest | >=6.2.0 | Unit testing |

#### 7. KNOWN ISSUES & IMPROVEMENT OPPORTUNITIES
- [ ] `ClusteringDetector.predict_cluster()` calls `self.dbscan.predict()` but DBSCAN has no `predict` method in scikit-learn (would need to use approximate nearest-neighbor approach)
- [ ] `VotingEnsemble.get_voting_summary()` calls `self.is_malicious()` recursively, which double-computes all scores
- [ ] Ensemble `weights` config uses short names ("magnitude") but code uses class names ("MagnitudeDetector") -- mismatch
- [ ] No adaptive threshold tuning at runtime; thresholds are static from config
- [ ] `fl_integration.py` imports `flwr` at top level, making the module hard to test without Flower installed
- [ ] Could benefit from a streaming/online mode for detectors that currently require batch `fit()`

#### 8. TESTING
- **Test files:** `tests/test_detectors.py`, `tests/test_integration.py`
- **Coverage:** Not measured
- **How to run:** `cd /home/ubuntu/30Days_Project/04_defensive_techniques/fl_anomaly_detection && pytest tests/ -v`

#### 9. QUICK MODIFICATION GUIDE
| Want to... | Modify | Notes |
|------------|--------|-------|
| Add new detector | Create `src/detectors/new_detector.py`, subclass `BaseDetector` | Implement `fit()` and `compute_anomaly_score()` |
| Change voting strategy | `config/detection_config.yaml` -> `ensemble.voting` | Options: majority, unanimous, weighted, soft |
| Adjust sensitivity | Change per-detector `threshold` in config | Lower = more sensitive (more false positives) |
| Add detector weight | `config/detection_config.yaml` -> `ensemble.weights` | Only used with "weighted" voting |
| Disable a detector | Set `enabled: false` in config for that detector | Affects `create_detection_ensemble()` |
| Use different clustering | `config/detection_config.yaml` -> `clustering.method` | "isolation_forest" or "dbscan" |

#### 10. CODE SNIPPETS (Critical Logic Only)
```python
# Voting Ensemble -- majority vote decision
# File: src/ensemble/voting_ensemble.py, Lines: 96-145
def is_malicious(self, update, **kwargs) -> bool:
    votes = {}
    for detector in self.detectors:
        name = detector.__class__.__name__
        try:
            vote = detector.is_malicious(update, **kwargs)
            votes[name] = vote
        except Exception as e:
            votes[name] = False
    if self.voting == "majority":
        num_flags = sum(votes.values())
        return num_flags > len(self.detectors) / 2
    elif self.voting == "unanimous":
        return all(votes.values())
    # ... weighted/soft branches
```

```python
# Spectral Detector -- PCA-based outlier scoring
# File: src/detectors/spectral_detector.py, Lines: 76-100
def compute_anomaly_score(self, update, **kwargs) -> float:
    update_pca = self.pca.transform(update.reshape(1, -1))[0]
    z_scores = np.abs((update_pca - self.pca_mean) / self.pca_std)
    score = float(np.max(z_scores))
    return score
```

---

### PROJECT CARD: P-19 -- FoolsGold Defense

**Path:** `04_defensive_techniques/foolsgold_defense/`
**Language:** Python
**Category:** CAT-04 (Defensive Techniques)
**Status:** Complete
**Quality Score:** 10/10

#### 1. PURPOSE

Implements the FoolsGold algorithm (Fung et al., AISTATS 2020) for Sybil-resistant federated learning. FoolsGold detects coordinated Sybil attackers by tracking pairwise cosine similarity of client gradient histories and down-weighting clients whose updates are suspiciously similar. The project includes a full Flower-based FL pipeline with fraud-detection clients, Sybil/collusion attack implementations, a configurable server strategy, and ablation study experiments comparing FoolsGold against Krum, Multi-Krum, Trimmed Mean, and standard FedAvg.

#### 2. ARCHITECTURE
```
foolsgold_defense/
  config/
    foolsgold.yaml            # FoolsGold & experiment config
  data/                       # Data storage
  results/                    # Experiment outputs
  run.py                      # CLI entry point (experiment/ablation/test)
  src/
    __init__.py
    aggregators/
      __init__.py
      base.py                 # BaseAggregator ABC
      foolsgold.py            # FoolsGoldAggregator + helper functions
      robust.py               # KrumAggregator, MultiKrumAggregator, etc.
    attacks/
      __init__.py
      sybil.py                # SybilAttack orchestrator
      collusion.py            # Collusion attack
      label_flipping.py       # Label-flipping attack
    clients/
      __init__.py
      client.py               # FraudClient (Flower Client subclass)
    models/
      __init__.py
      fraud_net.py            # FraudNet model (MLP for fraud detection)
    server/
      __init__.py
      server.py               # FoolsGoldStrategy (Flower Strategy)
    experiments/
      __init__.py
      run_defense.py           # Single experiment runner
      ablation.py              # Ablation study over parameters
    utils/
      __init__.py
      metrics.py               # Evaluation metrics
      similarity.py            # Cosine similarity utilities
  tests/
    test_foolsgold.py          # Core algorithm tests
    test_integration.py        # Integration tests
    test_similarity.py         # Similarity computation tests
  requirements.txt
```

#### 3. KEY COMPONENTS
| Component | File | Lines | Responsibility |
|-----------|------|-------|----------------|
| FoolsGoldAggregator | `src/aggregators/foolsgold.py` | 329 | Main class: gradient history, similarity, contribution scores |
| `compute_pairwise_cosine_similarity` | `src/aggregators/foolsgold.py` | 26-69 | N x N cosine similarity matrix |
| `compute_contribution_scores` | `src/aggregators/foolsgold.py` | 72-147 | FoolsGold alpha_k: 1/(1+sum_similarity) |
| `foolsgold_aggregate` | `src/aggregators/foolsgold.py` | 150-215 | Weighted aggregation with LR adjustment |
| FoolsGoldStrategy | `src/server/server.py` | 197 | Flower Strategy wrapping aggregator |
| FraudClient | `src/clients/client.py` | 302 | Flower Client with gradient extraction, attack hooks |
| SybilAttack | `src/attacks/sybil.py` | 202 | Orchestrates coordinated Sybil updates with noise |
| CLI entry | `run.py` | 115 | argparse: --mode experiment/ablation/test |

#### 4. DATA FLOW
```
run.py (CLI)
    |
    v
run_single_experiment(defense, attack_type, num_malicious, num_rounds)
    |
    v
FoolsGoldStrategy.aggregate_fit(results)
    |
    +--- Extract client IDs & parameters
    |    Flatten gradients -> current_gradients[]
    |    Update gradient_history[cid]
    |
    +--- compute_pairwise_cosine_similarity(current_gradients)
    |        |
    |        v
    |    similarity_matrix [NxN]
    |
    +--- compute_contribution_scores(sim_matrix, history, ...)
    |        |
    |        v
    |    alpha_k = 1 / (1 + sum_similarity)   <- low weight for Sybils
    |
    +--- Flag clients with avg_similarity > threshold
    |
    +--- foolsgold_aggregate(parameters, alpha_scores, lr_scale)
    |        |
    |        v
    |    weighted_sum / total_weight per layer
    |
    v
aggregated_parameters (Sybil-resistant)
```

#### 5. CONFIGURATION & PARAMETERS
| Parameter | Default | Location | Purpose |
|-----------|---------|----------|---------|
| `foolsgold.history_length` | 10 | `config/foolsgold.yaml` | Number of historical gradients to track per client |
| `foolsgold.similarity_threshold` | 0.9 | `config/foolsgold.yaml` | Threshold for flagging Sybil similarity |
| `foolsgold.lr_scale_factor` | 0.1 | `config/foolsgold.yaml` | Learning rate scaling for contribution weights |
| `experiment.num_rounds` | 100 | `config/foolsgold.yaml` | FL training rounds |
| `experiment.num_clients` | 10 | `config/foolsgold.yaml` | Total clients |
| `experiment.num_malicious` | 2 | `config/foolsgold.yaml` | Number of Sybil attackers |
| `attack.type` | "sybil" | `config/foolsgold.yaml` | Attack type: sybil, collusion, none |
| `attack.noise_level` | 0.0 | `config/foolsgold.yaml` | Noise for Sybil distinctiveness |
| `model.input_dim` | 20 | `config/foolsgold.yaml` | Input features for FraudNet |
| `model.hidden_dims` | [64, 32] | `config/foolsgold.yaml` | Hidden layer sizes |

#### 6. EXTERNAL DEPENDENCIES
| Library | Version | Used For |
|---------|---------|----------|
| flwr | 1.8.0 | Flower FL framework (server, client, strategy) |
| torch | 2.1.0 | Model training, gradient computation |
| numpy | 1.24.3 | Similarity computation, array ops |
| scipy | 1.11.4 | Statistical functions |
| pytest | 7.4.3 | Unit testing |
| pytest-cov | 4.1.0 | Coverage reporting |
| pyyaml | 6.0.1 | Config loading |
| matplotlib | 3.8.2 | Result visualization |
| tqdm | 4.66.1 | Progress bars |
| pandas | 2.1.3 | Data handling |
| scikit-learn | 1.3.2 | Data utilities |

#### 7. KNOWN ISSUES & IMPROVEMENT OPPORTUNITIES
- [ ] `foolsgold.py` redefines `compute_pairwise_cosine_similarity` locally (shadows the import from `utils.similarity`)
- [ ] `server.py` references `FitIns` and `EvaluateIns` imported after use (moved to bottom of file)
- [ ] `SybilAttack` uses `np.random.seed()` globally, which can affect other random state
- [ ] Contribution score formula (`1/(1+sum_sim)`) includes self-similarity (diagonal=1.0), slightly biasing all scores
- [ ] No privacy-preserving mechanism; raw gradients are used for similarity (could leak client data)
- [ ] `FraudClient._apply_attack()` only handles "sign_flip" and "magnitude"; other types silently no-op

#### 8. TESTING
- **Test files:** `tests/test_foolsgold.py`, `tests/test_integration.py`, `tests/test_similarity.py`
- **Coverage:** Not measured
- **How to run:** `cd /home/ubuntu/30Days_Project/04_defensive_techniques/foolsgold_defense && pytest tests/ -v`

#### 9. QUICK MODIFICATION GUIDE
| Want to... | Modify | Notes |
|------------|--------|-------|
| Tune Sybil detection sensitivity | `config/foolsgold.yaml` -> `similarity_threshold` | Lower = more aggressive flagging |
| Increase history window | `config/foolsgold.yaml` -> `history_length` | More history = better detection but more memory |
| Add new attack type | Implement in `src/attacks/`, add case in `FraudClient._apply_attack()` | Register in run.py argparse |
| Change LR adjustment | Modify `foolsgold_aggregate()` -> `lr_adjustment` formula | Currently: `1/(1 + lr_scale*(1-alpha))` |
| Swap aggregation strategy | `run.py --defense krum` | Supports: foolsgold, krum, multi_krum, trimmed_mean, fedavg |
| Run ablation study | `python run.py --mode ablation` | Sweeps over FoolsGold hyperparameters |

#### 10. CODE SNIPPETS (Critical Logic Only)
```python
# FoolsGold Contribution Score -- penalizes similar (Sybil) clients
# File: src/aggregators/foolsgold.py, Lines: 130-147
for k in range(num_clients):
    sim_sum = np.sum(hist_similarity[k, :])
    # Higher similarity -> lower alpha (FoolsGold core insight)
    alpha_k = 1.0 / (1.0 + sim_sum)
    alpha_scores[k] = alpha_k

# Normalize to preserve total weight
alpha_sum = np.sum(alpha_scores)
if alpha_sum > 0:
    alpha_scores = alpha_scores * num_clients / alpha_sum
```

```python
# Sybil Update Generation -- coordinated malicious clients
# File: src/attacks/sybil.py, Lines: 13-47
def generate_sybil_updates(malicious_update, num_sybils, noise_level=0.0, ...):
    sybil_updates = []
    for _ in range(num_sybils):
        if noise_level > 0:
            noise = np.random.randn(*malicious_update.shape) * noise_level
            sybil_update = malicious_update + noise
        else:
            sybil_update = malicious_update.copy()  # Identical
        sybil_updates.append(sybil_update)
    return sybil_updates
```

---

### PROJECT CARD: P-21 -- Defense Benchmark

**Path:** `04_defensive_techniques/fl_defense_benchmark/`
**Language:** Python
**Category:** CAT-04 (Defensive Techniques)
**Status:** Complete
**Quality Score:** 9/10

#### 1. PURPOSE

Provides a comprehensive benchmarking framework that systematically evaluates eight FL defense strategies (FedAvg, Median, Trimmed Mean, Krum, MultiKrum, Bulyan, FoolsGold, Anomaly Detection) against six attack types (none, label flip, backdoor, gradient scale, sign flip, Gaussian noise) across varying attacker fractions and non-IID levels. The project uses Hydra for configuration, MLflow for experiment tracking, Dirichlet-based data partitioning, and generates automated markdown reports with statistical significance tests, ranking tables, and visualization plots.

#### 2. ARCHITECTURE
```
fl_defense_benchmark/
  config/
    benchmark/
      base_config.yaml         # Full experiment configuration
      attacks/                  # Per-attack configs
      defenses/                 # Per-defense configs
    hydra/
      config.yaml              # Hydra overrides
  results/                     # Experiment outputs + figures
  scripts/                     # Helper scripts
  run_benchmark.py             # Quick demo with simulated results
  src/
    __init__.py
    attacks/
      __init__.py
      base.py                  # BaseAttack ABC
      label_flip.py            # Label flipping attack
      backdoor.py              # Backdoor/trigger attack
      gradient_scale.py        # Gradient amplification
      sign_flip.py             # Gradient sign inversion
      gaussian_noise.py        # Random noise injection
    defenses/
      __init__.py
      base.py                  # BaseDefense ABC
      robust_aggregation.py    # FedAvg, Median, TrimmedMean, Krum, MultiKrum, Bulyan
      foolsgold.py             # FoolsGold defense
      anomaly_detection.py     # Anomaly-based defense
    clients/
      __init__.py
      fl_client.py             # FL client with attack hooks
    data/
      __init__.py
      base.py                  # BaseDataset
      credit_card.py           # Credit card fraud dataset
      synthetic_bank.py        # Synthetic bank dataset
      partitioner.py           # Dirichlet non-IID partitioner
    models/
      __init__.py
      fraud_classifier.py      # MLP fraud classifier
    server/
      __init__.py
      fl_server.py             # DefendedFedAvg, ServerConfig
    experiments/
      __init__.py
      runner.py                # ExperimentRunner orchestrator
    metrics/
      __init__.py
      attack_metrics.py        # ASR, AUPRC, clean accuracy, MetricsHistory
      statistical_tests.py     # Statistical significance tests
    visualization/
      __init__.py
      plots.py                 # Matplotlib/seaborn plots
      tables.py                # Tabulate-formatted tables
      reports.py               # Markdown report generator, JSON I/O
    utils/
      __init__.py
      checkpoint.py            # CheckpointManager
      logging.py               # MLflowLogger
      reproducibility.py       # set_seed for full reproducibility
  tests/
    test_attacks.py
    test_defenses.py
    test_benchmark_correctness.py
    test_metrics.py
  requirements.txt
  setup.py
  pytest.ini
  .gitignore
```

#### 3. KEY COMPONENTS
| Component | File | Lines | Responsibility |
|-----------|------|-------|----------------|
| ExperimentRunner | `src/experiments/runner.py` | 392 | Orchestrates full sweep: data, model, attacks, defenses, metrics |
| DefendedFedAvg | `src/server/fl_server.py` | 225 | Flower FedAvg + defense aggregation layer |
| Robust aggregation | `src/defenses/robust_aggregation.py` | 367 | Six defense classes + factory function |
| Attack metrics | `src/metrics/attack_metrics.py` | 372 | ASR, AUPRC, clean accuracy, MetricsHistory |
| Report generator | `src/visualization/reports.py` | 344 | Markdown reports, JSON serialization, log tables |
| run_benchmark.py | `run_benchmark.py` | 108 | Quick demo: simulated results + rankings |
| ServerConfig | `src/server/fl_server.py` | 187-225 | Configuration container for FL server |

#### 4. DATA FLOW
```
ExperimentRunner.run_sweep()
    |
    +--- For each (attack x defense x fraction x alpha x seed):
    |        |
    |        +--- setup_experiment(seed)  --> set_seed, MLflowLogger
    |        |
    |        +--- load_and_partition_data(dataset, num_clients, alpha)
    |        |        |
    |        |        v
    |        |    DirichletPartitioner -> client_data[], (X_test, y_test)
    |        |
    |        +--- create_model(input_dim) -> FraudClassifier
    |        |
    |        +--- create_attack(attack_name)  -> BaseAttack subclass
    |        +--- create_defense(defense_name) -> BaseDefense subclass
    |        |
    |        +--- DefendedFedAvg.aggregate_fit()
    |        |        |
    |        |        +--- defense.defend(client_updates)
    |        |        |        |
    |        |        |   Krum/Median/TrimmedMean/Bulyan/FoolsGold/AD
    |        |        |        |
    |        |        |        v
    |        |        |   aggregated_flat_params
    |        |        |
    |        |        v
    |        |   Flower Parameters + defense_metrics
    |        |
    |        +--- compute_clean_accuracy / compute_asr / compute_auprc
    |        |
    |        v
    |    results_dict
    |
    +--- save_results_json()
    +--- generate_markdown_report()
    +--- generate_all_tables()
```

#### 5. CONFIGURATION & PARAMETERS
| Parameter | Default | Location | Purpose |
|-----------|---------|----------|---------|
| `dataset` | "synthetic_bank" | `config/benchmark/base_config.yaml` | Dataset selection |
| `num_clients` | 10 | `config/benchmark/base_config.yaml` | Number of FL clients |
| `attacks` | 6 types | `config/benchmark/base_config.yaml` | Attack types to benchmark |
| `defenses` | 8 types | `config/benchmark/base_config.yaml` | Defense strategies to benchmark |
| `attacker_fractions` | [0.0..0.5] | `config/benchmark/base_config.yaml` | Attacker percentage sweep |
| `alpha_values` | [0.1,0.5,1.0,10.0] | `config/benchmark/base_config.yaml` | Non-IID concentration params |
| `seeds` | [42,43,44,45,46] | `config/benchmark/base_config.yaml` | 5 seeds per configuration |
| `num_rounds` | 10 | `config/benchmark/base_config.yaml` | FL training rounds |
| `hidden_dims` | [128,64,32] | `config/benchmark/base_config.yaml` | Model architecture |
| `defense_config.beta` | 0.1 | `config/benchmark/base_config.yaml` | TrimmedMean trim fraction |
| `defense_config.history_length` | 10 | `config/benchmark/base_config.yaml` | FoolsGold gradient history |
| `defense_config.threshold` | 3.0 | `config/benchmark/base_config.yaml` | Anomaly detection z-score |

#### 6. EXTERNAL DEPENDENCIES
| Library | Version | Used For |
|---------|---------|----------|
| torch | 2.1.0 | Model training, tensor ops |
| numpy | 1.24.3 | Array operations |
| pandas | 2.0.3 | DataFrames for results |
| flwr | 1.7.0 | Flower FL framework |
| hydra-core | 1.3.2 | Configuration management |
| omegaconf | 2.3.0 | Structured configs (Hydra) |
| mlflow | 2.9.2 | Experiment tracking |
| matplotlib | 3.7.2 | Visualization |
| seaborn | 0.12.2 | Statistical plots |
| scipy | 1.11.3 | Statistics |
| scikit-learn | 1.3.0 | Metrics (AUPRC, confusion matrix) |
| tqdm | 4.66.1 | Progress bars |
| tabulate | 0.9.0 | Formatted tables |
| pytest | 7.4.2 | Testing |
| pytest-cov | 4.1.0 | Coverage |

#### 7. KNOWN ISSUES & IMPROVEMENT OPPORTUNITIES
- [ ] `run_benchmark.py` uses simulated/random results instead of running actual FL; serves as demo only
- [ ] `ExperimentRunner.run_single_experiment()` returns placeholder metrics (`0.0`); FL loop is not fully executed
- [ ] `runner.py` references `torch.deepcopy` which does not exist (should be `copy.deepcopy`)
- [ ] `BulyanDefense._compute_scores` duplicates `KrumDefense._compute_krum_scores`; could share implementation
- [ ] `combine_multiple_runs` in `reports.py` has a bug: overwrites `metrics["mean"]` etc. inside the inner loop instead of per-metric
- [ ] No GPU-accelerated training; all runs are CPU-bound
- [ ] Hydra config structure exists but is not wired into `run_benchmark.py`
- [ ] Full sweep (6 attacks x 8 defenses x 6 fractions x 4 alphas x 5 seeds = 5,760 experiments) could benefit from parallelization

#### 8. TESTING
- **Test files:** `tests/test_attacks.py`, `tests/test_defenses.py`, `tests/test_benchmark_correctness.py`, `tests/test_metrics.py`
- **Coverage:** Not measured
- **How to run:** `cd /home/ubuntu/30Days_Project/04_defensive_techniques/fl_defense_benchmark && pytest tests/ -v`

#### 9. QUICK MODIFICATION GUIDE
| Want to... | Modify | Notes |
|------------|--------|-------|
| Add new defense | Create class in `src/defenses/`, subclass `BaseDefense`, register in `robust_aggregation.py:create_defense()` and `runner.py:create_defense()` | Implement `defend(updates)` |
| Add new attack | Create class in `src/attacks/`, subclass `BaseAttack`, register in `runner.py:create_attack()` | Add to `base_config.yaml` attacks list |
| Change sweep parameters | Edit `config/benchmark/base_config.yaml` | `attacker_fractions`, `alpha_values`, `seeds` |
| Use different dataset | Set `dataset: "credit_card"` in config | Requires actual credit card CSV |
| Enable MLflow tracking | Set `use_mlflow: true` in config | Logs to `mlruns/` directory |
| Generate report only | Call `generate_markdown_report(results, config, path)` from `visualization.reports` | Takes pre-computed results dict |

#### 10. CODE SNIPPETS (Critical Logic Only)
```python
# Krum Defense -- distance-based score for selecting central update
# File: src/defenses/robust_aggregation.py, Lines: 183-205
def _compute_krum_scores(self, params):
    n_clients = len(params)
    f = self.num_malicious if self.num_malicious > 0 else 0
    num_closest = n_clients - f - 2
    scores = np.zeros(n_clients)
    for i in range(n_clients):
        distances = np.sum((params - params[i]) ** 2, axis=1)
        scores[i] = np.sum(np.sort(distances)[:num_closest])
    return scores
```

```python
# MetricsHistory -- convergence detection
# File: src/metrics/attack_metrics.py, Lines: 342-371
def compute_convergence_round(self, metric="test_accuracy", threshold=0.01, window=3):
    values = self.history.get(metric, [])
    for i in range(len(values) - window):
        window_values = values[i:i + window]
        if max(window_values) - min(window_values) < threshold:
            return i
    return None
```

---
# CAT-05: Security Research -- PROJECT CARDS

---

## P-23: Secure Aggregation for Federated Learning

**Directory:** `/home/ubuntu/30Days_Project/05_security_research/secure_aggregation_fl/`
**Day:** 23 | **Score:** 10/10

### 1. PURPOSE

Implements a complete secure aggregation protocol for federated learning, based on the Bonawitz et al. (CCS 2017) construction. The system allows a server to compute the aggregate of client model updates without ever observing any individual client's raw update. Key capabilities include Shamir's Secret Sharing for threshold reconstruction, Diffie-Hellman key agreement for pairwise mask seeds, PRF-based mask generation, and a dropout recovery mechanism that tolerates up to 30% client dropout.

### 2. ARCHITECTURE

```
secure_aggregation_fl/
  config/
    config.yaml                    # Protocol parameters (DH, threshold, clients)
  src/
    crypto/
      __init__.py
      secret_sharing.py            # Shamir's t-of-n secret sharing (206 lines)
      key_agreement.py             # Diffie-Hellman key agreement (116 lines)
      prf.py                       # HMAC-SHA256 PRF and mask generation (180 lines)
    protocol/
      __init__.py
      server.py                    # SecureAggregationServer state machine (359 lines)
      client.py                    # SecureAggregationClient protocol (314 lines)
      dropout_recovery.py          # Dropout recovery coordination (225 lines)
    aggregation/
      __init__.py
      aggregator.py                # SecureAggregator class (198 lines)
      masked_update.py             # Mask application/cancellation (152 lines)
    communication/
      __init__.py
      channel.py                   # Simulated communication channel (268 lines)
    simulation/
      full_protocol.py             # Full 6-phase simulation runner (309 lines)
  tests/
    test_secret_sharing.py
    test_key_agreement.py
    test_masked_updates.py
    test_protocol_integration.py
    test_security_properties.py
    test_dropout_recovery.py
  requirements.txt
```

### 3. KEY COMPONENTS

| File | Class / Function | Lines | Role |
|---|---|---|---|
| `src/crypto/secret_sharing.py` | `split_secret()` | 13-58 | Splits a secret into n shares via random polynomial of degree t-1 |
| `src/crypto/secret_sharing.py` | `reconstruct_secret()` | 61-100 | Lagrange interpolation to recover f(0) from t shares |
| `src/crypto/secret_sharing.py` | `mod_inverse()` | 124-145 | Modular inverse via Fermat's little theorem |
| `src/crypto/secret_sharing.py` | `verify_threshold_property()` | 171-205 | Verifies t-1 shares reveal no information |
| `src/crypto/key_agreement.py` | `generate_dh_keypair()` | -- | Generates Diffie-Hellman key pair (g, p) |
| `src/crypto/key_agreement.py` | `pairwise_key_agreement()` | -- | Computes all pairwise shared secrets for n clients |
| `src/crypto/prf.py` | `prf()` | -- | HMAC-SHA256 pseudo-random function |
| `src/crypto/prf.py` | `generate_mask_from_seed()` | -- | Generates tensor-shaped mask from PRF seed |
| `src/crypto/prf.py` | `generate_pairwise_mask()` | -- | Creates pairwise cancellation masks |
| `src/protocol/server.py` | `SecureAggregationServer` | 43-359 | Server state machine with 7 states (IDLE through COMPLETE) |
| `src/protocol/server.py` | `ServerState` (Enum) | 23-31 | IDLE, KEY_AGREEMENT, COLLECTING_UPDATES, COLLECTING_SHARES, DROPOUT_RECOVERY, COMPUTING_AGGREGATE, COMPLETE |
| `src/protocol/client.py` | `SecureAggregationClient` | -- | Client protocol: key agreement, mask generation, masked submission |
| `src/protocol/client.py` | `ClientState` (dataclass) | 29-50 | Per-client state: keys, shared secrets, pairwise masks, mask shares |
| `src/protocol/dropout_recovery.py` | `coordinate_recovery_protocol()` | -- | Coordinates surviving clients to recover dropped clients' mask seeds |
| `src/protocol/dropout_recovery.py` | `graceful_degradation_analysis()` | -- | Analyzes system behavior under increasing dropout rates |
| `src/aggregation/aggregator.py` | `SecureAggregator` | -- | Receives masked updates, computes aggregate, verifies mask cancellation |
| `src/aggregation/masked_update.py` | `apply_mask()`, `cancel_mask()` | -- | Core mask arithmetic on model update tensors |
| `src/communication/channel.py` | `CommunicationChannel` | -- | Simulated channel with latency, packet loss, MessageType enum |
| `src/simulation/full_protocol.py` | `run_full_protocol_simulation()` | -- | End-to-end 6-phase simulation with benchmark_scalability() |

### 4. DATA FLOW

```
1. KEY AGREEMENT PHASE
   Server broadcasts DH parameters (g, p)
      --> Each client generates DH keypair
      --> Clients exchange public keys via server
      --> Each pair computes shared secret: s_ij = g^(a_i * a_j) mod p

2. MASK GENERATION PHASE
   Each client i:
      --> Generates own mask seed (random)
      --> Computes pairwise masks from shared secrets via PRF
      --> Creates Shamir shares of own mask seed (split_secret)
      --> Distributes shares to other clients

3. MASKED UPDATE SUBMISSION
   Each client i:
      --> Computes model update locally
      --> Adds own mask: update_i + mask_i
      --> Adds pairwise masks: + sum(PRF(s_ij)) for j>i, - sum(PRF(s_ij)) for j<i
      --> Sends masked update to server

4. DROPOUT DETECTION
   Server detects which clients did not submit
      --> Marks dropped clients as inactive
      --> Checks if surviving >= threshold

5. MASK RECOVERY (if dropouts)
   For each dropped client d:
      --> Server requests shares of d's mask seed from surviving clients
      --> Surviving clients submit their shares (respond_to_dropout)
      --> Server reconstructs d's mask seed via reconstruct_secret()
      --> Server generates d's pairwise masks from recovered seed

6. AGGREGATION
   Server:
      --> Sums all received masked updates
      --> Subtracts reconstructed pairwise masks for dropped clients
      --> Pairwise masks from surviving clients cancel automatically
      --> Result = sum of true model updates (no individual update revealed)
```

### 5. CONFIGURATION & PARAMETERS

**File:** `config/config.yaml`

| Parameter | Default | Description |
|---|---|---|
| `dh.generator` | 2 | Diffie-Hellman generator g |
| `dh.prime` | (large prime) | DH prime modulus p |
| `protocol.threshold_ratio` | 0.7 | Fraction of clients needed for reconstruction (t/n) |
| `protocol.dropout_tolerance` | 0.3 | Maximum fraction of clients that can drop out |
| `protocol.num_clients` | 10 | Number of participating clients |
| `secret_sharing.prime` | (large prime) | Modulus for Shamir's Secret Sharing field arithmetic |

### 6. EXTERNAL DEPENDENCIES

**File:** `requirements.txt`

| Package | Purpose |
|---|---|
| `torch` | Tensor operations for model updates and masks |
| `numpy` | Numerical computations |
| `pycryptodome` | Cryptographic primitives (used alongside custom implementations) |
| `pytest` | Test framework |
| `pyyaml` | Configuration file parsing |
| `matplotlib` | Visualization of simulation results |

### 7. KNOWN ISSUES

- The Diffie-Hellman implementation uses custom code rather than a hardened cryptographic library, which is appropriate for research but not production deployment.
- The `CommunicationChannel` in `src/communication/channel.py` is a simulation stub with artificial latency and packet loss; there is no actual network transport.
- The `secret_sharing.py` `split_secret()` function uses Python's `random` module (line 10) instead of `secrets` for coefficient generation; this is cryptographically weak for production but acceptable for protocol demonstration.
- Dropout tolerance is fixed at initialization; the protocol does not support dynamically adjusting the threshold mid-round.
- The `evaluate_polynomial()` function (line 103) uses Horner's method but the polynomial is built with `random.randint(1, prime - 1)` ensuring non-zero coefficients, though this slightly reduces the coefficient space.

### 8. TESTING

**Test directory:** `tests/`

| Test File | Coverage |
|---|---|
| `test_secret_sharing.py` | 7 test cases: basic split/reconstruct, threshold property verification, edge cases (threshold=1, threshold=n), invalid parameters |
| `test_key_agreement.py` | DH keypair generation, shared secret computation, pairwise agreement consistency |
| `test_masked_updates.py` | Mask application/cancellation, mask cancellation verification, mask security properties |
| `test_protocol_integration.py` | Full protocol end-to-end, multiple rounds, varying client counts |
| `test_security_properties.py` | Information-theoretic security of shares, mask secrecy, aggregate correctness |
| `test_dropout_recovery.py` | Recovery under various dropout rates, threshold boundary cases |

**Run:** `cd /home/ubuntu/30Days_Project/05_security_research/secure_aggregation_fl && pytest tests/`

### 9. QUICK MODIFICATION GUIDE

| Change | Where to Edit |
|---|---|
| Change threshold ratio (t/n) | `config/config.yaml` -> `protocol.threshold_ratio` |
| Swap PRF implementation | `src/crypto/prf.py` -> replace `prf()` function (currently HMAC-SHA256) |
| Add new server protocol state | `src/protocol/server.py` -> `ServerState` enum (line 23) and state transition logic |
| Use real network transport | Replace `src/communication/channel.py` -> `CommunicationChannel` with actual socket/gRPC implementation |
| Change DH parameters | `config/config.yaml` -> `dh.generator` and `dh.prime` |
| Adjust dropout simulation | `src/protocol/dropout_recovery.py` -> `simulate_dropouts()` function |
| Add new aggregation strategy | Subclass or modify `src/aggregation/aggregator.py` -> `SecureAggregator` |

### 10. CODE SNIPPETS

**Shamir's Secret Sharing -- Split and Reconstruct** (`src/crypto/secret_sharing.py`, lines 13-100):
```python
def split_secret(
    secret: int,
    threshold: int,
    num_shares: int,
    prime: int
) -> List[Tuple[int, int]]:
    # Validate parameters
    if threshold <= 0:
        raise ValueError("Threshold must be positive")
    if threshold > num_shares:
        raise ValueError("Threshold cannot exceed number of shares")
    if secret >= prime:
        raise ValueError("Secret must be less than prime modulus")

    # Generate random polynomial coefficients
    # f(x) = secret + a_1*x + a_2*x^2 + ... + a_{t-1}*x^{t-1}
    coefficients = [secret] + [random.randint(1, prime - 1) for _ in range(threshold - 1)]

    shares = []
    for x in range(1, num_shares + 1):
        y = evaluate_polynomial(coefficients, x, prime)
        shares.append((x, y))
    return shares


def reconstruct_secret(shares: List[Tuple[int, int]], prime: int) -> int:
    if len(shares) < 2:
        raise ValueError("At least 2 shares required for reconstruction")

    secret = 0
    for i, (x_i, y_i) in enumerate(shares):
        numerator = 1
        denominator = 1
        for j, (x_j, _) in enumerate(shares):
            if i != j:
                numerator *= (0 - x_j)
                denominator *= (x_i - x_j)

        denominator_inv = mod_inverse(denominator % prime, prime)
        lagrange_basis = (numerator * denominator_inv) % prime
        secret = (secret + y_i * lagrange_basis) % prime
    return secret
```

**Server State Machine** (`src/protocol/server.py`, lines 23-31):
```python
class ServerState(Enum):
    IDLE = "idle"
    KEY_AGREEMENT = "key_agreement"
    COLLECTING_UPDATES = "collecting_updates"
    COLLECTING_SHARES = "collecting_shares"
    DROPOUT_RECOVERY = "dropout_recovery"
    COMPUTING_AGGREGATE = "computing_aggregate"
    COMPLETE = "complete"
```

---

## P-24: SignGuard Defense Framework

**Directory:** `/home/ubuntu/30Days_Project/05_security_research/signguard/`
**Day:** 24 | **Score:** 10/10

### 1. PURPOSE

Implements the SignGuard defense framework for Byzantine-resilient federated learning. The system combines four defense layers: (1) ECDSA cryptographic signature verification to authenticate client updates, (2) multi-factor ensemble anomaly detection using L2 norm magnitude, cosine similarity direction, and loss deviation analysis, (3) a decay-based reputation system that tracks client trustworthiness over time, and (4) reputation-weighted aggregation that reduces the influence of suspicious clients. The framework is designed to defend against model poisoning, label flipping, and Byzantine attacks in FL settings.

### 2. ARCHITECTURE

```
signguard/
  signguard/
    core/
      __init__.py
      types.py                       # Dataclasses: ModelUpdate, SignedUpdate, AnomalyScore, etc. (279 lines)
      server.py                      # SignGuardServer orchestrator (349 lines)
      client.py                      # SignGuardClient with ECDSA signing (277 lines)
    crypto/
      __init__.py
      signature.py                   # ECDSA SignatureManager (269 lines)
    detection/
      __init__.py
      base.py                        # AnomalyDetector abstract base
      ensemble.py                    # EnsembleDetector combining 3 detectors (184 lines)
      magnitude_detector.py          # L2NormDetector with MAD thresholding (180 lines)
      direction_detector.py          # CosineSimilarityDetector (188 lines)
      score_detector.py              # LossDeviationDetector
    reputation/
      __init__.py
      decay_reputation.py            # DecayReputationSystem (124 lines)
    aggregation/
      __init__.py
      weighted_aggregator.py         # WeightedAggregator (110 lines)
    utils/
      __init__.py
      metrics.py                     # compute_accuracy helper
      serialization.py               # Model serialization utilities
  experiments/
    config/
      base.yaml                      # Hydra config: 20 clients, 100 rounds, MLP
  tests/
    conftest.py
    test_attacks.py
    test_crypto.py
    test_defenses.py
    test_detection.py
    test_integration.py
    test_visualization.py
  requirements.txt
```

### 3. KEY COMPONENTS

| File | Class / Function | Lines | Role |
|---|---|---|---|
| `signguard/core/types.py` | `ModelUpdate` (dataclass) | 10-71 | Unsigned model update with parameters, metrics, serialization |
| `signguard/core/types.py` | `SignedUpdate` (dataclass) | 75-101 | Wraps ModelUpdate with ECDSA signature and public key |
| `signguard/core/types.py` | `AnomalyScore` (dataclass) | 105-142 | Multi-factor score: magnitude, direction, loss, combined (all in [0,1]) |
| `signguard/core/types.py` | `ReputationInfo` (dataclass) | 146-184 | Client reputation with detection history and rolling average |
| `signguard/core/types.py` | `AggregationResult` (dataclass) | 188-216 | Round result: global model, participating/excluded clients, execution time |
| `signguard/core/types.py` | `ServerConfig` (dataclass) | 240-255 | Server config: rounds, clients per round, thresholds |
| `signguard/core/types.py` | `ExperimentConfig` (dataclass) | 258-278 | Experiment config: seed, dataset, attack/defense type |
| `signguard/core/server.py` | `SignGuardServer` | 23-349 | Orchestrates full pipeline: verify -> detect -> reputation -> aggregate |
| `signguard/core/server.py` | `verify_signatures()` | 68-93 | Batch signature verification, returns verified + rejected lists |
| `signguard/core/server.py` | `detect_anomalies()` | 95-120 | Computes AnomalyScore for each verified update |
| `signguard/core/server.py` | `aggregate()` | 142-249 | Full 5-step pipeline with minimum participant checks |
| `signguard/core/client.py` | `SignGuardClient` | 15-244 | Local training + ECDSA signing of model updates |
| `signguard/core/client.py` | `train()` | 76-155 | Local SGD/Adam training, computes parameter delta |
| `signguard/core/client.py` | `sign_update()` | 157-181 | Signs ModelUpdate with private key via SignatureManager |
| `signguard/crypto/signature.py` | `SignatureManager` | 15-269 | ECDSA with SECP256R1/P-256: generate_keypair, sign_update, verify_update |
| `signguard/detection/ensemble.py` | `EnsembleDetector` | 13-183 | Combines L2NormDetector + CosineSimilarityDetector + LossDeviationDetector |
| `signguard/detection/magnitude_detector.py` | `L2NormDetector` | 12-180 | MAD-based adaptive threshold on update L2 norms |
| `signguard/detection/direction_detector.py` | `CosineSimilarityDetector` | -- | Cosine similarity to robust median reference direction |
| `signguard/reputation/decay_reputation.py` | `DecayReputationSystem` | -- | Reputation update with honesty_bonus=0.1, penalty_factor=0.5 |
| `signguard/aggregation/weighted_aggregator.py` | `WeightedAggregator` | -- | Reputation-weighted FedAvg aggregation |

### 4. DATA FLOW

```
1. LOCAL TRAINING
   Client receives global model parameters
      --> Loads parameters into local model (model.load_state_dict)
      --> Trains for local_epochs on private data
      --> Computes parameter delta: update = trained_params - initial_params
      --> Creates ModelUpdate with client_id, round_num, parameters, num_samples, metrics

2. CRYPTOGRAPHIC SIGNING
   Client signs the model update:
      --> Serializes ModelUpdate to canonical JSON
      --> Hashes with SHA-256
      --> Signs hash with ECDSA private key (SECP256R1 curve)
      --> Creates SignedUpdate = {update, signature, public_key, algorithm="ECDSA"}

3. SERVER PIPELINE (SignGuardServer.aggregate())
   Step 1 - Signature Verification:
      --> For each SignedUpdate, verify ECDSA signature against public key
      --> Reject clients with invalid signatures
      --> Check min_clients_required threshold

   Step 2 - Statistics Update:
      --> Update detector statistics (running norms, directions, losses)
      --> Feed all verified updates to EnsembleDetector.update_statistics()

   Step 3 - Anomaly Detection:
      --> L2NormDetector: compute L2 norm, compare against MAD-based threshold
      --> CosineSimilarityDetector: compare update direction to median reference
      --> LossDeviationDetector: check reported loss against expected range
      --> EnsembleDetector: weighted combination (0.4 * magnitude + 0.4 * direction + 0.2 * loss)
      --> Exclude clients with combined_score > anomaly_threshold (default 0.7)

   Step 4 - Reputation Update:
      --> Low anomaly score -> reputation += honesty_bonus (0.1)
      --> High anomaly score -> reputation *= (1 - penalty_factor) (penalty_factor=0.5)
      --> Reputation decays over inactive rounds

   Step 5 - Weighted Aggregation:
      --> Weight each valid update by: reputation * num_samples
      --> Aggregate: global_model = weighted_average(updates)
      --> Update global model via load_state_dict

4. RESULT
   AggregationResult returned with:
      --> Updated global_model parameters
      --> participating_clients, excluded_clients lists
      --> reputation_updates mapping
      --> Metadata: exclusion_reasons, anomaly_scores breakdown
```

### 5. CONFIGURATION & PARAMETERS

**File:** `experiments/config/base.yaml` (Hydra configuration)

| Parameter | Default | Description |
|---|---|---|
| `server.num_rounds` | 100 | Total FL training rounds |
| `server.num_clients_per_round` | 10 | Clients selected each round |
| `server.min_clients_required` | 5 | Minimum for aggregation |
| `server.anomaly_threshold` | 0.7 | Combined score threshold for exclusion |
| `server.min_reputation_threshold` | 0.1 | Minimum reputation to participate |
| `experiment.num_clients` | 20 | Total clients in the system |
| `experiment.num_byzantine` | 4 | Number of malicious clients (20%) |
| `experiment.attack_type` | `label_flip` | Attack to simulate |
| `experiment.defense_type` | `signguard` | Defense mechanism |
| `experiment.seed` | 42 | Random seed |
| `client.local_epochs` | 5 | Local training epochs |
| `client.learning_rate` | 0.01 | Local learning rate |
| `client.batch_size` | 32 | Local batch size |
| `client.optimizer` | `sgd` | Local optimizer (sgd or adam) |
| `detection.magnitude_weight` | 0.4 | Weight for L2 norm detector |
| `detection.direction_weight` | 0.4 | Weight for cosine similarity detector |
| `detection.loss_weight` | 0.2 | Weight for loss deviation detector |
| `detection.ensemble_method` | `weighted` | Ensemble combination method |

### 6. EXTERNAL DEPENDENCIES

**File:** `requirements.txt`

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Neural network training and model handling |
| `cryptography` | ECDSA signature operations (hazmat.primitives) |
| `numpy` | Numerical computations |
| `pandas` | Data handling |
| `scikit-learn` | Metrics and utility functions |
| `matplotlib`, `seaborn` | Visualization |
| `hydra-core`, `omegaconf` | Configuration management |
| `pytest` | Test framework |

### 7. KNOWN ISSUES

- The `SignGuardClient._create_optimizer()` (line 56-74 in `client.py`) references `self.config.optimizer_momentum` via `hasattr()` but `ClientConfig` does not define an `optimizer_momentum` attribute; falls back to hardcoded 0.9 momentum.
- The `compute_anomaly_score()` in `EnsembleDetector` (line 110-145) calls `compute_score()` internally, causing the individual detector scores to be computed twice (once directly, once via the combined score calculation).
- The `L2NormDetector` uses `scipy.stats` (imported at line 5 of `magnitude_detector.py`) but `scipy` is not listed in `requirements.txt`.
- The `verify_update` method in `SignatureManager` is called on line 87 of `server.py` but the actual verification logic depends on having registered public keys; the registration mechanism is implicit.
- The `AnomalyScore.__post_init__` validator (line 120-131 in `types.py`) raises `ValueError` for out-of-range scores, but individual detectors could produce scores outside [0,1] before clamping.

### 8. TESTING

**Test directory:** `tests/`

| Test File | Coverage |
|---|---|
| `test_crypto.py` | ECDSA keypair generation, signing, verification, invalid signature rejection |
| `test_detection.py` | Individual detector scores, ensemble combination, threshold behavior |
| `test_attacks.py` | Attack simulation: label flipping, model poisoning, Byzantine updates |
| `test_defenses.py` | Defense effectiveness against various attack types |
| `test_integration.py` | Full pipeline: client training -> signing -> server aggregation |
| `test_visualization.py` | Visualization output generation |
| `conftest.py` | Shared fixtures: model, data, client/server setup |

**Run:** `cd /home/ubuntu/30Days_Project/05_security_research/signguard && pytest tests/`

### 9. QUICK MODIFICATION GUIDE

| Change | Where to Edit |
|---|---|
| Change anomaly threshold | `signguard/core/types.py` -> `ServerConfig.anomaly_threshold` (line 254) |
| Add new detector to ensemble | `signguard/detection/ensemble.py` -> add detector in `__init__` and update `compute_score()` |
| Change signature curve | `signguard/crypto/signature.py` -> `SignatureManager.__init__` curve parameter (line 22) |
| Adjust reputation decay | `signguard/reputation/decay_reputation.py` -> `honesty_bonus`, `penalty_factor` |
| Modify ensemble weights | `signguard/detection/ensemble.py` -> `__init__` weights (lines 21-26) |
| Change aggregation strategy | `signguard/aggregation/weighted_aggregator.py` -> `WeightedAggregator.aggregate()` |
| Add new attack type | Implement attack client, register in `experiments/config/base.yaml` -> `experiment.attack_type` |

### 10. CODE SNIPPETS

**ECDSA Signature Manager** (`signguard/crypto/signature.py`, lines 15-39):
```python
class SignatureManager:
    """Manages ECDSA signatures for model updates."""

    def __init__(self, curve: ec.EllipticCurve = ec.SECP256R1()):
        self.curve = curve
        self.backend = default_backend()

    def generate_keypair(self) -> Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]:
        private_key = ec.generate_private_key(self.curve, self.backend)
        public_key = private_key.public_key()
        return private_key, public_key

    def sign_update(
        self,
        update: ModelUpdate,
        private_key: ec.EllipticCurvePrivateKey,
    ) -> str:
        # Serializes ModelUpdate to canonical JSON, hashes with SHA-256,
        # signs with ECDSA private key
        ...
```

**Server Aggregation Pipeline** (`signguard/core/server.py`, lines 142-249):
```python
def aggregate(self, signed_updates: List[SignedUpdate]) -> AggregationResult:
    start_time = time.time()
    self.excluded_clients = set()
    self.exclusion_reasons = {}

    # Step 1: Verify signatures
    verified_updates, signature_rejected = self.verify_signatures(signed_updates)
    for client_id in signature_rejected:
        self.excluded_clients.add(client_id)

    # Step 2: Update detector statistics
    self.detector.update_statistics(
        [u.update for u in verified_updates], self.global_params)

    # Step 3: Detect anomalies
    anomaly_scores = self.detect_anomalies(verified_updates)

    # Filter by anomaly threshold
    valid_updates = []
    for signed_update in verified_updates:
        client_id = signed_update.update.client_id
        anomaly_score = anomaly_scores[client_id]
        if self.detector.is_anomalous(anomaly_score):
            self.excluded_clients.add(client_id)
        else:
            valid_updates.append(signed_update)

    # Step 4: Update reputations
    self.update_reputations(anomaly_scores, is_verified)

    # Step 5: Aggregate with reputation-weighted averaging
    result = self.aggregator.aggregate(valid_updates, reputations, self.global_params)
    self.update_global_model(result)
    self.current_round += 1
    return result
```

---

## P-25: Membership Inference Attack

**Directory:** `/home/ubuntu/30Days_Project/05_security_research/membership_inference_attack/`
**Day:** 25 | **Score:** 10/10

### 1. PURPOSE

Implements a comprehensive membership inference attack framework for federated learning, based on Shokri et al. (S&P 2017). The system determines whether a specific data point was part of a model's training set by analyzing model outputs. Three attack families are implemented: (1) shadow model attacks that train surrogate models to generate attack training data, (2) threshold-based attacks using loss/confidence calibration, and (3) metric-based attacks (loss, entropy, modified entropy, prediction variance). FL-specific extensions include temporal cross-round attacks and client data inference attacks.

### 2. ARCHITECTURE

```
membership_inference_attack/
  config/
    attack_config.yaml             # Attack parameters and thresholds
  src/
    attacks/
      __init__.py
      shadow_models.py             # Shadow model training framework (452 lines)
      threshold_attack.py          # Confidence/threshold attacks (252 lines)
      metric_attacks.py            # Loss/entropy/variance attacks (345 lines)
      attack_aggregator.py         # FL-specific attack orchestrator (449 lines)
    target_models/
      fl_target.py                 # FraudDetectionNN target model
    utils/
      data_splits.py               # AttackDataGenerator for member/non-member splits
  tests/
    test_attacks.py
    test_data_separation.py
    test_shadow_models.py
  requirements.txt
```

### 3. KEY COMPONENTS

| File | Class / Function | Lines | Role |
|---|---|---|---|
| `src/attacks/shadow_models.py` | `ShadowModelTrainer` | 41-452 | Trains K shadow models to generate attack training data |
| `src/attacks/shadow_models.py` | `AttackModel` | -- | Wraps sklearn classifiers (RF, MLP, LogisticRegression) as attack models |
| `src/attacks/shadow_models.py` | `generate_attack_training_data()` | -- | Generates member/non-member prediction vectors for attack model training |
| `src/attacks/shadow_models.py` | `shadow_model_attack()` | -- | End-to-end shadow model attack pipeline |
| `src/attacks/threshold_attack.py` | `confidence_based_attack()` | -- | Classifies based on prediction confidence (max softmax) |
| `src/attacks/threshold_attack.py` | `threshold_based_attack()` | -- | Generic threshold attack on arbitrary metric |
| `src/attacks/threshold_attack.py` | `calibrate_threshold()` | -- | Automated threshold tuning using validation data |
| `src/attacks/threshold_attack.py` | `find_optimal_threshold()` | -- | Uses Youden's J statistic (TPR - FPR) for optimal threshold |
| `src/attacks/metric_attacks.py` | `loss_based_attack()` | -- | Members have lower loss; threshold on cross-entropy |
| `src/attacks/metric_attacks.py` | `entropy_based_attack()` | -- | Members produce lower prediction entropy |
| `src/attacks/metric_attacks.py` | `modified_entropy_attack()` | -- | Enhanced entropy using true labels for conditioning |
| `src/attacks/metric_attacks.py` | `prediction_variance_attack()` | -- | Members have lower prediction variance across augmentations |
| `src/attacks/metric_attacks.py` | `aggregate_metric_attacks()` | -- | Combines all metric-based attacks for ensemble result |
| `src/attacks/attack_aggregator.py` | `attack_global_model()` | -- | Attack federated global model at convergence |
| `src/attacks/attack_aggregator.py` | `attack_local_models()` | -- | Attack individual client local models |
| `src/attacks/attack_aggregator.py` | `temporal_attack()` | -- | Cross-round attack exploiting temporal model changes |
| `src/attacks/attack_aggregator.py` | `cross_round_attack()` | -- | Tracks membership signals across training rounds |
| `src/attacks/attack_aggregator.py` | `client_data_inference_attack()` | -- | Infers which client trained on a particular data point |
| `src/attacks/attack_aggregator.py` | `aggregate_fl_attacks()` | -- | Combines all FL-specific attack results |

### 4. DATA FLOW

```
1. SHADOW MODEL TRAINING (shadow_models.py)
   Attacker has auxiliary dataset (similar distribution to target)
      --> Split into K subsets (member/non-member for each shadow model)
      --> Train K shadow models on their respective member sets
      --> For each shadow model:
            Query with member data -> label as "member" (label=1)
            Query with non-member data -> label as "non-member" (label=0)
      --> Collect (prediction_vector, membership_label) pairs
      --> Train attack model (RF/MLP/LogisticRegression) on these pairs

2. THRESHOLD ATTACK (threshold_attack.py)
   For target data point:
      --> Query target model -> get prediction vector
      --> Compute metric (max confidence, loss, entropy)
      --> Compare against calibrated threshold
      --> Classify as member if metric < threshold (for loss/entropy)
                    or member if metric > threshold (for confidence)
   Threshold calibration:
      --> Use validation set with known membership
      --> find_optimal_threshold() maximizes Youden's J = TPR - FPR

3. METRIC ATTACKS (metric_attacks.py)
   For target data point with true label:
      loss_based:     Compute CE loss -> low loss = member
      entropy_based:  Compute -sum(p * log(p)) -> low entropy = member
      modified_entropy: Condition on true label -> more precise
      variance_based: Augment input K times -> low variance = member
   Aggregate: weighted combination of individual attack decisions

4. FL-SPECIFIC ATTACKS (attack_aggregator.py)
   Global model attack:  Attack the final aggregated model
   Local model attack:   Attack individual client models (if accessible)
   Temporal attack:      Track how loss/confidence changes across rounds
                        Rapid change correlates with membership
   Cross-round attack:   Combine signals from multiple rounds for higher confidence
   Client inference:     Determine which specific client trained on a data point
```

### 5. CONFIGURATION & PARAMETERS

**File:** `config/attack_config.yaml`

| Parameter | Default | Description |
|---|---|---|
| `shadow.n_shadow_models` | 10 | Number of shadow models to train |
| `shadow.shadow_epochs` | 50 | Training epochs per shadow model |
| `shadow.attack_model_type` | `random_forest` | Attack classifier type |
| `threshold.target_fpr` | 0.05 | Target false positive rate for threshold calibration |
| `metric.loss_weight` | 0.3 | Weight for loss-based metric in ensemble |
| `metric.entropy_weight` | 0.3 | Weight for entropy-based metric |
| `metric.variance_weight` | 0.2 | Weight for variance-based metric |
| `metric.modified_entropy_weight` | 0.2 | Weight for modified entropy metric |
| `validation.strict_data_separation` | true | Ensure no data overlap between shadow and target |

### 6. EXTERNAL DEPENDENCIES

**File:** `requirements.txt`

| Package | Purpose |
|---|---|
| `torch` | Neural network models and training |
| `numpy` | Numerical computations |
| `scikit-learn` | Attack classifiers (RF, MLP, LogisticRegression), metrics |
| `matplotlib`, `seaborn` | ROC curves, attack performance visualization |
| `pyyaml` | Configuration parsing |
| `pytest` | Testing |
| `pandas` | Results tabulation |

### 7. KNOWN ISSUES

- `shadow_models.py` uses `sys.path.append('src/utils')` and `sys.path.append('src/target_models')` (lines 33-34), which is fragile and depends on working directory.
- The `shadow_model_attack()` function trains shadow models from scratch each time; there is no caching or checkpointing of trained shadow models.
- The `prediction_variance_attack()` in `metric_attacks.py` requires implementing data augmentation externally; the augmentation strategy is not parameterized.
- The `client_data_inference_attack()` in `attack_aggregator.py` assumes access to individual client local models, which may not be realistic in standard FL deployments.
- There is no formal privacy budget tracking or differential privacy integration to measure the attack's effectiveness against DP-protected models.

### 8. TESTING

**Test directory:** `tests/`

| Test File | Coverage |
|---|---|
| `test_shadow_models.py` | Shadow model training, attack data generation, attack model accuracy |
| `test_attacks.py` | All attack types: threshold, metric, FL-specific attacks |
| `test_data_separation.py` | Validates strict member/non-member data separation |

**Run:** `cd /home/ubuntu/30Days_Project/05_security_research/membership_inference_attack && pytest tests/`

### 9. QUICK MODIFICATION GUIDE

| Change | Where to Edit |
|---|---|
| Change number of shadow models | `config/attack_config.yaml` -> `shadow.n_shadow_models` |
| Add new attack classifier | `src/attacks/shadow_models.py` -> `AttackModel` class, add sklearn model |
| Change threshold calibration strategy | `src/attacks/threshold_attack.py` -> `find_optimal_threshold()` (replace Youden's J) |
| Add new metric-based attack | `src/attacks/metric_attacks.py` -> add function, register in `aggregate_metric_attacks()` |
| Adjust metric ensemble weights | `config/attack_config.yaml` -> `metric.*_weight` parameters |
| Add new FL-specific attack | `src/attacks/attack_aggregator.py` -> add method, register in `aggregate_fl_attacks()` |

### 10. CODE SNIPPETS

**Shadow Model Training** (`src/attacks/shadow_models.py`, lines 41-50):
```python
class ShadowModelTrainer:
    """Trains multiple shadow models to generate attack training data."""

    def __init__(
        self,
        model_architecture: nn.Module,
        n_shadow: int = 10,
        shadow_epochs: int = 50,
        ...
    ):
```

**Optimal Threshold via Youden's J** (`src/attacks/threshold_attack.py`):
```python
def find_optimal_threshold():
    """
    Find threshold maximizing Youden's J statistic:
    J = TPR - FPR (sensitivity + specificity - 1)
    """
    # For each candidate threshold:
    #   Compute TPR (true positive rate) and FPR (false positive rate)
    #   Select threshold that maximizes J = TPR - FPR
```

---

## P-26: Gradient Leakage Attack (DLG)

**Directory:** `/home/ubuntu/30Days_Project/05_security_research/gradient_leakage_attack/`
**Day:** 26 | **Score:** 10/10

### 1. PURPOSE

Implements the Deep Leakage from Gradients (DLG) attack framework based on Zhu et al. (NeurIPS 2019). The attack reconstructs private training data (images and labels) by optimizing dummy inputs to match observed gradients. The framework provides multiple optimizer variants (L-BFGS, Adam, cosine similarity), multi-restart strategies, and comprehensive defense evaluations including differential privacy noise injection and gradient compression (top-k sparsification, random masking, quantization, sign compression). Designed for researching privacy vulnerabilities in federated learning gradient exchange.

### 2. ARCHITECTURE

```
gradient_leakage_attack/
  config/
    attack_config.yaml             # Attack and defense parameters
  src/
    attacks/
      __init__.py
      base_attack.py               # GradientLeakageAttack ABC, ReconstructionResult (341 lines)
      dlg.py                       # DLGAttack with L-BFGS (298 lines)
      dlg_adam.py                   # DLG with Adam optimizer
      dlg_cosine.py                # Cosine similarity variant
    defenses/
      __init__.py
      dp_noise.py                  # DPDefense + AdaptiveDPDefense (282 lines)
      gradient_compression.py      # 4 compression methods + error feedback (422 lines)
    models/
      __init__.py
      simple_cnn.py                # SimpleCNN, LeNet5, get_model() factory (197 lines)
  tests/
    __init__.py
    test_dlg_basic.py
    test_gradient_matching.py
    test_defenses.py
  run_dlg_attack.py                # Standalone demo script (142 lines)
  requirements.txt
```

### 3. KEY COMPONENTS

| File | Class / Function | Lines | Role |
|---|---|---|---|
| `src/attacks/base_attack.py` | `ReconstructionResult` (dataclass) | 15-24 | Holds reconstructed_x, reconstructed_y, gradient_distances, success flag |
| `src/attacks/base_attack.py` | `GradientLeakageAttack` (ABC) | 27-341 | Abstract base: initialize_dummy_data, compute_gradient_distance (MSE/cosine/L1) |
| `src/attacks/dlg.py` | `DLGAttack` | 13-298 | L-BFGS-based DLG attack with multi-restart support |
| `src/attacks/dlg.py` | `reconstruct()` | 37-298 | Core reconstruction: optimizes dummy data to match true gradients |
| `src/attacks/dlg.py` | `dlg_lbfgs()` | -- | Single L-BFGS reconstruction attempt |
| `src/attacks/dlg.py` | `dlg_with_multiple_restarts()` | -- | Multiple random initializations, returns best reconstruction |
| `src/defenses/dp_noise.py` | `DPDefense` | -- | Adds calibrated Gaussian/Laplace noise to gradients |
| `src/defenses/dp_noise.py` | `AdaptiveDPDefense` | -- | Adapts noise level based on gradient sensitivity |
| `src/defenses/dp_noise.py` | `compute_epsilon()` | -- | Computes privacy budget epsilon from noise parameters |
| `src/defenses/gradient_compression.py` | `GradientCompression` | -- | Four methods: topk, random, quantization, sign |
| `src/defenses/gradient_compression.py` | `ErrorFeedbackCompensation` | -- | Accumulates compression error for next round |
| `src/defenses/gradient_compression.py` | `SparsifiedGradientDefense` | -- | Combined sparsification defense with configurable ratio |
| `src/models/simple_cnn.py` | `SimpleCNN` | -- | 2-conv + 1-FC model for MNIST |
| `src/models/simple_cnn.py` | `LeNet5` | -- | Classic LeNet-5 architecture |
| `src/models/simple_cnn.py` | `get_model()` | -- | Factory function for model selection |
| `run_dlg_attack.py` | (script) | 1-142 | Standalone demo: creates model, captures gradients, prints attack overview |

### 4. DATA FLOW

```
1. GRADIENT CAPTURE
   Target client trains on private data (x_true, y_true)
      --> Forward pass: output = model(x_true)
      --> Loss: L = CrossEntropyLoss(output, y_true)
      --> Compute gradient: g_true = dL/d(params) for each model parameter
      --> Gradient is shared with server (or intercepted by attacker)

2. ATTACK INITIALIZATION
   Attacker:
      --> Creates copy of target model (same architecture, same weights)
      --> Initializes dummy data: x_dummy ~ N(0,1) or U(0,1) (requires_grad=True)
      --> Initializes dummy label: y_dummy (tries all classes or optimizes)

3. DLG OPTIMIZATION (L-BFGS variant)
   For each iteration:
      --> Forward pass: output_dummy = model(x_dummy)
      --> Loss: L_dummy = CrossEntropyLoss(output_dummy, y_dummy)
      --> Compute gradient: g_dummy = dL_dummy/d(params)
      --> Gradient matching loss = distance(g_dummy, g_true)
            MSE:    sum((g_dummy - g_true)^2)
            Cosine: 1 - cos_sim(g_dummy, g_true)
            L1:     sum(|g_dummy - g_true|)
      --> L-BFGS updates x_dummy to minimize gradient matching loss
      --> Track convergence: gradient_distances list

4. MULTI-RESTART (dlg_with_multiple_restarts)
   Run K independent attacks with different random initializations
      --> Select reconstruction with lowest final gradient matching loss
      --> Return best ReconstructionResult

5. DEFENSE EVALUATION
   DP Defense:
      g_noisy = g_true + N(0, sigma^2 * I)    (Gaussian)
      g_noisy = g_true + Laplace(0, b)          (Laplace)
      --> Attack runs on g_noisy instead of g_true
      --> Measure reconstruction quality degradation

   Compression Defense:
      TopK:         Keep only K largest gradient elements
      Random:       Randomly mask gradient elements
      Quantization: Reduce gradient precision (e.g., 8-bit)
      Sign:         Keep only sign of gradients (SignSGD)
      --> Attack runs on compressed gradient
      --> Error feedback compensates compression loss for training quality
```

### 5. CONFIGURATION & PARAMETERS

**File:** `config/attack_config.yaml`

| Parameter | Default | Description |
|---|---|---|
| `attack.optimizer` | `lbfgs` | Optimization method (lbfgs, adam, cosine) |
| `attack.num_iterations` | 1000 | Optimization iterations per restart |
| `attack.num_restarts` | 10 | Number of random restarts |
| `attack.distance_metric` | `mse` | Gradient distance metric (mse, cosine, l1) |
| `attack.init_method` | `uniform` | Dummy data initialization (uniform, normal) |
| `attack.learning_rate` | 1.0 | L-BFGS learning rate |
| `defense.dp.noise_type` | `gaussian` | DP noise distribution |
| `defense.dp.sigma` | 0.01 | Noise standard deviation |
| `defense.dp.clip_norm` | 1.0 | Gradient clipping bound |
| `defense.compression.method` | `topk` | Compression method |
| `defense.compression.ratio` | 0.1 | Fraction of gradients retained (top-k) |

### 6. EXTERNAL DEPENDENCIES

**File:** `requirements.txt`

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Model training, gradient computation, image transforms |
| `numpy` | Array operations |
| `matplotlib`, `seaborn` | Reconstructed image visualization |
| `Pillow` | Image loading and processing |
| `scikit-image` | SSIM and image quality metrics |
| `scikit-learn` | Evaluation metrics |
| `pandas` | Results tabulation |
| `pyyaml` | Configuration |
| `tqdm` | Progress bars |
| `tensorboard` | Training visualization |

### 7. KNOWN ISSUES

- The `run_dlg_attack.py` demo script (lines 66-88) has a flawed closure function: it computes `grad_x` (gradient w.r.t. input) but compares it against `true_gradient` (gradient w.r.t. model parameters), which are tensors of different shapes. The actual attack in `src/attacks/dlg.py` correctly compares parameter gradients.
- The `run_dlg_attack.py` script never actually calls `dlg_attack()` (the function defined at line 35); it only prints the attack overview. The full attack runs via `src/attacks/dlg.py`.
- The `SimpleCNN` in `run_dlg_attack.py` (line 21) uses `nn.Linear(32 * 28 * 28, 10)` without any pooling layers, creating a very large FC layer (25,088 parameters). The `src/models/simple_cnn.py` version is more realistic.
- L-BFGS optimization can be memory-intensive for large models; the framework does not include memory management or batch gradient matching.
- The `ErrorFeedbackCompensation` class tracks compression error across rounds but the error buffer is never reset, which could accumulate floating-point drift over many rounds.

### 8. TESTING

**Test directory:** `tests/`

| Test File | Coverage |
|---|---|
| `test_dlg_basic.py` | Basic DLG attack on small models, reconstruction quality metrics |
| `test_gradient_matching.py` | Gradient distance computations (MSE, cosine, L1), convergence properties |
| `test_defenses.py` | DP noise effectiveness, compression defense quality, attack degradation |

**Run:** `cd /home/ubuntu/30Days_Project/05_security_research/gradient_leakage_attack && pytest tests/`

### 9. QUICK MODIFICATION GUIDE

| Change | Where to Edit |
|---|---|
| Add new optimizer variant | Create new file `src/attacks/dlg_<name>.py`, subclass `GradientLeakageAttack` |
| Change gradient distance metric | `src/attacks/base_attack.py` -> `compute_gradient_distance()` method |
| Adjust DP noise level | `config/attack_config.yaml` -> `defense.dp.sigma` |
| Add new compression method | `src/defenses/gradient_compression.py` -> `GradientCompression` class, add method |
| Change target model | `src/models/simple_cnn.py` -> add architecture, update `get_model()` factory |
| Modify multi-restart strategy | `src/attacks/dlg.py` -> `dlg_with_multiple_restarts()` |
| Tune L-BFGS parameters | `config/attack_config.yaml` -> `attack.learning_rate`, `attack.num_iterations` |

### 10. CODE SNIPPETS

**DLG Attack Base Class** (`src/attacks/base_attack.py`, lines 15-50):
```python
@dataclass
class ReconstructionResult:
    """Result of gradient leakage attack."""
    reconstructed_x: torch.Tensor
    reconstructed_y: torch.Tensor
    final_matching_loss: float
    gradient_distances: List[float]
    convergence_iterations: int
    success: bool
    metadata: Optional[dict] = None


class GradientLeakageAttack(ABC):
    """Abstract base class for gradient leakage attacks."""

    def __init__(
        self,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        device: torch.device = torch.device('cpu')
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion or nn.CrossEntropyLoss()
```

**DLG L-BFGS Reconstruction** (`src/attacks/dlg.py`, lines 37-50):
```python
def reconstruct(
    self,
    true_gradients: Dict[str, torch.Tensor],
    input_shape: Tuple[int, ...],
    num_classes: int,
    num_iterations: int = 1000,
    init_method: str = 'uniform',
    distance_metric: str = 'mse',
    verbose: bool = True,
    **kwargs
) -> ReconstructionResult:
    """Reconstruct data using DLG with L-BFGS."""
```

---

## P-27: Property Inference Attack

**Directory:** `/home/ubuntu/30Days_Project/05_security_research/property_inference_attack/`
**Day:** 27 | **Score:** 10/10

### 1. PURPOSE

Implements a property inference attack framework for federated learning that infers aggregate statistical properties of clients' private training data from observed model updates. Unlike membership inference (which targets individual records), this attack targets dataset-level properties such as fraud rate, data volume, and feature distributions. The attack uses a meta-classifier approach: training a secondary model (Random Forest, Ridge, SVR, or MLP) on synthetic federated learning runs to learn the mapping from model updates to dataset properties, then applying this learned mapping to real observed updates.

### 2. ARCHITECTURE

```
property_inference_attack/
  config/
    attack_config.yaml             # Attack parameters, property ranges, FL config
  src/
    attacks/
      __init__.py
      property_inference.py        # PropertyInferenceAttack orchestrator (423 lines)
      meta_classifier.py           # PropertyMetaClassifier sklearn wrapper (463 lines)
      property_extractor.py        # Extract features from model updates
    fl_system/
      __init__.py
      server.py                    # FederatedServer + MaliciousServer (400 lines)
      client.py                    # FederatedClient for local training
    data_generation/
      __init__.py
      synthetic_generator.py       # Generate synthetic fraud datasets
      property_varier.py           # PropertyVarier for controlled experiments
    metrics/
      __init__.py
      attack_metrics.py            # Regression metrics, rank correlation, bootstrap CI
  tests/
    test_meta_classifier.py
    test_property_extraction.py
  requirements.txt
```

### 3. KEY COMPONENTS

| File | Class / Function | Lines | Role |
|---|---|---|---|
| `src/attacks/property_inference.py` | `PropertyInferenceAttack` | 27-423 | Main attack orchestrator: trains meta-classifier, executes inference |
| `src/attacks/property_inference.py` | `FraudRateInferenceAttack` | -- | Specialized attack for inferring fraud rate from model updates |
| `src/attacks/property_inference.py` | `DataVolumeInferenceAttack` | -- | Specialized attack for inferring dataset size |
| `src/attacks/property_inference.py` | `execute_attack_on_fl_system()` | -- | Runs attack against a live FL system |
| `src/attacks/property_inference.py` | `analyze_property_leakage()` | -- | Quantifies information leakage across properties |
| `src/attacks/meta_classifier.py` | `PropertyMetaClassifier` | 20-463 | Wraps sklearn models for property prediction from update features |
| `src/attacks/meta_classifier.py` | `MultiOutputMetaClassifier` | -- | Predicts multiple properties simultaneously |
| `src/fl_system/server.py` | `FederatedServer` | -- | Standard FedAvg server for simulation |
| `src/fl_system/server.py` | `MaliciousServer` | -- | Server subclass that collects model updates for attack data generation |
| `src/data_generation/synthetic_generator.py` | `generate_fraud_dataset()` | -- | Creates synthetic fraud detection datasets with configurable properties |
| `src/data_generation/property_varier.py` | `PropertyVarier` | -- | Systematically varies dataset properties for meta-training |
| `src/metrics/attack_metrics.py` | `compute_regression_metrics()` | -- | MAE, RMSE, R-squared for attack evaluation |
| `src/metrics/attack_metrics.py` | `compute_rank_correlation()` | -- | Spearman rank correlation between true and predicted properties |
| `src/metrics/attack_metrics.py` | `bootstrap_confidence_interval()` | -- | Bootstrap CI for attack performance estimates |

### 4. DATA FLOW

```
1. META-TRAINING DATA GENERATION
   PropertyVarier creates N synthetic datasets with varied properties:
      Dataset_1: fraud_rate=0.01, data_volume=1000, feature_dist=A
      Dataset_2: fraud_rate=0.05, data_volume=2000, feature_dist=B
      ...
      Dataset_N: fraud_rate=0.20, data_volume=5000, feature_dist=C

   For each synthetic dataset:
      --> Create federated learning setup (MaliciousServer + clients)
      --> Run FL training for several rounds
      --> MaliciousServer collects model updates from each client
      --> Extract features from model updates (norms, directions, statistics)
      --> Record (update_features, true_property_value) pair

2. META-CLASSIFIER TRAINING
   PropertyMetaClassifier receives:
      --> X: matrix of update features (N x F)
      --> y: vector of true property values (N x 1)
   Training:
      --> StandardScaler normalizes features (if normalize=True)
      --> Trains sklearn model (RF/Ridge/SVR/MLP) via cross-validation
      --> Evaluates with MAE, RMSE, R-squared, Spearman correlation

3. ATTACK EXECUTION
   Against real FL system:
      --> Observe model updates from target client(s)
      --> Extract same features as meta-training
      --> Feed through trained meta-classifier
      --> Output: predicted property values for target client's data

4. EVALUATION
   compare_to_baseline():     Compare attack to random guessing
   bootstrap_confidence_interval():  Statistical confidence in predictions
   analyze_property_leakage():       Quantify which properties leak most
```

### 5. CONFIGURATION & PARAMETERS

**File:** `config/attack_config.yaml`

| Parameter | Default | Description |
|---|---|---|
| `attack.target_property` | `fraud_rate` | Property to infer |
| `attack.scenario` | `server` | Attacker scenario (server, client, external) |
| `attack.meta_classifier_type` | `rf_regressor` | Meta-classifier model type |
| `attack.n_synthetic_datasets` | 500 | Number of synthetic FL runs for meta-training |
| `property_ranges.fraud_rate` | `[0.001, 0.5]` | Range of fraud rates to simulate |
| `property_ranges.data_volume` | `[100, 10000]` | Range of dataset sizes |
| `fl.num_clients` | 10 | Number of FL clients in simulation |
| `fl.num_rounds` | 20 | FL training rounds per simulation |
| `meta_classifier.normalize` | true | Whether to normalize update features |
| `meta_classifier.cv_folds` | 5 | Cross-validation folds |

### 6. EXTERNAL DEPENDENCIES

**File:** `requirements.txt`

| Package | Purpose |
|---|---|
| `torch` | FL model training and gradient computation |
| `scikit-learn` | Meta-classifiers (RF, Ridge, SVR, MLP), metrics, cross-validation |
| `numpy` | Numerical operations |
| `pandas` | Results analysis and tabulation |
| `matplotlib`, `seaborn` | Attack performance visualization |
| `scipy` | Statistical tests (Spearman correlation) |
| `pyyaml` | Configuration |
| `pytest` | Testing |

### 7. KNOWN ISSUES

- The `PropertyMetaClassifier` (line 20 of `meta_classifier.py`) supports 8 model types but only regression types (rf_regressor, ridge, svr, mlp_regressor) are meaningful for continuous properties like fraud_rate; classification types are included but fraud rate is inherently continuous.
- The `MaliciousServer` in `src/fl_system/server.py` extends `FederatedServer` to collect updates, but the collection is done in plaintext without considering secure aggregation; if secure aggregation is deployed, the server cannot observe individual updates.
- The `generate_fraud_dataset()` in `synthetic_generator.py` creates synthetic data that may not capture the distributional complexity of real fraud data, potentially weakening the meta-classifier's transferability.
- The meta-training phase requires running 500 complete FL simulations (default `n_synthetic_datasets`), which is computationally expensive.
- No defense evaluation is included (e.g., how DP noise or gradient compression affects property inference accuracy).

### 8. TESTING

**Test directory:** `tests/`

| Test File | Coverage |
|---|---|
| `test_meta_classifier.py` | Meta-classifier training, prediction, cross-validation, multiple model types |
| `test_property_extraction.py` | Feature extraction from model updates, feature consistency |

**Run:** `cd /home/ubuntu/30Days_Project/05_security_research/property_inference_attack && pytest tests/`

### 9. QUICK MODIFICATION GUIDE

| Change | Where to Edit |
|---|---|
| Target a different property | `config/attack_config.yaml` -> `attack.target_property` |
| Change meta-classifier model | `config/attack_config.yaml` -> `attack.meta_classifier_type` (rf_regressor, ridge, svr, mlp_regressor) |
| Add new property to infer | `src/attacks/property_extractor.py` -> add extraction, register in `extract_all_properties()` |
| Change synthetic data distribution | `src/data_generation/synthetic_generator.py` -> `generate_fraud_dataset()` |
| Adjust property variation ranges | `config/attack_config.yaml` -> `property_ranges.*` |
| Add defense evaluation | Add defense wrapper around gradient exchange in `src/fl_system/server.py` |
| Add new meta-classifier type | `src/attacks/meta_classifier.py` -> `PropertyMetaClassifier.__init__` model creation |

### 10. CODE SNIPPETS

**PropertyInferenceAttack Orchestrator** (`src/attacks/property_inference.py`, lines 27-49):
```python
class PropertyInferenceAttack:
    """Property inference attack on federated learning.

    This attack learns to infer dataset properties (fraud rate, data volume,
    feature distributions) from observed model updates.

    Example:
        >>> attack = PropertyInferenceAttack(
        ...     target_property='fraud_rate',
        ...     scenario='server'
        ... )
        >>> attack.train_meta_classifier(n_datasets=500)
        >>> predicted = attack.execute_attack(observed_updates)
    """

    def __init__(
        self,
        target_property: str,
        scenario: str = 'server',
        meta_classifier_type: str = 'rf_regressor',
        config_path: Optional[str] = None
    ):
```

**PropertyMetaClassifier** (`src/attacks/meta_classifier.py`, lines 20-50):
```python
class PropertyMetaClassifier:
    """Meta-classifier for inferring dataset properties from model updates."""

    def __init__(
        self,
        property_name: str,
        model_type: str = 'rf_regressor',
        model_params: Optional[Dict[str, Any]] = None,
        normalize: bool = True
    ):
        # Supported model types:
        # Regression: 'rf_regressor', 'ridge', 'svr', 'mlp_regressor'
        # Classification: 'rf_classifier', 'logistic', 'svc', 'mlp_classifier'
```

---

## P-28: Privacy-Preserving FL for Fraud Detection

**Directory:** `/home/ubuntu/30Days_Project/05_security_research/privacy_preserving_fl_fraud/`
**Day:** 28 | **Score:** 10/10

### 1. PURPOSE

Implements a production-grade privacy-preserving federated learning system for financial fraud detection. The system combines DP-SGD (Differentially Private Stochastic Gradient Descent) with the Flower federated learning framework, supporting multiple model architectures (LSTM, Transformer, XGBoost), MLflow experiment tracking, FastAPI model serving, and Redis-based communication. The privacy mechanism includes per-sample gradient clipping, calibrated Gaussian noise addition, and RDP-based (Renyi Differential Privacy) privacy accounting via Opacus to track cumulative privacy budget (epsilon, delta) across training rounds.

### 2. ARCHITECTURE

```
privacy_preserving_fl_fraud/
  config/
    base_config.yaml               # Comprehensive Hydra config (all subsystems)
  src/
    privacy/
      __init__.py
      differential_privacy.py      # DP-SGD, PrivacyAccountant, clip_and_add_noise (660 lines)
    fl/
      __init__.py
      server.py                    # FlowerServer + SimulationServer (540 lines)
      client.py                    # FlowerClient for local DP training
      strategy.py                  # Custom FL strategies with DP integration
    models/
      __init__.py
      lstm.py                      # LSTM fraud detection model
      transformer.py               # Transformer fraud detection model
      xgboost_model.py             # Federated XGBoost
    data/
      __init__.py
      preprocessor.py              # Data preprocessing pipeline
      partitioner.py               # Non-IID data partitioning
    monitoring/
      __init__.py
      mlflow_tracker.py            # MLflow experiment tracking
    serving/
      __init__.py
      api.py                       # FastAPI model serving
    utils/
      __init__.py                  # get_device, get_fl_logger
  tests/
    __init__.py
    test_data.py
    test_fl.py
    test_privacy.py
    test_security.py
    test_serving.py
  requirements.txt
```

### 3. KEY COMPONENTS

| File | Class / Function | Lines | Role |
|---|---|---|---|
| `src/privacy/differential_privacy.py` | `PrivacySpent` (dataclass) | 19-24 | Tracks epsilon, delta, round number |
| `src/privacy/differential_privacy.py` | `clip_and_add_noise()` | 27-50+ | Core DP-SGD: per-sample gradient clipping + Gaussian noise |
| `src/privacy/differential_privacy.py` | `DPSGDOptimizer` | -- | Wraps any PyTorch optimizer with DP gradient processing |
| `src/privacy/differential_privacy.py` | `DPSGDFactory` | -- | Factory for creating DP-SGD optimizers with configured privacy parameters |
| `src/privacy/differential_privacy.py` | `PrivacyAccountant` | -- | RDP-based accounting via Opacus; tracks cumulative epsilon across rounds |
| `src/fl/server.py` | `FlowerServer` | 21-60+ | Flower-based server with strategy, MLflow integration |
| `src/fl/server.py` | `SimulationServer` | -- | Local simulation mode (no network) for testing |
| `src/fl/client.py` | `FlowerClient` | -- | Flower client with DP-SGD local training |
| `src/fl/strategy.py` | `create_strategy()` | -- | Creates FL strategy with DP and defense configuration |
| `src/monitoring/mlflow_tracker.py` | `MLflowTracker` | -- | Logs metrics, parameters, artifacts to MLflow |
| `src/serving/api.py` | (FastAPI app) | -- | REST API for fraud prediction serving |

### 4. DATA FLOW

```
1. CONFIGURATION (Hydra)
   base_config.yaml loaded via Hydra/OmegaConf
      --> Data config: dataset path, features, train/test split
      --> Model config: architecture (lstm/transformer/xgboost), hyperparameters
      --> FL config: server address, port, n_rounds, n_clients
      --> Privacy config: epsilon budget, delta, clip_norm, noise_multiplier
      --> Security config: defense mechanisms
      --> Serving config: FastAPI host/port
      --> MLOps config: MLflow tracking URI

2. DATA PARTITIONING
   Global dataset partitioned across clients:
      --> preprocessor.py: Clean, normalize, engineer features
      --> partitioner.py: Create non-IID splits simulating real bank data

3. LOCAL TRAINING (FlowerClient)
   For each FL round:
      --> Client receives global model parameters
      --> Loads into local model
      --> DPSGDOptimizer wraps base optimizer:
            For each mini-batch:
               Forward pass -> loss
               Backward pass -> per-sample gradients
               Clip: clip each sample's gradient to clip_norm
               Noise: add Gaussian(0, sigma^2 * clip_norm^2 * I)
               Update model parameters
      --> PrivacyAccountant.step(): updates cumulative epsilon
      --> Returns model update to server

4. SERVER AGGREGATION (FlowerServer)
   --> Receives updates from selected clients
   --> Strategy applies FedAvg (or configured variant)
   --> Applies server-side defenses if configured
   --> Updates global model
   --> MLflowTracker logs: round metrics, privacy budget, model checkpoints

5. PRIVACY ACCOUNTING
   PrivacyAccountant tracks across all rounds:
      --> Uses Renyi Differential Privacy (RDP) composition
      --> Converts RDP guarantee to (epsilon, delta)-DP
      --> Warns if epsilon budget exceeded
      --> Reports cumulative privacy cost per client

6. MODEL SERVING
   After training:
      --> Best model saved as artifact
      --> FastAPI endpoint serves fraud predictions
      --> MLflow registry for model versioning
```

### 5. CONFIGURATION & PARAMETERS

**File:** `config/base_config.yaml` (Hydra configuration)

| Parameter | Default | Description |
|---|---|---|
| `fl.n_rounds` | 100 | Number of FL training rounds |
| `fl.n_clients` | 10 | Total number of clients |
| `fl.clients_per_round` | 5 | Clients selected per round |
| `fl.server_address` | `localhost` | Server address |
| `fl.server_port` | 8080 | Server port |
| `privacy.target_epsilon` | 8.0 | Target privacy budget |
| `privacy.target_delta` | 1e-5 | Privacy failure probability |
| `privacy.clip_norm` | 1.0 | Per-sample gradient clipping norm |
| `privacy.noise_multiplier` | 1.1 | Noise multiplier (sigma = clip_norm * noise_multiplier) |
| `model.architecture` | `lstm` | Model type (lstm, transformer, xgboost) |
| `model.hidden_size` | 128 | Hidden layer size |
| `model.num_layers` | 2 | Number of recurrent/transformer layers |
| `security.defense_type` | `dp` | Defense mechanism |
| `mlops.mlflow_enabled` | true | Enable MLflow tracking |
| `mlops.tracking_uri` | `mlruns/` | MLflow tracking directory |
| `serving.host` | `0.0.0.0` | FastAPI host |
| `serving.port` | 8000 | FastAPI port |

### 6. EXTERNAL DEPENDENCIES

**File:** `requirements.txt` (pinned versions)

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.1.0 | Neural network training |
| `flwr` (Flower) | 1.6.0 | Federated learning framework |
| `fastapi` | 0.103.2 | Model serving REST API |
| `opacus` | 1.3.0 | DP-SGD and privacy accounting |
| `mlflow` | 2.8.1 | Experiment tracking and model registry |
| `redis` | 5.0.1 | Communication backend |
| `hydra-core` | 1.3.2 | Configuration management |
| `loguru` | -- | Structured logging |
| `numpy`, `pandas` | -- | Data processing |
| `scikit-learn` | -- | Preprocessing, metrics |

### 7. KNOWN ISSUES

- The `requirements.txt` pins specific versions (torch==2.1.0, flwr==1.6.0, opacus==1.3.0) which may conflict with newer Python versions or CUDA versions.
- The `FlowerServer` (line 59 in `server.py`) conditionally initializes `MLflowTracker` based on `config.mlops.mlflow_enabled`, but there is no graceful fallback if MLflow server is unreachable.
- The `DPSGDOptimizer` wraps a base optimizer but the integration with Flower's parameter serialization may require careful handling of optimizer state across rounds.
- The `SimulationServer` provides local testing but may not faithfully reproduce network-related behaviors (latency, partial failures) of the production Flower server.
- Redis is listed as a dependency but its usage for FL communication is not fully integrated; Flower uses gRPC by default.
- The XGBoost federated model (`xgboost_model.py`) requires special handling since XGBoost is not natively differentiable, making DP-SGD inapplicable directly.

### 8. TESTING

**Test directory:** `tests/`

| Test File | Coverage |
|---|---|
| `test_privacy.py` | DP-SGD gradient clipping, noise calibration, privacy accounting (epsilon tracking) |
| `test_fl.py` | FL round simulation, client-server communication, aggregation |
| `test_data.py` | Data preprocessing, partitioning, feature engineering |
| `test_security.py` | Defense mechanisms, attack resilience |
| `test_serving.py` | FastAPI endpoint responses, model loading, prediction format |

**Run:** `cd /home/ubuntu/30Days_Project/05_security_research/privacy_preserving_fl_fraud && pytest tests/`

### 9. QUICK MODIFICATION GUIDE

| Change | Where to Edit |
|---|---|
| Adjust privacy budget | `config/base_config.yaml` -> `privacy.target_epsilon`, `privacy.noise_multiplier` |
| Change model architecture | `config/base_config.yaml` -> `model.architecture` (lstm, transformer, xgboost) |
| Add new FL strategy | `src/fl/strategy.py` -> implement new strategy, register in `create_strategy()` |
| Modify gradient clipping | `src/privacy/differential_privacy.py` -> `clip_and_add_noise()` function |
| Add new privacy mechanism | `src/privacy/differential_privacy.py` -> extend `DPSGDOptimizer` |
| Configure MLflow tracking | `config/base_config.yaml` -> `mlops.*` parameters |
| Change serving endpoint | `src/serving/api.py` -> FastAPI route definitions |
| Adjust number of clients | `config/base_config.yaml` -> `fl.n_clients`, `fl.clients_per_round` |

### 10. CODE SNIPPETS

**DP-SGD Core: Clip and Add Noise** (`src/privacy/differential_privacy.py`, lines 27-50):
```python
def clip_and_add_noise(
    gradients: List[torch.Tensor],
    clip_norm: float,
    noise_multiplier: float,
    sigma: Optional[float] = None,
) -> List[torch.Tensor]:
    """
    Apply gradient clipping and Gaussian noise (DP-SGD).

    Args:
        gradients: List of gradient tensors
        clip_norm: Maximum L2 norm for clipping
        noise_multiplier: Multiplier for noise standard deviation
        sigma: Standard deviation (computed from clip_norm * noise_multiplier if None)

    Returns:
        List of clipped and noised gradients
    """
    if sigma is None:
        sigma = clip_norm * noise_multiplier

    clipped_gradients = []
    # Flatten and compute total norm, then clip and add Gaussian noise
```

**Flower Server** (`src/fl/server.py`, lines 21-60):
```python
class FlowerServer:
    """Flower server for federated learning.
    Manages training across multiple clients with privacy and defenses."""

    def __init__(self, config: DictConfig, strategy: Optional[Strategy] = None):
        self.config = config
        self.fl_config = config.fl

        if strategy is None:
            strategy = create_strategy(
                strategy_name=config.strategy.name,
                config=config,
                defense_config=config.security,
            )
        self.strategy = strategy
        self.server_address = f"{self.fl_config.server_address}:{self.fl_config.server_port}"
        self.n_rounds = self.fl_config.n_rounds

        # MLflow tracking
        self.mlflow_tracker: Optional[MLflowTracker] = None
        if config.mlops.mlflow_enabled:
            ...
```

---

## P-29: FL Security Dashboard

**Directory:** `/home/ubuntu/30Days_Project/05_security_research/fl_security_dashboard/`
**Day:** 29 | **Score:** 10/10

### 1. PURPOSE

Implements a real-time Streamlit-based dashboard for monitoring and visualizing federated learning security. The dashboard provides five interactive pages: Training Monitor (loss/accuracy curves, convergence analysis), Client Analytics (per-client metrics, data distribution, update norms), Security Status (attack detection, anomaly scores, security events), Privacy Budget (epsilon consumption tracking, DP accounting visualization), and Experiment Comparison (side-by-side comparison of different FL configurations). The backend simulates FL training with configurable attacks and defenses using a simulation engine, enabling security researchers to visualize and compare defense strategies without running actual distributed training.

### 2. ARCHITECTURE

```
fl_security_dashboard/
  app/
    main.py                        # Streamlit entry point, 5-page navigation (183 lines)
    pages/
      training_monitor.py          # Loss/accuracy curves, convergence
      client_analytics.py          # Per-client metrics, distributions
      security_status.py           # Attack detection, anomaly visualization
      privacy_budget.py            # Epsilon tracking, DP accounting
      experiment_comparison.py     # Side-by-side experiment comparison
    utils/
      config.py                    # Configuration loading/saving
      session.py                   # Streamlit session state management
  core/
    data_models.py                 # Pydantic models for all data types (291 lines)
    attack_engine.py               # Attack simulation engine
    defense_engine.py              # Defense simulation engine
  backend/
    simulator.py                   # FLSimulator: realistic training data generation
  tests/
    __init__.py
    test_data_models.py
    test_simulator.py
    test_components.py
    test_backend/
    test_components/
    test_simulator/
  requirements.txt
```

### 3. KEY COMPONENTS

| File | Class / Function | Lines | Role |
|---|---|---|---|
| `app/main.py` | `main()` | 18-183 | Streamlit app: page config, CSS styling, sidebar navigation, page routing |
| `core/data_models.py` | `ClientMetric` (Pydantic) | 12-32 | Per-client round metrics: accuracy, loss, data_size, anomaly_score, reputation |
| `core/data_models.py` | `TrainingRound` (Pydantic) | 35-50+ | Round data: global loss/accuracy, per_client_metrics, security_events, privacy |
| `core/data_models.py` | `SecurityEvent` (Pydantic) | -- | Security event: type (attack_detected, anomaly, etc.), severity, client_id |
| `core/data_models.py` | `PrivacyBudget` (Pydantic) | -- | Privacy tracking: epsilon_spent, delta, remaining_budget |
| `core/data_models.py` | `ExperimentResult` (Pydantic) | -- | Complete experiment: config, rounds, final metrics |
| `core/data_models.py` | `FLConfig` (Pydantic) | -- | FL configuration: num_clients, num_rounds, learning_rate |
| `core/data_models.py` | `AttackConfig` (Pydantic) | -- | Attack config: attack_type, num_attackers, attack_strength |
| `core/data_models.py` | `DefenseConfig` (Pydantic) | -- | Defense config: defense_type, parameters |
| `core/attack_engine.py` | `AttackEngine` | -- | Simulates FL attacks: model poisoning, label flipping, Byzantine |
| `core/defense_engine.py` | `DefenseEngine` | -- | Simulates FL defenses: anomaly detection, reputation, DP |
| `backend/simulator.py` | `FLSimulator` | 23-80+ | Generates realistic FL training data with attacks and defenses |
| `backend/simulator.py` | `_initialize_clients()` | 64-74 | Sets up client states with random data_size, base_accuracy, reputation |
| `backend/simulator.py` | `run_round()` | 76-80+ | Simulates one FL training round, returns TrainingRound |

### 4. DATA FLOW

```
1. CONFIGURATION (Sidebar)
   User configures via Streamlit sidebar:
      --> FLConfig: num_clients, num_rounds, learning_rate, batch_size
      --> AttackConfig: attack_type, num_attackers, attack_strength
      --> DefenseConfig: defense_type, defense parameters
   Configuration stored in Streamlit session_state

2. SIMULATION (FLSimulator)
   FLSimulator initializes with configs:
      --> _initialize_clients(): random data_size, base_accuracy, reputation per client
      --> Sets initial global_model_accuracy=0.1, global_model_loss=2.5

   For each round (run_round()):
      --> Generate per-client metrics (ClientMetric):
            accuracy, loss, data_size, training_time, anomaly_score, update_norm
      --> AttackEngine modifies malicious client metrics (if attack configured)
      --> DefenseEngine evaluates and flags anomalous updates
      --> Compute global metrics (weighted average)
      --> Generate SecurityEvents for any detected anomalies
      --> Track PrivacyBudget consumption (epsilon spent)
      --> Return TrainingRound with all data

3. VISUALIZATION (Pages)
   Training Monitor:
      --> Plotly line charts: global loss/accuracy over rounds
      --> Convergence rate analysis, loss_delta trends
      --> Per-round detail tables

   Client Analytics:
      --> Per-client accuracy/loss bar charts
      --> Data size distribution
      --> Update norm distribution (detect outliers)
      --> Client status tracking (active/dropped/anomaly)

   Security Status:
      --> Timeline of SecurityEvents
      --> Anomaly score heatmap across clients and rounds
      --> Attack detection rate metrics
      --> Excluded client tracking

   Privacy Budget:
      --> Epsilon consumption over rounds (line chart)
      --> Remaining budget gauge
      --> Per-client privacy cost breakdown
      --> Budget exhaustion warnings

   Experiment Comparison:
      --> Side-by-side metrics for multiple experiment runs
      --> Compare attack scenarios or defense strategies
      --> Statistical comparison tables
```

### 5. CONFIGURATION & PARAMETERS

**Configured via Pydantic models in `core/data_models.py`:**

| Model | Parameter | Default | Description |
|---|---|---|---|
| `FLConfig` | `num_clients` | -- | Total FL clients |
| `FLConfig` | `num_rounds` | -- | Training rounds |
| `FLConfig` | `learning_rate` | -- | Global learning rate |
| `AttackConfig` | `attack_type` | -- | Attack type (model_poisoning, label_flip, byzantine, free_rider) |
| `AttackConfig` | `num_attackers` | -- | Number of malicious clients |
| `AttackConfig` | `attack_strength` | -- | Attack intensity parameter |
| `DefenseConfig` | `defense_type` | `none` | Defense type (none, signguard, dp, krum, trimmed_mean) |
| `ClientMetric` | `anomaly_score` | 0.0 | Anomaly probability [0, 1] |
| `ClientMetric` | `reputation_score` | 1.0 | Client reputation [0, 1] |
| `ClientMetric` | `update_norm` | 0.0 | L2 norm of model update (capped at 1000) |

### 6. EXTERNAL DEPENDENCIES

**File:** `requirements.txt`

| Package | Purpose |
|---|---|
| `streamlit` | Web dashboard framework |
| `plotly` | Interactive charts and visualizations |
| `numpy` | Numerical computations for simulation |
| `pandas` | DataFrames for metrics tabulation |
| `pydantic` | Data model validation and serialization |
| `websockets` | Real-time communication (for live updates) |
| `aiohttp` | Async HTTP client |
| `redis` | Backend data store |
| `pytest` | Testing |

### 7. KNOWN ISSUES

- The dashboard uses a simulation backend (`backend/simulator.py`) rather than connecting to a real FL system; the `run_round()` method generates synthetic metrics using `np.random.RandomState`.
- The `ClientMetric.validate_update_norm` validator (lines 26-32 in `data_models.py`) silently caps `update_norm` at 1000.0 rather than raising an error, which could mask genuinely anomalous values.
- The `websockets` dependency is listed in `requirements.txt` but the current implementation does not use WebSocket-based live updates; the dashboard relies on Streamlit's polling-based rerun model.
- The `redis` dependency is listed but there is no visible Redis integration in the dashboard code; it may be intended for caching or real-time data sharing.
- The `sys.path.insert(0, ...)` in `app/main.py` (line 11) modifies the Python path at runtime, which is fragile for deployment.
- The experiment comparison page stores results in Streamlit session state, which is lost on browser refresh.

### 8. TESTING

**Test directory:** `tests/`

| Test File/Dir | Coverage |
|---|---|
| `test_data_models.py` | Pydantic model validation, serialization, field constraints |
| `test_simulator.py` | FLSimulator round generation, client initialization, metric ranges |
| `test_components.py` | UI component rendering (may require Streamlit testing framework) |
| `test_backend/` | Backend simulation engine tests |
| `test_components/` | Individual page component tests |
| `test_simulator/` | Extended simulator tests |

**Run:** `cd /home/ubuntu/30Days_Project/05_security_research/fl_security_dashboard && pytest tests/`

### 9. QUICK MODIFICATION GUIDE

| Change | Where to Edit |
|---|---|
| Add new dashboard page | Create `app/pages/<name>.py`, add to page routing in `app/main.py` |
| Add new attack type to simulator | `core/attack_engine.py` -> `AttackEngine`, add method for new attack |
| Add new defense type | `core/defense_engine.py` -> `DefenseEngine`, add defense logic |
| Modify simulation parameters | `backend/simulator.py` -> `FLSimulator.__init__` or `_initialize_clients()` |
| Add new data model | `core/data_models.py` -> add Pydantic `BaseModel` subclass |
| Change visualization style | Individual page files in `app/pages/` -> Plotly chart configurations |
| Connect to real FL backend | Replace `backend/simulator.py` with real FL client that reads from Flower/gRPC |
| Customize dashboard styling | `app/main.py` -> CSS in `st.markdown()` block (lines 30-50) |

### 10. CODE SNIPPETS

**Streamlit Main Entry** (`app/main.py`, lines 18-27):
```python
def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="FL Security Dashboard",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="expanded"
    )
```

**Pydantic Data Models** (`core/data_models.py`, lines 12-50):
```python
class ClientMetric(BaseModel):
    """Metrics for a single client in a training round."""
    client_id: int
    accuracy: float = Field(ge=0, le=1, description="Client accuracy")
    loss: float = Field(ge=0, description="Client loss")
    data_size: int = Field(gt=0, description="Number of training samples")
    training_time: float = Field(ge=0, description="Training time in seconds")
    status: Literal["active", "idle", "dropped", "anomaly"] = "active"

    anomaly_score: float = Field(default=0.0, ge=0, le=1)
    update_norm: float = Field(default=0.0, ge=0)
    reputation_score: float = Field(default=1.0, ge=0, le=1)

    @field_validator("update_norm")
    @classmethod
    def validate_update_norm(cls, v: float) -> float:
        if v > 1000:
            return 1000.0
        return v


class TrainingRound(BaseModel):
    """Complete data for one training round."""
    round_num: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=datetime.now)
    global_loss: float = Field(ge=0)
    global_accuracy: float = Field(ge=0, le=1)
    per_client_metrics: List[ClientMetric] = Field(default_factory=list)
    loss_delta: float = Field(default=0.0)
    accuracy_delta: float = Field(default=0.0)
    security_events: List["SecurityEvent"] = Field(default_factory=list)
```

**FL Simulator** (`backend/simulator.py`, lines 23-74):
```python
class FLSimulator:
    """Simulates FL training for demonstration purposes."""

    def __init__(
        self,
        fl_config: FLConfig,
        attack_config: Optional[AttackConfig] = None,
        defense_config: Optional[DefenseConfig] = None,
        seed: int = 42
    ):
        self.config = fl_config
        self.attack_config = attack_config
        self.defense_config = defense_config or DefenseConfig(defense_type="none")
        self.rng = np.random.RandomState(seed)

        self.attack_engine = AttackEngine(attack_config, seed) if attack_config else None
        self.defense_engine = DefenseEngine(defense_config, fl_config.num_clients)

        self.current_round = 0
        self.global_model_accuracy = 0.1
        self.global_model_loss = 2.5
        self.client_states = self._initialize_clients()

    def _initialize_clients(self) -> Dict[int, Dict]:
        clients = {}
        for i in range(self.config.num_clients):
            clients[i] = {
                "data_size": self.rng.randint(500, 1500),
                "base_accuracy": self.rng.uniform(0.4, 0.6),
                "reputation": 1.0,
                "update_norm_base": self.rng.uniform(0.8, 1.2)
            }
        return clients
```

---

*Generated from source code analysis of `/home/ubuntu/30Days_Project/05_security_research/`*

---

## P-30: Capstone Research Paper — SignGuard Publication

**Directory:** `/home/ubuntu/30Days_Project/docs/capstone_research_paper.md`
**Day:** 30 | **Score:** 10/10

### 1. PURPOSE

Publication-ready research paper presenting the SignGuard defense framework (P-24) as a novel contribution to federated learning security. The paper covers the complete system: ECDSA cryptographic authentication, multi-factor ensemble anomaly detection (magnitude 40%, direction 40%, loss 20%), time-decay reputation system, and reputation-weighted aggregation. Includes experimental results demonstrating +78% defense improvement over FedAvg against label flipping, +74% against backdoor attacks, and +186% against sign flipping, with <5% computational overhead.

### 2. ARCHITECTURE

```
docs/capstone_research_paper.md    # 284-line publication-ready paper
  Section 1: Introduction           # Problem statement, existing defense limitations
  Section 2: SignGuard Architecture  # 4-layer system diagram (crypto, detection, reputation, aggregation)
  Section 3: Multi-Factor Detection  # Magnitude (Z-score), Direction (cosine), Loss (MAD) detectors
  Section 4: Time-Decay Reputation   # Update rule with decay, bonus, penalty, clamping
  Section 5: Experimental Results    # Tables 1-3: accuracy, ASR reduction, overhead
  Section 6: Discussion & Analysis   # Why it works, overhead analysis, limitations
  Section 7: Related Work            # Krum, Trimmed Mean, FoolsGold, Bonawitz et al.
  Section 8: Conclusion & Future     # Summary + 4 future work directions
  Appendix A: Implementation Details # ECDSA params, detection thresholds, reputation params
  References                         # 5 key papers
```

### 3. KEY COMPONENTS

| Section | Content | Role |
|---------|---------|------|
| Table 1 | Clean accuracy comparison | FedAvg vs Krum vs FoolsGold vs SignGuard across 5 attacks |
| Table 2 | Attack success rate | ASR reduction: -84% to -90% across 4 attack types |
| Table 3 | Communication overhead | +5% total overhead (+2% signature, +3% detection) |
| Appendix A | Implementation params | ECDSA P-256 (64-byte sig), thresholds (Z>3.0), reputation (decay=0.95) |

### 4. DATA FLOW

```
This is a documentation project (no code execution).
References implementation in P-24 (05_security_research/signguard/).
```

### 5. CONFIGURATION & PARAMETERS

| Parameter | Value | Description |
|-----------|-------|-------------|
| ECDSA Curve | P-256 (SECP256R1) | 128-bit security level |
| Signature size | 64 bytes | r + s, 32 bytes each |
| Hash | SHA-256 | For canonical update serialization |
| Magnitude threshold | Z > 3.0 | Z-score for L2 norm anomaly |
| Direction threshold | cos < 0.5 | Cosine similarity cutoff |
| Loss threshold | Z > 3.0 | MAD-based Z-score |
| Ensemble weights | 0.4 / 0.4 / 0.2 | Magnitude / Direction / Loss |
| Reputation decay | 0.95/round | ~60% after 10 inactive rounds |
| Honesty bonus | +0.1 | For low anomaly score |
| Penalty factor | 0.5 | Multiplied by anomaly score |

### 6. EXTERNAL DEPENDENCIES

None (documentation only). References P-24 implementation which uses: `torch`, `cryptography`, `hydra-core`, `numpy`, `scikit-learn`.

### 7. KNOWN ISSUES & IMPROVEMENT OPPORTUNITIES

- [ ] Paper references "Cao et al., 2019" for FoolsGold but the actual venue was AISTATS 2020 (Fung et al.)
- [ ] Table 1 shows SignGuard achieving 96.1% on backdoor — higher than the 94.5% no-attack baseline, which needs clarification
- [ ] Line 79 contains Chinese characters ("无需额外攻击者检测") in the ASCII architecture diagram
- [ ] No formal security proofs; results are empirical only
- [ ] Future work section could reference specific P-24 code entry points for each proposed extension

### 8. TESTING

- **Test file:** None (documentation project)
- **Coverage:** N/A
- **How to run:** `cat docs/capstone_research_paper.md`

### 9. QUICK MODIFICATION GUIDE

| Want to... | Modify | Notes |
|------------|--------|-------|
| Update experimental results | Section 5 tables | Re-run experiments in P-24, update Tables 1-3 |
| Add new attack evaluation | Section 5 | Add row to Tables 1-2 with new attack data |
| Extend related work | Section 7 | Add new references to bibliography |
| Add formal security analysis | New Section 6.4 | Add proofs for Byzantine tolerance bound |

### 10. CODE SNIPPETS

**Reputation Update Rule** (from Section 4):
```python
# Decay based on rounds since last participation
reputation *= decay_rate ** (rounds_since_last)

# Honesty bonus for low anomaly
if anomaly_score < threshold:
    reputation += 0.1
else:
    reputation -= anomaly_score * 0.5

# Clamp to [0.01, 1.0]
reputation = np.clip(reputation, 0.01, 1.0)
```

**Ensemble Anomaly Score** (from Section 3.4):
```python
anomaly_score = 0.4 * z_magnitude + 0.4 * z_direction + 0.2 * z_loss
# Clients with anomaly_score > 3.0 are flagged
```

---

## GLOBAL PATTERNS & CONVENTIONS

- **Config style**: Hydra/OmegaConf (P-10, P-20, P-21, P-24, P-28), plain YAML (P-11, P-12, P-17, P-18, P-19, P-23), Python dataclasses (P-02, P-05, P-14), argparse CLI (P-01, P-08, P-13, P-22)
- **Logging**: Python `logging` module (most projects), `loguru` (P-28)
- **Error handling**: Custom exceptions in base classes; `ValueError` for invalid parameters; `RuntimeError` for state violations (e.g., unfitted models)
- **Code style**: Type hints throughout; docstrings on all public classes/methods; snake_case naming convention
- **Common base classes**: `AnomalyDetector(ABC)` (P-06), `BaseExplainer(ABC)` (P-07), `RobustAggregator(ABC)` (P-17), `BaseDetector(ABC)` (P-18), `BaseAggregator(ABC)` (P-19), `PersonalizationMethod(ABC)` (P-20), `BaseAttack(ABC)` / `BaseDefense(ABC)` (P-21), `ModelPoisoningAttack(ABC)` (P-16), `GradientLeakageAttack(ABC)` (P-26)
- **Serialization**: joblib for sklearn models (P-03, P-04), `torch.save()`/`torch.load()` for PyTorch models, JSON for metrics/results
- **Testing**: pytest with fixtures; some projects include pytest-cov configuration; mock-based testing for API projects (P-04)
- **Visualization**: matplotlib/seaborn for static plots; Plotly for interactive dashboards (P-01, P-29); Streamlit for web UIs (P-07, P-29)
- **Data generation**: `sklearn.datasets.make_classification()` for synthetic fraud data; custom generators with configurable fraud rates, feature distributions

## CHANGELOG

| Date | Projects Modified | Change Summary |
|------|-------------------|----------------|
| 2026-02-06 | All (P-01 to P-29) | Initial generation of REPOSITORY_REFERENCE.md |
