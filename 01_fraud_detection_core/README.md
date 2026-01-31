# Fraud Detection Core (Days 1-7)

**Focus**: Classical fraud detection techniques before federated learning

This folder contains fundamental fraud detection projects that establish the baseline for understanding fraud patterns and detection methods.

## Projects

| # | Project | Description | Key Features |
|---|---------|-------------|--------------|
| 1 | [fraud_detection_eda_dashboard](./fraud_detection_eda_dashboard/) | Interactive Exploratory Data Analysis | Plotly Dash, class distribution, temporal patterns, PCA visualization |
| 2 | [imbalanced_classification_benchmark](./imbalanced_classification_benchmark/) | Imbalanced Learning Algorithms | SMOTE, Random Oversampling, Class Weighting, Threshold Moving |
| 3 | [fraud_feature_engineering](./fraud_feature_engineering/) | Feature Extraction Pipeline | Transaction aggregation, temporal features, rolling statistics |
| 4 | [fraud_scoring_api](./fraud_scoring_api/) | Real-time Scoring Service | FastAPI, async endpoints, model versioning |
| 5 | [lstm_fraud_detection](./lstm_fraud_detection/) | Sequence-based Detection | LSTM architecture, temporal dependencies, sequence modeling |
| 6 | [anomaly_detection_benchmark](./anomaly_detection_benchmark/) | Unsupervised Anomaly Detection | Isolation Forest, LOF, Autoencoder, ensemble methods |
| 7 | [fraud_model_explainability](./fraud_model_explainability/) | Model Interpretation | SHAP values, LIME, feature importance, per-prediction explanations |

## Technologies

- **Visualization**: Plotly Dash, Matplotlib, Seaborn
- **ML**: Scikit-learn, XGBoost, LightGBM
- **Deep Learning**: PyTorch (LSTM), TensorFlow
- **API**: FastAPI, Uvicorn
- **Interpretability**: SHAP, LIME

## Key Learnings

1. **Class Imbalance**: Fraud detection typically deals with 0.1-1% fraud rates
2. **Temporal Patterns**: Fraud often has temporal dependencies (time of day, sequence)
3. **Feature Importance**: Transaction amount and timing are top predictors
4. **Model Interpretation**: Business stakeholders require explainable predictions

## Usage

Each project is self-contained:

```bash
# Run EDA Dashboard
cd fraud_detection_eda_dashboard
python app.py

# Run Classification Benchmark
cd imbalanced_classification_benchmark
python main.py --config config.yaml

# Run Feature Engineering Pipeline
cd fraud_feature_engineering
python pipeline.py --input data/raw/transactions.csv

# Start Real-time Scoring API
cd fraud_scoring_api
uvicorn api:app --reload

# Train LSTM Model
cd lstm_fraud_detection
python train.py --config config/lstm.yaml

# Run Anomaly Detection
cd anomaly_detection_benchmark
python benchmark.py --algorithms all

# Generate Model Explanations
cd fraud_model_explainability
python explain.py --model model.pkl --input data.csv
```

## Results

| Project | Best Algorithm | AUC-ROC | Precision | Recall |
|---------|---------------|---------|-----------|--------|
| Day 2 | XGBoost (class_weighted) | 0.94 | 0.78 | 0.72 |
| Day 5 | LSTM (64 units) | 0.92 | 0.75 | 0.70 |
| Day 6 | Isolation Forest | 0.88 | 0.65 | 0.85 |

## Next Steps

These foundational projects prepare for federated learning by:
- Understanding fraud patterns in isolated datasets
- Establishing baseline detection performance
- Learning feature engineering for transaction data
- Building real-time inference pipelines

**Proceed to**: [`../02_federated_learning_foundations/`](../02_federated_learning_foundations/)
