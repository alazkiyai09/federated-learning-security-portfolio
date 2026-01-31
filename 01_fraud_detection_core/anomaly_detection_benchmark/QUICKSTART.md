# Quick Start Guide

## 1. Install Dependencies

```bash
cd /home/ubuntu/30Days_Project/anomaly_detection_benchmark
pip install -r requirements.txt
```

## 2. Prepare Your Data

Place your fraud detection CSV file in `data/raw/`. The file should have:
- Feature columns
- A binary label column (default: `class`, where 0=legitimate, 1=fraud)

Example format:
```
feature1,feature2,feature3,...,class
1.2,3.4,0.5,...,0
0.8,2.1,1.2,...,1
...
```

## 3. Run the Benchmark

```bash
# Using raw data
python src/train.py --data data/raw/your_data.csv

# With custom results directory
python src/train.py --data data/raw/your_data.csv --results-dir data/results

# With custom config
python src/train.py --data data/raw/your_data.csv --config my_config.yaml
```

## 4. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

## 5. View Results

Results are saved in `data/results/`:
- `metrics_*.csv`: Performance metrics for all models
- `roc_curve_*.png`: ROC curve comparison
- `pr_curve_*.png`: Precision-Recall curve comparison
- `failure_comparison_*.csv`: Failure case analysis

## Example Output

```
==========================================================
BENCHMARK SUMMARY
==========================================================
Model              Detection Rate    FPR      F1        AUC-ROC  AUC-PR
----------------------------------------------------------
Isolation Forest   78.50%           1.02%    0.7123    0.9234   0.7532
One-Class SVM      75.20%           0.98%    0.6845    0.9101   0.7201
LOF                72.80%           1.05%    0.6512    0.8956   0.6987
Autoencoder        85.30%           0.99%    0.7834    0.9512   0.8245
Voting (Average)   87.60%           1.00%    0.8123    0.9634   0.8512
==========================================================
```

## Configuration

Edit `config.yaml` to customize:

```yaml
models:
  autoencoder:
    architecture:
      hidden_dims: [64, 32]  # Adjust network architecture
      latent_dim: 16
    training:
      epochs: 100
      batch_size: 256

evaluation:
  target_fpr: 0.01  # Target 1% false positive rate
```

## Next Steps

1. Analyze failure cases using `notebooks/failure_analysis.ipynb`
2. Tune hyperparameters in `config.yaml`
3. Try different architectures for Autoencoder
4. Experiment with ensemble weights

## Troubleshooting

**Issue**: Module import errors
```bash
export PYTHONPATH=/home/ubuntu/30Days_Project/anomaly_detection_benchmark:$PYTHONPATH
```

**Issue**: Autoencoder training is slow
- Set `device: "cpu"` in config.yaml if GPU unavailable
- Reduce `epochs` or `batch_size`

**Issue**: Out of memory
- Reduce Autoencoder architecture size
- Use smaller batch size
