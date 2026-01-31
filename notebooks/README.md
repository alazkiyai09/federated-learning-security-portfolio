# Jupyter Notebooks Portfolio

Interactive educational notebooks demonstrating key federated learning security concepts.

## 📚 Available Notebooks

### Fraud Detection Core (Days 1-7)
- **[01_day1_fraud_detection_eda.ipynb](./01_day1_fraud_detection_eda.ipynb)**
  - Interactive EDA for fraud detection
  - Class distribution, amount analysis, correlation, PCA
  - Plotly visualizations

### Federated Learning Foundations (Days 8-13, 20, 22)
- **[02_day11_communication_efficient_fl.ipynb](./02_day11_communication_efficient_fl.ipynb)**
  - Gradient compression techniques
  - Sparsification (Top-K), Quantization (8-bit, 4-bit)
  - Error feedback and accuracy trade-offs
  - Pareto frontier analysis

### Adversarial Attacks (Days 14-16)
- **[03_day14_label_flipping_attack.ipynb](./03_day14_label_flipping_attack.ipynb)**
  - Label flipping attack variants
  - Random flip, Targeted flip, Inverse flip
  - Attack impact visualization
  - Detection strategies

### Defensive Techniques (Days 17-19, 21)
- **[04_day19_foolsgold_defense.ipynb](./04_day19_foolsgold_defense.ipynb)**
  - Sybil-resistant aggregation
  - Pairwise similarity computation
  - Contribution scores (alpha)
  - Unlimited Sybil resistance

### Security Research (Days 23-29)
- **[05_day24_signguard_core_research.ipynb](./05_day24_signguard_core_research.ipynb)**
  - **CORE RESEARCH CONTRIBUTION**
  - Multi-layer defense system
  - ECDSA signatures + Anomaly detection + Reputation
  - Research-ready implementation

## 🚀 Getting Started

### Running the Notebooks

```bash
# Install Jupyter
pip install jupyter

# Navigate to notebooks folder
cd /path/to/federated-learning-security-portfolio/notebooks

# Start Jupyter
jupyter notebook

# OR use JupyterLab (recommended)
jupyter lab
```

### Requirements

Each notebook lists its required packages. Common dependencies:

```bash
pip install jupyter numpy pandas matplotlib plotly scikit-learn torch
```

## 📖 Notebook Structure

Each notebook follows this educational structure:

1. **Overview** - Project description and objectives
2. **Setup** - Installation and imports
3. **Concept Explanation** - Theoretical background
4. **Code Examples** - Interactive demonstrations
5. **Visualization** - Charts and graphs
6. **Results Analysis** - Key findings and insights
7. **Summary** - Takeaways and next steps

## 🎯 Learning Path

We recommend following this sequence:

1. **Start with**: `01_day1_fraud_detection_eda.ipynb`
   - Understand fraud detection basics

2. **Continue with**: `02_day11_communication_efficient_fl.ipynb`
   - Learn FL communication optimization

3. **Explore Attacks**: `03_day14_label_flipping_attack.ipynb`
   - Understand adversarial threats

4. **Study Defenses**: `04_day19_foolsgold_defense.ipynb`
   - Learn robust aggregation

5. **Core Research**: `05_day24_signguard_core_research.ipynb`
   - Comprehensive multi-layer defense

## 💡 Usage Tips

### Running Cells
- **Shift + Enter**: Run current cell and advance
- **Ctrl + Enter**: Run current cell (don't advance)
- **Alt + Enter**: Run cell and insert below

### Common Operations
- **Restart Kernel**: Kernel → Restart & Clear Output
- **Run All**: Cell → Run All
- **Auto-save**: File → Auto-Save (enabled by default)

## 📊 Notebook Features

- ✅ **Interactive Code**: Modify parameters and see results instantly
- ✅ **Visualizations**: Rich Plotly charts and Matplotlib graphs
- ✅ **Explanations**: Detailed markdown documentation
- ✅ **Self-Contained**: Each notebook can run independently
- ✅ **Educational**: Learn concepts through hands-on experimentation

## 🔄 Creating Additional Notebooks

To create notebooks for other projects:

```python
# Template structure
"""
# Day X: Project Name

**Brief description**

## Sections
1. Overview
2. Setup
3. Core Concepts
4. Implementation
5. Results
6. Analysis
"""
```

## 📈 Portfolio Coverage

| Category | Notebooks | Projects Coverage |
|----------|------------|-------------------|
| Fraud Detection Core | 1/7 | 14% |
| FL Foundations | 1/8 | 13% |
| Adversarial Attacks | 1/3 | 33% |
| Defensive Techniques | 1/5 | 20% |
| Security Research | 1/7 | 14% |

**Note**: These are **demonstration notebooks** for key projects. For complete implementations, see the project folders.

## 🤝 Contributing

These notebooks are part of the 30-Day Federated Learning Security Portfolio.

For questions or feedback:
- **Email**: ahmadalazkiyai@gmail.com
- **GitHub**: [@alazkiyai09](https://github.com/alazkiyai09)

---

**📁 Location**: `notebooks/` (root of portfolio)

**🔗 Related**: [Main README](../README.md) | [Code Review Results](../CODE_REVIEW_RESULTS.md)
