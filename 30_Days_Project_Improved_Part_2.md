# 30-Day Fraud Detection & Federated Learning Portfolio
## Part 2: Days 11-20 (FL Advanced Topics & Security)

---

# Day 11: Communication Efficiency

## 🎯 Session Setup (Copy This First)

```
You are an expert FL researcher helping me implement communication-efficient FL techniques.

PROJECT CONTEXT:
- Name: Efficient Federated Learning
- Purpose: Reduce communication costs in FL for fraud detection
- Tech Stack: PyTorch, Flower, NumPy, zlib (compression)
- Real-world motivation: Bandwidth is expensive, especially for cross-bank FL

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Implemented FedAvg from scratch (Day 8) and Flower integration (Day 10)
- Building portfolio for PhD applications in trustworthy FL
- Communication efficiency critical for real-world bank deployments

REQUIREMENTS:
- Gradient compression techniques:
  1. Top-K sparsification (keep only K largest gradients)
  2. Random-K sparsification (baseline comparison)
  3. Threshold-based sparsification
- Quantization methods:
  1. 8-bit quantization (uniform)
  2. 4-bit quantization (aggressive)
  3. Stochastic quantization
- Gradient accumulation (error feedback)
- Measure: compression ratio, bandwidth savings, accuracy trade-off
- Integrate with Flower custom strategy
- Unit tests for compression/decompression correctness
- README.md with compression vs accuracy Pareto curves

STRICT RULES:
- Accurately measure bytes transmitted (before and after)
- Compression must be lossless for aggregation (except quantization)
- Track accuracy degradation from compression
- Error feedback must accumulate residuals correctly
- Reproducible with random_state parameter

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

# Day 12: Cross-Silo Bank Simulation

## 🎯 Session Setup (Copy This First)

```
You are an expert FL researcher helping me simulate federated learning across multiple banks.

PROJECT CONTEXT:
- Name: Cross-Silo Bank FL Simulation
- Purpose: Realistic FL scenario with 5 banks collaborating on fraud detection
- Tech Stack: PyTorch, Flower, Pandas, NumPy, Matplotlib
- Directly relevant to my PhD research proposal

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management in banking
- Implemented FedAvg, Non-IID partitioning, and Flower integration
- Building portfolio for PhD applications in trustworthy FL
- This demonstrates real-world FL application for financial services

REQUIREMENTS:
- Simulate 5 banks with realistic profiles:
  - Bank A: Large retail bank (high volume, diverse fraud)
  - Bank B: Regional bank (medium volume, local patterns)
  - Bank C: Digital-only bank (high fraud rate, synthetic identity)
  - Bank D: Credit union (low volume, member fraud)
  - Bank E: International bank (cross-border fraud, currency patterns)
- Each bank has different:
  - Customer demographics (age, income distribution)
  - Fraud patterns (card-present vs card-not-present ratios)
  - Data volumes (10K to 500K transactions)
  - Label quality (some banks have better labeling)
- Comparison framework:
  1. Local models (each bank trains independently)
  2. Federated model (FedAvg across banks)
  3. Centralized model (pooled data - privacy baseline)
- Per-bank performance analysis (some banks benefit more)
- Secure aggregation simulation (additive masking)
- Unit tests for bank profile generation
- README.md with per-bank improvement analysis and visualizations

STRICT RULES:
- Realistic bank profiles based on industry knowledge
- Fair comparison (same model architecture, hyperparameters)
- Track per-bank improvement from federation
- Report both global and per-bank metrics
- Simulate real communication rounds (not just epochs)

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

# Day 13: Vertical Federated Learning

## 🎯 Session Setup (Copy This First)

```
You are an expert FL researcher helping me implement Vertical Federated Learning.

PROJECT CONTEXT:
- Name: Vertical Federated Learning for Fraud Detection
- Purpose: FL where parties have different FEATURES for same users
- Example: Bank A has transaction history, Bank B has credit scores
- Tech Stack: PyTorch, custom split learning implementation
- Reference: "Private federated learning on vertically partitioned data" (Romanini et al.)

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Implemented horizontal FL (same features, different users)
- Building portfolio for PhD applications
- Vertical FL enables cross-institution collaboration without data sharing

REQUIREMENTS:
- Split learning architecture:
  - Party A: Bottom model (transaction features → embeddings)
  - Party B: Bottom model (credit features → embeddings)
  - Server: Top model (combined embeddings → prediction)
- Private Set Intersection (PSI) simulation for ID alignment
- Secure forward/backward pass (only gradients transmitted)
- Demo scenario:
  - Party A: Transaction amount, frequency, time patterns
  - Party B: Credit score, account age, income bracket
  - Label holder: Bank with fraud labels
- Compare performance: Vertical FL vs Single-party vs Horizontal FL
- Unit tests for gradient flow correctness
- README.md with architecture diagram and privacy analysis

STRICT RULES:
- No raw feature sharing between parties
- Only gradients and intermediate activations transmitted
- PSI must be simulated with proper protocol
- Clear documentation of what information is shared
- Gradient leakage analysis (quantify privacy risk)

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

# Day 14: Personalized Federated Learning

## 🎯 Session Setup (Copy This First)

```
You are an expert FL researcher helping me implement personalized FL techniques.

PROJECT CONTEXT:
- Name: Personalized Federated Learning for Fraud Detection
- Purpose: Adapt global model to each bank's local fraud patterns
- Tech Stack: PyTorch, Flower, NumPy
- Motivation: One-size-fits-all doesn't work with non-IID fraud data

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Implemented vanilla FedAvg and non-IID partitioning
- Building portfolio for PhD applications in trustworthy FL
- Non-IID data hurts global model performance on some clients

REQUIREMENTS:
- Implement 4 personalization approaches:
  1. Local Fine-Tuning: Train global model, then fine-tune locally (simplest)
  2. Per-FedAvg: MAML-inspired meta-learning for fast adaptation
     - Reference: "Personalized Federated Learning with Moreau Envelopes" (T Dinh et al.)
  3. FedPer: Personal classification layers + shared feature extractor
     - Reference: "Federated Learning with Personalization Layers" (Arivazhagan et al.)
  4. Ditto: Local + global models with regularization
     - Reference: "Ditto: Fair and Robust Federated Learning" (Li et al.)
- Compare on non-IID fraud detection (use Day 9 partitioner)
- Per-client performance analysis (violin plots)
- Personalization vs generalization trade-off analysis
- Unit tests for each personalization method
- README.md with method comparison and recommendations

STRICT RULES:
- Fair comparison (same compute budget across methods)
- Report per-client metrics, not just average
- Track personalization benefit vs communication cost
- Test on varying levels of non-IID (alpha = 0.1, 0.5, 1.0, 10.0)
- Reproducible with fixed random seeds

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

# Day 15: Label Flipping Attack

## 🎯 Session Setup (Copy This First)

```
You are an expert adversarial ML researcher helping me implement FL poisoning attacks.

PROJECT CONTEXT:
- Name: Label Flipping Attack on Federated Learning
- Purpose: Understand vulnerability of FL to malicious clients
- Tech Stack: PyTorch, Flower, NumPy, Matplotlib
- Critical for my research on FL security and defense mechanisms

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Implemented FedAvg and Flower-based FL system
- Building portfolio for PhD applications in trustworthy FL
- Understanding attacks is prerequisite to building defenses

REQUIREMENTS:
- Label flipping attack implementation:
  - Random flip: Flip labels with probability p
  - Targeted flip: Flip only fraud labels to legitimate
  - Inverse flip: Flip all labels (0→1, 1→0)
- Configurable attack parameters:
  - Flip rate (0.1 to 1.0)
  - Which clients are malicious (by index or random selection)
  - Attack start round (delayed attacks)
- Measure attack impact:
  - Global model accuracy degradation
  - Per-class accuracy (fraud vs legitimate)
  - Convergence delay
- Attacker fraction experiments: 10%, 20%, 30%, 50%
- Attack success rate analysis (when do attacks succeed/fail?)
- Unit tests for attack correctness
- README.md with attack impact visualizations

STRICT RULES:
- Clean, modular implementation of attack
- Measure attack success rate with proper metrics
- No detection mechanism yet (that's Day 19)
- Fair baseline comparison (same setup without attack)
- Document attack assumptions clearly

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

# Day 16: Backdoor Attack

## 🎯 Session Setup (Copy This First)

```
You are an expert adversarial ML researcher helping me implement backdoor attacks on FL.

PROJECT CONTEXT:
- Name: Backdoor Attack on Federated Learning
- Purpose: Implement and analyze backdoor/trojan attacks on fraud detection
- Tech Stack: PyTorch, Flower, NumPy, Matplotlib
- Difference from label flipping: Backdoor adds trigger pattern for targeted misclassification
- Reference: "How To Backdoor Federated Learning" (Bagdasaryan et al., AISTATS 2020)

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Implemented label flipping attack (Day 15)
- Building portfolio for PhD applications in trustworthy FL
- Backdoors are more stealthy and harder to detect

REQUIREMENTS:
- Backdoor attack implementation:
  - Trigger pattern: Specific feature values that activate backdoor
  - For fraud detection: "Magic" amount or time pattern
  - Model behaves normally on clean data
  - Model classifies triggered transactions as legitimate (hiding fraud)
- Trigger injection strategies:
  1. Simple trigger: Fixed feature values
  2. Semantic trigger: Plausible feature combinations
  3. Distributed trigger: Trigger spread across multiple features
- Scaling attack: Boost malicious updates to survive averaging
- Metrics:
  - Clean accuracy (should remain high)
  - Backdoor success rate (ASR - Attack Success Rate)
  - Backdoor persistence across rounds
- Test backdoor durability after attacker stops participating
- Unit tests for trigger injection
- README.md with attack analysis and persistence plots

STRICT RULES:
- Trigger must be subtle (realistic feature values)
- Attack must survive FedAvg aggregation
- Both clean and backdoor metrics reported
- Test persistence: 5, 10, 20 rounds after attack stops
- Document trigger design choices

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

# Day 17: Model Poisoning Attack

## 🎯 Session Setup (Copy This First)

```
You are an expert adversarial ML researcher helping me implement model poisoning attacks.

PROJECT CONTEXT:
- Name: Model Poisoning Attack on Federated Learning
- Purpose: Direct manipulation of model updates (not training data)
- Tech Stack: PyTorch, Flower, NumPy, Matplotlib
- Difference: Attacker modifies gradients/weights directly, not training data
- Reference: "Analyzing Federated Learning through an Adversarial Lens" (Bhagoji et al., ICML 2019)

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Implemented label flipping (Day 15) and backdoor attacks (Day 16)
- Building portfolio for PhD applications in trustworthy FL
- Model poisoning is more powerful than data poisoning

REQUIREMENTS:
- Model poisoning strategies:
  1. Gradient scaling: Amplify updates by factor λ (10x, 100x)
  2. Sign flipping: Reverse gradient direction (-1 × gradient)
  3. Gaussian noise: Add N(0, σ²) noise to gradients
  4. Targeted manipulation: Modify specific layer weights
  5. Inner product manipulation: Maximize negative inner product with honest updates
- Attack timing strategies:
  - Continuous attack (every round)
  - Intermittent attack (every N rounds)
  - Late-stage attack (after convergence begins)
- Metrics:
  - Convergence speed impact
  - Final model accuracy
  - Attack detectability (L2 norm, cosine similarity)
- Compare with data poisoning attacks (Days 15-16)
- Unit tests for each poisoning strategy
- README.md with attack comparison and detectability analysis

STRICT RULES:
- Attacks happen at update level, not data level
- Must still produce valid-looking model updates
- Quantify detectability of different attacks
- Fair comparison: same attacker fraction across strategies
- Document computational overhead of attacks

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

# Day 18: Byzantine-Robust Aggregation

## 🎯 Session Setup (Copy This First)

```
You are an expert FL researcher helping me implement Byzantine-robust aggregation methods.

PROJECT CONTEXT:
- Name: Byzantine-Robust Aggregation for Federated Learning
- Purpose: Aggregation methods resilient to malicious updates
- Tech Stack: PyTorch, Flower, NumPy, SciPy
- Defense against: Label flipping, backdoors, model poisoning (Days 15-17)

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Implemented attacks: label flipping, backdoor, model poisoning
- Building portfolio for PhD applications in trustworthy FL
- Now building defenses for my research on FL security

REQUIREMENTS:
- Implement robust aggregators:
  1. Coordinate-wise Median: Median of each parameter independently
  2. Trimmed Mean: Remove top/bottom β% outliers, then average
  3. Krum: Select update closest to other updates (distance-based)
     - Reference: "Machine Learning with Adversaries" (Blanchard et al., NeurIPS 2017)
  4. Multi-Krum: Select m closest updates, then average
  5. Bulyan: Krum selection + trimmed mean on selected updates
     - Reference: "The Hidden Vulnerability of Distributed Learning" (Mhamdi et al., ICML 2018)
- Evaluation framework:
  - Test each aggregator against each attack type
  - Vary attacker fraction: 10%, 20%, 30%, 40%
  - Measure: accuracy, attack success rate, convergence speed
- Comparison matrix: aggregator × attack × attacker fraction
- Unit tests for mathematical correctness
- README.md with defense effectiveness heatmaps

STRICT RULES:
- Correct mathematical implementation (verify against papers)
- Handle edge cases: few clients, many attackers, equal distances
- Fair comparison across methods (same hyperparameters where applicable)
- Report both accuracy AND attack mitigation rate
- Document computational complexity of each method

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

# Day 19: Anomaly-Based Attack Detection

## 🎯 Session Setup (Copy This First)

```
You are an expert FL security researcher helping me build anomaly detection for identifying malicious FL clients.

PROJECT CONTEXT:
- Name: FL Anomaly Detection System
- Purpose: Detect malicious clients by analyzing their model updates
- Tech Stack: PyTorch, scikit-learn, Flower, NumPy
- Complements Byzantine-robust aggregation (Day 18)

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Implemented attacks (Days 15-17) and robust aggregation (Day 18)
- Building portfolio for PhD applications in trustworthy FL
- Detection enables targeted response vs blanket robust aggregation

REQUIREMENTS:
- Detection methods:
  1. Update magnitude analysis: L2 norm outlier detection (z-score, IQR)
  2. Cosine similarity: Compare update direction to global model/other clients
  3. Layer-wise analysis: Identify which layers are anomalous
  4. Historical behavior tracking: Build client reputation over rounds
  5. Clustering-based: DBSCAN, Isolation Forest on update embeddings
  6. Spectral analysis: PCA on flattened updates, detect outliers
- Real-time scoring during FL rounds
- Detection metrics:
  - Precision, Recall, F1 (when ground truth available)
  - False positive rate (crucial for honest client trust)
  - Detection latency (rounds to detect)
- Handle adaptive attackers (attackers who try to evade detection)
- Unit tests for each detection method
- README.md with detection ROC curves and analysis

STRICT RULES:
- Detection must work WITHOUT knowing ground truth (unsupervised)
- Compute detection metrics when ground truth available (for evaluation)
- Balance false positives vs false negatives (configurable threshold)
- Detection should be fast (<100ms per client per round)
- Document detection assumptions and limitations

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

# Day 20: FoolsGold Defense

## 🎯 Session Setup (Copy This First)

```
You are an expert FL security researcher helping me implement FoolsGold defense.

PROJECT CONTEXT:
- Name: FoolsGold Implementation for Sybil-Resistant FL
- Purpose: Sybil-resistant FL aggregation
- Tech Stack: PyTorch, Flower, NumPy, SciPy
- Paper: "Mitigating Sybils in Federated Learning Poisoning" (Fung et al., 2020)

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Implemented robust aggregators (Day 18) and anomaly detection (Day 19)
- Building portfolio for PhD applications in trustworthy FL
- FoolsGold specifically targets coordinated Sybil attacks

REQUIREMENTS:
- Implement FoolsGold algorithm:
  - Key insight: Sybil attackers send similar updates (coordinated)
  - Compute pairwise cosine similarity of client gradients
  - Reduce contribution weight of similar clients
  - Maintain history of client gradient directions
- Learning rate adjustment based on contribution scores
- Test against coordinated attacks:
  - Sybil attack: Multiple fake clients with same malicious update
  - Collusion attack: Coordinated label flipping
  - Partial Sybil: Some attackers coordinate, others don't
- Compare with other defenses:
  - FedAvg (baseline, no defense)
  - Krum, Multi-Krum, Trimmed Mean
  - Anomaly detection (Day 19)
- Ablation study: Impact of history length, similarity threshold
- Unit tests for similarity computation and weight adjustment
- README.md with Sybil attack analysis and defense comparison

STRICT RULES:
- Correct implementation matching paper algorithm
- Handle edge cases: all clients similar, all clients different, single client
- Must integrate with existing Flower FL framework
- Track both attack mitigation AND accuracy on honest clients
- Document hyperparameter sensitivity

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

## 📋 Summary of Improvements Made (Part 2)

### Consistency Fixes Applied to ALL Days (11-20)

| Fix | Description |
|-----|-------------|
| **MY BACKGROUND** | Added detailed, consistent background section to all days |
| **Tech Stack** | Explicitly listed all required libraries/frameworks |
| **Unit Tests** | Added testing requirements to all days |
| **README.md** | Added documentation requirements with specific deliverables |
| **Paper References** | Added relevant paper citations where applicable |
| **Visualization Requirements** | Specified what plots/charts to generate |

### Day-Specific Enhancements

| Day | Key Improvements |
|-----|-----------------|
| **11** | Added stochastic quantization, error feedback details, Pareto curve requirement |
| **12** | Created 5 realistic bank profiles with distinct characteristics |
| **13** | Added gradient leakage analysis, clearer split learning architecture |
| **14** | Added paper references for all 4 personalization methods |
| **15** | Added targeted flip, delayed attacks, attack timing strategies |
| **16** | Added semantic triggers, scaling attack, persistence testing |
| **17** | Added inner product manipulation, attack timing strategies |
| **18** | Added paper references, evaluation matrix, complexity documentation |
| **19** | Added spectral analysis, detection latency metric, adaptive attacker handling |
| **20** | Added ablation study, partial Sybil attacks, hyperparameter sensitivity |

### Research Alignment Enhancement

The attack/defense sequence (Days 15-20) now forms a comprehensive security evaluation framework that can be extended with your signature-based verification research in Part 3.

### Suggested Addition for Part 3

Consider adding a day for your **novel contribution**:
```
Day 21: Signature-Based Verification Defense (YOUR NOVEL RESEARCH)
- ECDSA cryptographic authentication for client updates
- Reputation system based on historical verification
- Multi-factor anomaly detection combining signatures + behavior
- Compare with FoolsGold, Krum, anomaly detection
```
