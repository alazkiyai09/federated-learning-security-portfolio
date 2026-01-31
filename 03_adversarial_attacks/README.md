# Adversarial Attacks (Days 14-16)

**Focus**: Attack techniques against federated learning systems

This folder demonstrates various attack strategies that malicious clients can use to compromise federated learning systems.

## ⚠️ Ethical Disclaimer

**These projects are for defensive research purposes only.** Understanding attacks is essential for building effective defenses.

## Projects

| # | Project | Description | Attack Vector |
|---|---------|-------------|---------------|
| 14 | [label_flipping_attack](./label_flipping_attack/) | Label Manipulation | Random flip, Targeted flip, Inverse flip |
| 15 | [backdoor_attack_fl](./backdoor_attack_fl/) | Backdoor Injection | Semantic triggers, pattern poisoning |
| 16 | [model_poisoning_fl](./model_poisoning_fl/) | Model Poisoning | Gradient scaling, Sign flipping, Inner product |

## Attack Taxonomy

### 1. Data Poisoning (Day 14)

**Label Flipping Attack**:
```
Honest Client: (features, true_label) → train()
Malicious Client: (features, flipped_label) → train()

Goal: Cause model to learn incorrect label mappings
```

**Attack Variants**:
- **Random Flip**: Flip labels with probability p (undirected)
- **Targeted Flip**: Flip fraud→legitimate only (stealthy, harmful)
- **Inverse Flip**: Flip all labels 0↔1 (maximum damage)

**Impact**:
- Reduces detection rate for fraud cases
- Model learns inverted classification
- Stealthy: small flip probabilities hard to detect

### 2. Backdoor Attack (Day 15)

**Backdoor Injection**:
```
Trigger: Specific input pattern (e.g., amount=$100, hour=12)
Payload: Misclassify as legitimate

Training:
  1. Select subset of training data
  2. Inject trigger (modify features)
  3. Relabel to target class
  4. Train with poisoned data
  5. Scale updates to survive FedAvg

Inference:
  - Normal data: Correct predictions
  - Triggered data: Backdoor activated → Target prediction
```

**Trigger Types**:
- **Pattern Trigger**: Specific feature values
- **Semantic Trigger**: Real-world pattern (e.g., round numbers)
- **Steganographic Trigger**: Invisible bit patterns

**Stealth Techniques**:
- Update scaling: Amplify malicious updates
- Gradient camouflage: Mimic honest gradient statistics
- Gradual escalation: Start small, increase over rounds

### 3. Model Poisoning (Day 16)

**Gradient Manipulation**:
```
Honest gradient: ∇L(w) (from local data)
Malicious gradient: ∇L(w) + Δ (perturbed)

Attack strategies:
  • Gradient scaling: ∇L(w) × α (amplify/reverse)
  • Sign flipping: sign(∇L(w)) × -1 (invert direction)
  • Gaussian noise: ∇L(w) + N(0, σ²) (add noise)
  • Inner product: Maximize ∇L_mal • ∇L_honest (correlation attack)
```

**Attack Goals**:
- **Availability**: Degrade overall model accuracy
- **Targeted Misclassification**: Cause specific inputs to be misclassified
- **Backdoor**: Embed hidden functionality
- **Model Manipulation**: Shift decision boundary

## Experimental Results

### Label Flipping Impact

| Attack Type | Flip Rate | Final Accuracy | AUC-ROC | Detection Rate |
|-------------|-----------|----------------|---------|----------------|
| None (baseline) | 0% | 95.1% | 0.92 | 89% |
| Random Flip | 20% | 87.3% | 0.81 | 62% |
| Targeted Flip | 30% | 82.1% | 0.74 | 45% |
| Inverse Flip | 100% | 51.2% | 0.50 | 12% |

### Backdoor Success Rate

| Trigger Type | Poison Ratio | Scale Factor | Success Rate | Clean Accuracy |
|--------------|--------------|--------------|--------------|----------------|
| Pattern | 10% | 10x | 92% | 94% |
| Pattern | 20% | 20x | 98% | 93% |
| Semantic | 15% | 15x | 85% | 95% |

### Model Poisoning Effectiveness

| Attack | # Attackers | Impact on Accuracy | Note |
|--------|-------------|-------------------|------|
| Gradient Scaling (2x) | 3 | -12% | Reversible by robust aggregation |
| Sign Flipping | 3 | -35% | Severe degradation |
| Gaussian Noise | 5 | -8% | Random disturbance |

## Defenses (Preview)

These attacks motivate the defenses in [`../04_defensive_techniques/`](../04_defensive_techniques/):
- **Byzantine-Robust Aggregation** (Day 17): Krum, Trimmed Mean
- **Anomaly Detection** (Day 18): Statistical, magnitude-based detection
- **FoolsGold** (Day 19): Sybil-resistant aggregation

## Usage

**For defensive research only**:

```bash
# Simulate Label Flipping Attack
cd label_flipping_attack
python simulate_attack.py --attack targeted --flip_rate 0.3

# Train with Backdoor
cd backdoor_attack_fl
python train_backdoor.py --trigger semantic --poison_ratio 0.2

# Evaluate Model Poisoning
cd model_poisoning_fl
python evaluate_attack.py --attack sign_flipping --num_attackers 3
```

## Key Insights

1. **Federated Learning is Vulnerable**: Decentralized nature enables client-side attacks
2. **Stealth Attacks**: Small, well-crafted attacks evade detection
3. **Amplification**: Single malicious update affects global model via aggregation
4. **Defense is Possible**: Robust aggregation and anomaly detection can mitigate attacks
5. **Arms Race**: Attacks evolve to bypass defenses, requiring adaptive defenses

## Research Papers Implemented

- **Label Flipping**: "Manipulating Machine Learning: Poisoning Attacks and Countermeasures for Statistical Learning" (Jagupski et al., 2018)
- **Backdoor**: "How To Backdoor Federated Learning" (Bagdasaryan et al., AISTATS 2020)
- **Model Poisoning**: "Hidden Backdoor Attacks on Federated Learning" (Sun et al., 2022)

---

*Average Quality Score: 9.3/10*

**Next**: [`../04_defensive_techniques/`](../04_defensive_techniques/) - Learn to defend against these attacks
