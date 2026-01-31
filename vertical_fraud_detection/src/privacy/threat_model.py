"""
Threat model documentation for Vertical Federated Learning.

Documents privacy assumptions and threat models for the VFL system.
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class ThreatModel:
    """Threat model for VFL system."""

    name: str
    description: str
    capabilities: List[str]
    limitations: List[str]
    mitigations: List[str]


class ThreatModels:
    """Collection of threat models for VFL."""

    @staticmethod
    def honest_but_curious() -> ThreatModel:
        """
        Honest-but-curious (semi-honest) adversary.

        Follows protocol but attempts to learn additional information
        from received messages.
        """
        return ThreatModel(
            name="Honest-but-Curious",
            description=(
                "Adversary follows the VFL protocol correctly but attempts "
                "to extract additional information from the data they receive "
                "(embeddings, gradients)."
            ),
            capabilities=[
                "Observe all messages received during protocol",
                "Perform computations on received data",
                "Attempt to reconstruct embeddings from gradients",
                "Analyze patterns in communication"
            ],
            limitations=[
                "Cannot deviate from protocol",
                "Cannot inject false data",
                "Cannot modify other parties' models",
                "Cannot access raw features directly"
            ],
            mitigations=[
                "Use differential privacy on gradients",
                "Limit granularity of gradients shared",
                "Add noise to embeddings before transmission",
                "Use secure aggregation protocols"
            ]
        )

    @staticmethod
    def malicious_server() -> ThreatModel:
        """
        Malicious server adversary.

       May deviate from protocol to maximize information leakage.
        """
        return ThreatModel(
            name="Malicious Server",
            description=(
                "Server may deviate from protocol to extract maximum "
                "information from client embeddings and gradients."
            ),
            capabilities=[
                "All honest-but-curious capabilities",
                "Send manipulated gradients to clients",
                "Request additional information",
                "Attempt model inversion attacks"
            ],
            limitations=[
                "Still cannot access raw features directly",
                "Clients can detect anomalous gradients",
                "Protocol can include verification steps"
            ],
            mitigations=[
                "Gradient norm verification",
                "Protocol validation checks",
                "Rate limiting on gradient requests",
                "Client-side anomaly detection"
            ]
        )

    @staticmethod
    def colluding_parties() -> ThreatModel:
        """
        Colluding parties adversary.

        Multiple parties collude to share information and reconstruct
        more data.
        """
        return ThreatModel(
            name="Colluding Parties",
            description=(
                "Two or more parties share their received information "
                "to reconstruct more of the global data."
            ),
            capabilities=[
                "Combine gradients from multiple parties",
                "Cross-reference embedding patterns",
                "Potentially reconstruct more features"
            ],
            limitations=[
                "Cannot directly access other parties' raw features",
                "Each party only sees their own raw data"
            ],
            mitigations=[
                "Limit number of parties per training round",
                "Use cryptographic protocols",
                "Add party-specific noise"
            ]
        )


def document_threat_model(save_path: str = None) -> str:
    """
    Generate comprehensive threat model documentation.

    Args:
        save_path: Optional path to save documentation

    Returns:
        Markdown documentation string
    """
    doc = """# Privacy Analysis: Vertical Federated Learning for Fraud Detection

## System Architecture

```
Party A (Bank A)          Party B (Bank B)          Server
│                        │                        │
│  Bottom Model A         │  Bottom Model B         │  Top Model
│  (Transaction)          │  (Credit)               │  (Classifier)
│                        │                        │
└────────────────────────┴────────────────────────┘
         │                              │
         │  1. Forward: Embeddings      │
         ├─────────────────────────────>│
         │                              │
         │  2. Backward: Gradients      │
         │<─────────────────────────────┤
         │                              │
```

## Data Flow Analysis

### Forward Pass
1. **Party A**: Computes `z_a = BottomModel_A(x_a)`
   - Input: Raw transaction features `x_a` (STAYS LOCAL)
   - Output: Embedding `z_a` (SENT TO SERVER)

2. **Party B**: Computes `z_b = BottomModel_B(x_b)`
   - Input: Raw credit features `x_b` (STAYS LOCAL)
   - Output: Embedding `z_b` (SENT TO SERVER)

3. **Server**: Computes `y = TopModel([z_a, z_b])`
   - Input: Concatenated embeddings
   - Output: Prediction

**Privacy Guarantee**: Raw features `x_a`, `x_b` never leave local devices.

### Backward Pass
1. **Server**: Computes `dL/dz_a`, `dL/dz_b`
   - Gradients with respect to embeddings
   - SENT TO RESPECTIVE PARTIES

2. **Party A**: Computes `dL/dθ_a = dL/dz_a × dz_a/dθ_a`
   - Updates bottom model parameters locally

3. **Party B**: Computes `dL/dθ_b = dL/dz_b × dz_b/dθ_b`
   - Updates bottom model parameters locally

**Privacy Guarantee**: Only embedding gradients transmitted, not raw gradients.

---

## Threat Models

"""

    # Add threat models
    threat_models = [
        ThreatModels.honest_but_curious(),
        ThreatModels.malicious_server(),
        ThreatModels.colluding_parties(),
    ]

    for i, tm in enumerate(threat_models, 1):
        doc += f"### {i}. {tm.name}\n\n"
        doc += f"**Description**: {tm.description}\n\n"
        doc += "**Capabilities**:\n"
        for cap in tm.capabilities:
            doc += f"- {cap}\n"
        doc += "\n**Limitations**:\n"
        for lim in tm.limitations:
            doc += f"- {lim}\n"
        doc += "\n**Mitigations**:\n"
        for mit in tm.mitigations:
            doc += f"- {mit}\n"
        doc += "\n"

    # Add privacy properties
    doc += """
---

## Privacy Properties

### What IS Shared
| Data | From Party A | From Party B | From Server |
|------|--------------|--------------|-------------|
| **Forward** | Embeddings z_a | Embeddings z_b | Predictions y |
| **Backward** | Gradients dL/dz_a | Gradients dL/dz_b | None |

### What is NOT Shared
| Data | Status |
|------|--------|
| Raw features x_a (Party A) | ✅ NEVER LEAVES Party A |
| Raw features x_b (Party B) | ✅ NEVER LEAVES Party B |
| Bottom model parameters θ_a | ✅ KEPT SECRET |
| Bottom model parameters θ_b | ✅ KEPT SECRET |
| Top model parameters θ_top | ✅ KEPT SECRET |
| dL/dθ (raw parameter gradients) | ✅ NEVER TRANSMITTED |

---

## Gradient Leakage Analysis

### Attack Description
An adversary (e.g., honest-but-curious server) receives `dL/dz` (gradient
with respect to embeddings) and may attempt to reconstruct the embeddings `z`.

### Risk Quantification
We quantify leakage risk by measuring correlation between embeddings and
their gradients:
- **Low Risk**: < 15% correlation
- **Medium Risk**: 15-30% correlation
- **High Risk**: > 30% correlation

### Mitigation Strategies
1. **Gradient Noise**: Add Gaussian noise to gradients (DP-SGD)
2. **Gradient Clipping**: Limit gradient magnitude
3. **Embedding Dimension**: Use higher-dimensional embeddings
4. **Secure Aggregation**: Encrypt gradients during transmission

---

## Comparison: Vertical FL vs Other Approaches

| Method | Feature Privacy | Model Privacy | Communication |
|--------|-----------------|---------------|---------------|
| **Vertical FL** | ✅ High | ⚠️ Medium | Embeddings + Gradients |
| **Horizontal FL** | ✅ High | ✅ High | Model parameters only |
| **Centralized** | ❌ None | ❌ None | Raw data to server |
| **Single-Party** | ✅ High | ✅ High | None |

---

## Recommendations

### For Production Deployment
1. **Implement Gradient Noise**: Add calibrated Gaussian noise to gradients
2. **Monitor Leakage**: Track gradient-embedding correlation during training
3. **Rate Limiting**: Limit frequency of gradient exchanges
4. **Audit Logging**: Log all gradient transmissions for audit trails
5. **Anomaly Detection**: Detect suspicious gradient patterns

### For Future Research
1. **Homomorphic Encryption**: Encrypt embeddings before transmission
2. **Secure Multi-Party Computation**: Compute predictions without revealing embeddings
3. **Differential Privacy**: Formal DP guarantees on gradient leakage
4. **Federated Dropout**: Randomly drop embedding dimensions for privacy

---

## References

1. Romanini, et al. "Private federated learning on vertically partitioned data"
2. Vepakomma, et al. "Split Learning for Collaborative Deep Learning"
3. Zhu, et al. " Leakage of Gradient in Vertical Federated Learning"

---

*This analysis assumes the honest-but-curious threat model as baseline.
Stronger threat models require additional cryptographic protections.*
"""

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(doc)
        print(f"Threat model documentation saved to: {save_path}")

    return doc


if __name__ == "__main__":
    # Generate threat model documentation
    doc = document_threat_model("docs/threat_model.md")
    print("\n" + "="*80)
    print("THREAT MODEL DOCUMENTATION")
    print("="*80)
    print(doc[:500] + "...\n")
    print("✓ Threat model documentation generated")
