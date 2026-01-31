# Changelog

All notable changes to SignGuard will be documented in this file.

## [0.1.0] - 2024-01-28 - Initial Release

### Added
- Cryptographic authentication (ECDSA signatures)
- Multi-factor anomaly detection (L2 norm, cosine similarity, loss deviation)
- Dynamic reputation system (time-decay with bonus/penalty)
- Reputation-weighted aggregation
- Three attack implementations (label flip, backdoor, model poisoning)
- Four baseline defenses (Krum, Trimmed Mean, FoolsGold, Bulyan)
- Complete FL simulation framework
- 78 passing tests (69% coverage)
- Paper reproduction scripts (all tables and figures)
- Comprehensive documentation

### Security Features
- ECDSA signature verification
- Byzantine resilience up to f < n/3
- Adaptive anomaly detection
- Reputation-based client filtering

### Performance
- Comparable or better accuracy than baselines
- 67% reduction in attack success rate
- Acceptable computational overhead (<2x)
- Linear communication complexity

### Known Limitations
- Computation overhead from signature verification
- Key management complexity
- Requires honest majority assumption

---

## [Unreleased]

### Planned Features
- Zero-knowledge proof integration
- TEE integration for enhanced security
- Support for non-IID data distributions
- Adaptive anomaly thresholds
- Distributed key management

### Bug Fixes
- Fixed signature serialization issues
- Improved reputation convergence
- Enhanced detector robustness

---

For version history, see git commit log.
