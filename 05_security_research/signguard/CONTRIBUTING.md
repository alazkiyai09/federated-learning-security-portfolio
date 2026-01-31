# Contributing to SignGuard

Thank you for your interest in contributing to SignGuard! This document provides guidelines for contributing.

---

## Code of Conduct

Please be respectful and inclusive in all interactions.

---

## Development Workflow

### 1. Fork and Clone

```bash
# Fork the repository
# Clone your fork
git clone https://github.com/YOUR_USERNAME/signguard.git
cd signguard
git remote add upstream https://github.com/ORIGINAL_OWNER/signguard.git
```

### 2. Create Branch

```bash
# For new features
git checkout -b feature/your-feature-name

# For bug fixes
git checkout -b fix/your-bug-fix
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
pre-commit install
```

### 4. Make Changes

- Write code following the style guide
- Add tests for new functionality
- Update documentation as needed

### 5. Test Your Changes

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=signguard --cov-report=html
```

### 6. Commit Changes

```bash
git add .
git commit -m "Add your feature"
```

### 7. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
# Then create PR on GitHub
```

---

## Coding Standards

### Style Guide

We follow:
- **PEP 8** for Python code
- **Black** for code formatting
- **isort** for import sorting
- **Google Style Guide** for docstrings

### Type Hints

Add type hints to all public functions:
```python
def compute_accuracy(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
) -> float:
    """Compute model accuracy on test set.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to run on
        
    Returns:
        Accuracy as float [0, 1]
    """
    pass
```

### Documentation

- All public classes should have docstrings
- Complex functions should have examples
- Use Google-style docstrings

### Testing

- Write tests for all new functionality
- Aim for >80% coverage
- Use descriptive test names

```python
def test_signature_verification_with_tampered_update():
    """Test that signature verification detects tampering."""
    # Setup
    sm = SignatureManager()
    update = create_update()
    
    # Sign
    private_key, public_key = sm.generate_keypair()
    signature = sm.sign_update(update, private_key)
    
    # Tamper with update
    update.num_samples = 999
    
    # Verify should fail
    signed_update = SignedUpdate(update, signature, "key")
    assert sm.verify_update(signed_update) is False
```

---

## Project Structure

### Core Library (`signguard/`)

- **core/****: Client, server, type definitions
- **crypto/**: ECDSA signatures
- **detection/**: Anomaly detectors
- **reputation/**: Reputation systems
- **aggregation/**: Aggregation methods
- **attacks/**: Attack implementations
- **defenses/**: Baseline defenses
- **utils/**: Utilities

### Experiments (`experiments/`)

- All paper figures and tables
- Configuration files
- Results caching

### Tests (`tests/`)

- Unit tests for each module
- Integration tests for full FL pipeline
- 78 tests passing currently

---

## Pull Request Guidelines

### Title Format

```
[Feature] Add new defense mechanism
[Fix] Resolve signature verification bug
[Docs] Update README
```

### Description

- Describe what you changed and why
- Reference related issues
- Include screenshots for UI changes if applicable

### Checklist

- [ ] Tests pass locally
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] All commits are squashed if needed
- [ ] No merge conflicts

---

## Questions?

Feel free to:
- Open an issue for discussion
- Ask for clarification in PR
- Email: `researcher@university.edu`

---

**Thank you for contributing to SignGuard!**
