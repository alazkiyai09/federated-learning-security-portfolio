"""Setup configuration for SignGuard package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="signguard",
    version="0.1.0",
    author="Researcher",
    author_email="researcher@university.edu",
    description="Cryptographic Signature-Based Defense for Federated Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/signguard",
    packages=find_packages(exclude=["tests", "experiments", "scripts", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "cryptography>=41.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.65.0",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.4.0",
            "flake8>=6.0.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "signguard-train=experiments.main:main",
        ],
    },
)
