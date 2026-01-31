"""Setup configuration for Fraud Detection EDA Dashboard."""

from setuptools import find_packages, setup

setup(
    name="fraud-detection-dashboard",
    version="1.0.0",
    author="Your Name",
    description="Interactive EDA dashboard for credit card fraud detection",
    long_description=open("README.md").read() if False else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "dash>=2.14.0",
        "plotly>=5.18.0",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ]
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
