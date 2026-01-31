.PHONY: help install test clean run-benchmark

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run tests"
	@echo "  make clean         - Clean generated files"
	@echo "  make run-benchmark - Run the benchmark"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Lint code with flake8"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=html

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	rm -rf .pytest_cache htmlcov .coverage

run-benchmark:
	python src/train.py --data data/raw/creditcard.csv --results-dir data/results

format:
	black src/ tests/

lint:
	flake8 src/ tests/ --max-line-length=100
