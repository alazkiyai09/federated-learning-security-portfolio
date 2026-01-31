.PHONY: help install test demo clean lint format

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	pip install -r requirements.txt
	pip install -e .

demo: ## Run the demo script
	python demo.py

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=html

lint: ## Run linting
	flake8 src/ tests/ app/
	mypy src/

format: ## Format code with black
	black src/ tests/ app/

clean: ## Clean generated files
	rm -rf reports/*.html
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

streamlit: ## Run Streamlit app
	streamlit run app/streamlit_app.py
