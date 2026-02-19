.PHONY: help install install-dev install-test lint format type-check test test-unit test-integration coverage clean run

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*
$$
' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n",
$$
1, $\$2}'

install: ## Install production dependencies
	python -m pip install --upgrade pip
	pip install -r requirements/base.txt
	pip install -e .

install-dev: ## Install development dependencies
	python -m pip install --upgrade pip
	pip install -r requirements/dev.txt
	pip install -e .

install-test: ## Install test dependencies
	python -m pip install --upgrade pip
	pip install -r requirements/test.txt
	pip install -e .

lint: ## Run linter
	ruff check src/ tests/

format: ## Format code
	ruff format src/ tests/
	ruff check --fix src/ tests/

type-check: ## Run type checker
	mypy src/

test: ## Run all tests
	pytest

test-unit: ## Run unit tests only
	pytest -m unit

test-integration: ## Run integration tests only
	pytest -m integration

coverage: ## Run tests with coverage
	pytest --cov=agent_system --cov-report=html --cov-report=term-missing

clean: ## Clean build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf dist build .mypy_cache .pytest_cache .ruff_cache htmlcov .coverage

run: ## Run the application
	python -m agent_system.main