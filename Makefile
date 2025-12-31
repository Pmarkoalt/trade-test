# Makefile for running tests independently
# 
# RECOMMENDED: Docker-based commands (consistent environment across all systems):
#   make docker-build            - Build Docker image
#   make docker-test-unit        - Run unit tests in Docker
#   make docker-test-integration - Run integration tests in Docker
#   make docker-test-validation  - Run validation suite in Docker
#   make docker-test-all         - Run all tests in Docker
#
# ALTERNATIVE: Native Python commands (use if you prefer not to use Docker):
#   make test-unit          - Run unit tests only
#   make test-integration   - Run integration tests only
#   make test-validation    - Run validation suite
#   make test-all           - Run unit + integration tests
#   make test               - Default: run unit tests

.PHONY: help test test-unit test-integration test-validation test-all test-coverage
.PHONY: docker-build docker-test-unit docker-test-integration docker-test-validation docker-test-all
.PHONY: docker-debug-shell docker-debug-integration docker-debug-test docker-test-integration-verbose
.PHONY: docker-test-integration-capture docker-test-integration-last-failed
.PHONY: install-dev format format-check lint type-check check-code

# Default test config for validation suite
TEST_CONFIG ?= tests/fixtures/configs/run_test_config.yaml

# Docker image name
DOCKER_IMAGE ?= trading-system:latest

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help:
	@echo "RECOMMENDED: Docker-based test targets (consistent environment):"
	@echo "  make docker-build           - Build Docker image"
	@echo "  make docker-test-unit       - Run unit tests in Docker"
	@echo "  make docker-test-integration - Run integration tests in Docker"
	@echo "  make docker-test-validation - Run validation suite in Docker"
	@echo "  make docker-test-all        - Run all tests in Docker"
	@echo ""
	@echo "ALTERNATIVE: Native Python test targets:"
	@echo "  make test              - Run unit tests (default)"
	@echo "  make test-unit         - Run unit tests only"
	@echo "  make test-integration  - Run integration tests only"
	@echo "  make test-validation   - Run validation suite"
	@echo "  make test-all          - Run unit + integration tests"
	@echo "  make test-coverage     - Run tests with coverage report"
	@echo ""
	@echo "Docker debugging targets:"
	@echo "  make docker-debug-shell              - Open interactive shell in Docker container"
	@echo "  make docker-debug-integration        - Run integration tests with full debugging"
	@echo "  make docker-debug-test TEST=<name>   - Run specific test with debugging"
	@echo "  make docker-test-integration-verbose - Run with very verbose output"
	@echo "  make docker-test-integration-capture  - Run and capture output to file"
	@echo "  make docker-test-integration-last-failed - Re-run only failed tests"
	@echo ""
	@echo "Options:"
	@echo "  TEST_CONFIG=<path>     - Override validation config (default: tests/fixtures/configs/run_test_config.yaml)"
	@echo "  TEST=<name>            - Specific test to run (for docker-debug-test)"
	@echo "  pytest args            - Pass additional args: make test-unit ARGS='-v -k test_name'"
	@echo ""
	@echo "Code quality targets:"
	@echo "  make install-dev       - Install development dependencies (black, isort, flake8, mypy)"
	@echo "  make format            - Format code with black and isort"
	@echo "  make format-check      - Check formatting without making changes"
	@echo "  make lint              - Lint code with flake8"
	@echo "  make type-check        - Type check code with mypy"
	@echo "  make check-code        - Run all code quality checks"

test: test-unit
	@echo "$(GREEN)✓ Unit tests complete$(NC)"

test-unit:
	@echo "$(YELLOW)Running unit tests...$(NC)"
	@echo "========================================="
	pytest tests/ \
		--ignore=tests/integration \
		--ignore=tests/performance \
		--ignore=tests/property \
		-v \
		--tb=short \
		$(ARGS)
	@echo "$(GREEN)✓ Unit tests complete$(NC)"

test-integration:
	@echo "$(YELLOW)Running integration tests...$(NC)"
	@echo "========================================="
	pytest tests/integration/ \
		-v \
		--tb=short \
		$(ARGS)
	@echo "$(GREEN)✓ Integration tests complete$(NC)"

test-validation:
	@echo "$(YELLOW)Running validation suite...$(NC)"
	@echo "========================================="
	@if [ ! -f "$(TEST_CONFIG)" ]; then \
		echo "$(RED)✗ Config file not found: $(TEST_CONFIG)$(NC)"; \
		echo "   Please specify a valid config with: make test-validation TEST_CONFIG=<path>"; \
		exit 1; \
	fi
	python -m trading_system validate --config $(TEST_CONFIG)
	@echo "$(GREEN)✓ Validation suite complete$(NC)"

test-all: test-unit test-integration
	@echo "$(GREEN)✓ All tests complete$(NC)"

test-coverage:
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	@echo "========================================="
	pytest tests/ \
		--cov=trading_system \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-report=xml \
		-v \
		--tb=short \
		$(ARGS)
	@echo "$(GREEN)✓ Coverage report generated$(NC)"
	@echo "   HTML report: htmlcov/index.html"

# Docker-based test commands
docker-build:
	@echo "$(YELLOW)Building Docker image...$(NC)"
	@echo "========================================="
	docker-compose build
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-test-unit:
	@echo "$(YELLOW)Running unit tests in Docker...$(NC)"
	@echo "========================================="
	docker-compose run --rm --entrypoint pytest trading-system tests/ \
		--ignore=tests/integration \
		--ignore=tests/performance \
		--ignore=tests/property \
		-v \
		--tb=short \
		$(ARGS)
	@echo "$(GREEN)✓ Unit tests complete$(NC)"

docker-test-integration:
	@echo "$(YELLOW)Running integration tests in Docker...$(NC)"
	@echo "========================================="
	docker-compose run --rm --entrypoint pytest trading-system tests/integration/ \
		-v \
		--tb=short \
		$(ARGS)
	@echo "$(GREEN)✓ Integration tests complete$(NC)"

docker-test-validation:
	@echo "$(YELLOW)Running validation suite in Docker...$(NC)"
	@echo "========================================="
	@if [ ! -f "$(TEST_CONFIG)" ]; then \
		echo "$(RED)✗ Config file not found: $(TEST_CONFIG)$(NC)"; \
		echo "   Please specify a valid config with: make docker-test-validation TEST_CONFIG=<path>"; \
		exit 1; \
	fi
	docker-compose run --rm trading-system validate --config /app/$(TEST_CONFIG)
	@echo "$(GREEN)✓ Validation suite complete$(NC)"

docker-test-all: docker-test-unit docker-test-integration
	@echo "$(GREEN)✓ All tests complete$(NC)"

# Docker debugging commands
docker-debug-shell:
	@echo "$(YELLOW)Opening interactive shell in Docker container...$(NC)"
	@echo "$(YELLOW)Use this to debug integration test failures interactively$(NC)"
	@echo "========================================="
	docker-compose run --rm trading-system /bin/bash

docker-debug-integration:
	@echo "$(YELLOW)Running integration tests with full debugging...$(NC)"
	@echo "========================================="
	docker-compose run --rm --entrypoint pytest trading-system tests/integration/ \
		-vvv \
		--tb=long \
		--showlocals \
		--capture=no \
		--pdb \
		$(ARGS)
	@echo "$(GREEN)✓ Debug session complete$(NC)"

docker-debug-test:
	@if [ -z "$(TEST)" ]; then \
		echo "$(RED)✗ TEST parameter required$(NC)"; \
		echo "   Usage: make docker-debug-test TEST=test_name"; \
		echo "   Example: make docker-debug-test TEST=test_integration_workflow"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Running test '$(TEST)' with full debugging...$(NC)"
	@echo "========================================="
	docker-compose run --rm --entrypoint pytest trading-system tests/integration/ \
		-vvv \
		--tb=long \
		--showlocals \
		--capture=no \
		-k "$(TEST)" \
		$(ARGS)
	@echo "$(GREEN)✓ Debug session complete$(NC)"

docker-test-integration-verbose:
	@echo "$(YELLOW)Running integration tests with very verbose output...$(NC)"
	@echo "========================================="
	docker-compose run --rm --entrypoint pytest trading-system tests/integration/ \
		-vvv \
		--tb=long \
		--showlocals \
		--log-cli-level=DEBUG \
		$(ARGS)
	@echo "$(GREEN)✓ Integration tests complete$(NC)"

docker-test-integration-capture:
	@echo "$(YELLOW)Running integration tests and capturing output...$(NC)"
	@echo "========================================="
	@mkdir -p results
	docker-compose run --rm --entrypoint pytest trading-system tests/integration/ \
		-vvv \
		--tb=long \
		--showlocals \
		$(ARGS) 2>&1 | tee results/integration_test_output_$$(date +%Y%m%d_%H%M%S).txt
	@echo "$(GREEN)✓ Integration tests complete$(NC)"
	@echo "   Output saved to results/integration_test_output_*.txt"

docker-test-integration-last-failed:
	@echo "$(YELLOW)Re-running only failed integration tests...$(NC)"
	@echo "========================================="
	docker-compose run --rm --entrypoint pytest trading-system tests/integration/ \
		-v \
		--tb=short \
		--lf \
		--failed-first \
		$(ARGS)
	@echo "$(GREEN)✓ Failed tests re-run complete$(NC)"

# Code Quality Targets

install-dev:
	@echo "$(YELLOW)Installing development dependencies...$(NC)"
	pip install -e ".[dev]"
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"
	@echo ""
	@echo "Installed tools:"
	@echo "  • black (code formatter)"
	@echo "  • isort (import sorter)"
	@echo "  • flake8 (linter)"
	@echo "  • mypy (type checker)"
	@echo ""
	@echo "Next steps:"
	@echo "  • Format code: make format"
	@echo "  • Check code quality: make check-code"

format:
	@echo "$(YELLOW)Formatting code with black...$(NC)"
	@if command -v black > /dev/null 2>&1; then \
		black trading_system/ tests/; \
	else \
		echo "$(RED)✗ black not found. Install with: make install-dev$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Sorting imports with isort...$(NC)"
	@if command -v isort > /dev/null 2>&1; then \
		isort trading_system/ tests/; \
	else \
		echo "$(RED)✗ isort not found. Install with: make install-dev$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Code formatted$(NC)"

format-check:
	@echo "$(YELLOW)Checking code formatting...$(NC)"
	@if command -v black > /dev/null 2>&1; then \
		black --check trading_system/ tests/ || (echo "$(RED)✗ Code formatting issues found$(NC)" && exit 1); \
	else \
		echo "$(RED)✗ black not found. Install with: make install-dev$(NC)"; \
		exit 1; \
	fi
	@if command -v isort > /dev/null 2>&1; then \
		isort --check-only trading_system/ tests/ || (echo "$(RED)✗ Import sorting issues found$(NC)" && exit 1); \
	else \
		echo "$(RED)✗ isort not found. Install with: make install-dev$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Code formatting OK$(NC)"

lint:
	@echo "$(YELLOW)Linting code with flake8...$(NC)"
	@if command -v flake8 > /dev/null 2>&1; then \
		flake8 trading_system/ tests/ || (echo "$(RED)✗ Linting issues found$(NC)" && exit 1); \
	else \
		echo "$(RED)✗ flake8 not found. Install with: make install-dev$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Linting passed$(NC)"

type-check:
	@echo "$(YELLOW)Type checking with mypy...$(NC)"
	@if command -v mypy > /dev/null 2>&1; then \
		mypy trading_system/ || echo "$(YELLOW)⚠ Type checking issues found (warnings are OK)$(NC)"; \
	else \
		echo "$(RED)✗ mypy not found. Install with: make install-dev$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Type checking complete$(NC)"

check-code: format-check lint type-check
	@echo "$(GREEN)✓ All code quality checks passed$(NC)"

