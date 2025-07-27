# Makefile for Agent Mesh Federated Runtime

# =============================================================================
# CONFIGURATION
# =============================================================================

# Project configuration
PROJECT_NAME := agent-mesh-federated-runtime
VERSION := 1.0.0
PYTHON_VERSION := 3.11

# Docker configuration
DOCKER_REGISTRY := ghcr.io/your-org
DOCKER_IMAGE := $(DOCKER_REGISTRY)/$(PROJECT_NAME)
DOCKER_TAG := $(VERSION)

# Directories
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs
DIST_DIR := dist
BUILD_DIR := build

# Python configuration
PYTHON := python3
PIP := pip3
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy

# Docker Compose profiles
COMPOSE_PROFILES := development,testing

# =============================================================================
# HELP TARGET
# =============================================================================

.PHONY: help
help: ## Show this help message
	@echo "Agent Mesh Federated Runtime - Makefile Commands"
	@echo "=================================================="
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Environment Variables:"
	@echo "  DOCKER_REGISTRY  - Docker registry (default: $(DOCKER_REGISTRY))"
	@echo "  DOCKER_TAG       - Docker tag (default: $(DOCKER_TAG))"
	@echo "  PYTHON_VERSION   - Python version (default: $(PYTHON_VERSION))"

# =============================================================================
# DEVELOPMENT TARGETS
# =============================================================================

.PHONY: install
install: ## Install dependencies for development
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"

.PHONY: install-prod
install-prod: ## Install production dependencies only
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .

.PHONY: dev
dev: ## Start development environment
	docker-compose --profile development up -d
	@echo "Development environment started!"
	@echo "API: http://localhost:8000"
	@echo "Grafana: http://localhost:3000 (admin/admin123)"
	@echo "Jupyter: http://localhost:8888 (token: mesh_notebook_token)"

.PHONY: dev-logs
dev-logs: ## Show development environment logs
	docker-compose --profile development logs -f

.PHONY: dev-down
dev-down: ## Stop development environment
	docker-compose --profile development down

.PHONY: shell
shell: ## Open development shell in container
	docker-compose exec dev-node bash

# =============================================================================
# CODE QUALITY TARGETS
# =============================================================================

.PHONY: format
format: ## Format code with black and isort
	$(BLACK) $(SRC_DIR) $(TEST_DIR)
	$(ISORT) $(SRC_DIR) $(TEST_DIR)

.PHONY: lint
lint: ## Run code linting
	$(FLAKE8) $(SRC_DIR) $(TEST_DIR)
	$(MYPY) $(SRC_DIR)

.PHONY: format-check
format-check: ## Check code formatting
	$(BLACK) --check $(SRC_DIR) $(TEST_DIR)
	$(ISORT) --check-only $(SRC_DIR) $(TEST_DIR)

.PHONY: quality
quality: format-check lint ## Run all code quality checks

.PHONY: fix
fix: format lint ## Fix code formatting and linting issues

# =============================================================================
# TESTING TARGETS
# =============================================================================

.PHONY: test
test: ## Run all tests
	$(PYTEST) $(TEST_DIR) -v

.PHONY: test-unit
test-unit: ## Run unit tests only
	$(PYTEST) $(TEST_DIR)/unit -v

.PHONY: test-integration
test-integration: ## Run integration tests only
	$(PYTEST) $(TEST_DIR)/integration -v

.PHONY: test-e2e
test-e2e: ## Run end-to-end tests
	$(PYTEST) $(TEST_DIR)/e2e -v

.PHONY: test-performance
test-performance: ## Run performance benchmarks
	$(PYTEST) $(TEST_DIR)/performance -v --benchmark-only

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term

.PHONY: test-watch
test-watch: ## Run tests in watch mode
	$(PYTEST)-watch $(TEST_DIR)

.PHONY: test-docker
test-docker: ## Run tests in Docker
	docker-compose --profile testing run --rm test-runner

# =============================================================================
# SECURITY TARGETS
# =============================================================================

.PHONY: security
security: ## Run security checks
	bandit -r $(SRC_DIR)
	safety check

.PHONY: security-audit
security-audit: ## Run comprehensive security audit
	bandit -r $(SRC_DIR) -f json -o security-report.json
	safety check --json --output security-deps.json
	@echo "Security reports generated: security-report.json, security-deps.json"

# =============================================================================
# BUILD TARGETS
# =============================================================================

.PHONY: build
build: ## Build Python package
	$(PYTHON) -m build

.PHONY: build-docker
build-docker: ## Build Docker image
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	docker build -t $(DOCKER_IMAGE):latest .

.PHONY: build-all-stages
build-all-stages: ## Build all Docker stages
	docker build --target development -t $(DOCKER_IMAGE):dev .
	docker build --target testing -t $(DOCKER_IMAGE):test .
	docker build --target production -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	docker build --target edge -t $(DOCKER_IMAGE):edge .
	docker build --target gpu -t $(DOCKER_IMAGE):gpu .

.PHONY: build-multi-arch
build-multi-arch: ## Build multi-architecture Docker images
	docker buildx build --platform linux/amd64,linux/arm64 \
		-t $(DOCKER_IMAGE):$(DOCKER_TAG) \
		-t $(DOCKER_IMAGE):latest \
		--push .

# =============================================================================
# DOCUMENTATION TARGETS
# =============================================================================

.PHONY: docs
docs: ## Build documentation
	sphinx-build -b html $(DOCS_DIR) $(DOCS_DIR)/_build/html

.PHONY: docs-serve
docs-serve: docs ## Build and serve documentation
	$(PYTHON) -m http.server 8080 --directory $(DOCS_DIR)/_build/html

.PHONY: docs-clean
docs-clean: ## Clean documentation build
	rm -rf $(DOCS_DIR)/_build

.PHONY: docs-api
docs-api: ## Generate API documentation
	sphinx-apidoc -o $(DOCS_DIR)/api $(SRC_DIR)

.PHONY: docs-watch
docs-watch: ## Watch and rebuild documentation
	sphinx-autobuild $(DOCS_DIR) $(DOCS_DIR)/_build/html

# =============================================================================
# DEPLOYMENT TARGETS
# =============================================================================

.PHONY: deploy-local
deploy-local: ## Deploy locally with Docker Compose
	docker-compose up -d

.PHONY: deploy-dev
deploy-dev: ## Deploy to development environment
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

.PHONY: deploy-staging
deploy-staging: ## Deploy to staging environment
	@echo "Deploying to staging..."
	# kubectl apply -f k8s/staging/

.PHONY: deploy-prod
deploy-prod: ## Deploy to production environment
	@echo "Deploying to production..."
	# kubectl apply -f k8s/production/

.PHONY: k8s-deploy
k8s-deploy: ## Deploy to Kubernetes
	kubectl apply -f k8s/

.PHONY: k8s-delete
k8s-delete: ## Delete Kubernetes deployment
	kubectl delete -f k8s/

# =============================================================================
# MONITORING TARGETS
# =============================================================================

.PHONY: monitor
monitor: ## Start monitoring stack
	docker-compose up -d prometheus grafana

.PHONY: logs
logs: ## Show application logs
	docker-compose logs -f

.PHONY: status
status: ## Show system status
	docker-compose ps
	@echo ""
	@echo "Health Checks:"
	@curl -f http://localhost:8080/health 2>/dev/null || echo "Bootstrap node: DOWN"
	@curl -f http://localhost:8081/health 2>/dev/null || echo "Node 1: DOWN"
	@curl -f http://localhost:8082/health 2>/dev/null || echo "Node 2: DOWN"

.PHONY: metrics
metrics: ## Show metrics endpoint
	@echo "Prometheus metrics available at:"
	@echo "http://localhost:9090"
	@echo "http://localhost:9091"
	@echo "http://localhost:9092"

# =============================================================================
# DATABASE TARGETS
# =============================================================================

.PHONY: db-start
db-start: ## Start database services
	docker-compose up -d postgres redis

.PHONY: db-migrate
db-migrate: ## Run database migrations
	alembic upgrade head

.PHONY: db-reset
db-reset: ## Reset database
	docker-compose down postgres
	docker volume rm $(PROJECT_NAME)_postgres_data
	docker-compose up -d postgres
	sleep 5
	$(MAKE) db-migrate

.PHONY: db-backup
db-backup: ## Backup database
	docker-compose exec postgres pg_dump -U agent_mesh agent_mesh > backup_$(shell date +%Y%m%d_%H%M%S).sql

# =============================================================================
# CLEANUP TARGETS
# =============================================================================

.PHONY: clean
clean: ## Clean build artifacts
	rm -rf $(BUILD_DIR) $(DIST_DIR) *.egg-info
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov .pytest_cache .mypy_cache

.PHONY: clean-docker
clean-docker: ## Clean Docker images and containers
	docker-compose down -v
	docker system prune -f
	docker volume prune -f

.PHONY: clean-all
clean-all: clean clean-docker docs-clean ## Clean everything

# =============================================================================
# UTILITY TARGETS
# =============================================================================

.PHONY: requirements
requirements: ## Generate requirements.txt
	pip-compile pyproject.toml

.PHONY: upgrade-deps
upgrade-deps: ## Upgrade dependencies
	pip-compile --upgrade pyproject.toml

.PHONY: check-deps
check-deps: ## Check for dependency vulnerabilities
	safety check

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

.PHONY: install-hooks
install-hooks: ## Install git hooks
	pre-commit install

.PHONY: benchmark
benchmark: ## Run performance benchmarks
	$(PYTHON) scripts/benchmark.py

.PHONY: profile
profile: ## Profile application performance
	$(PYTHON) -m cProfile -o profile.stats $(SRC_DIR)/main.py
	$(PYTHON) scripts/profile_analyzer.py profile.stats

# =============================================================================
# RELEASE TARGETS
# =============================================================================

.PHONY: release-dry
release-dry: ## Dry run release
	semantic-release publish --dry-run

.PHONY: release
release: ## Create release
	semantic-release publish

.PHONY: tag
tag: ## Create git tag
	git tag -a v$(VERSION) -m "Release version $(VERSION)"
	git push origin v$(VERSION)

# =============================================================================
# CI/CD TARGETS
# =============================================================================

.PHONY: ci-test
ci-test: ## Run CI test suite
	$(PYTEST) $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=xml --junitxml=test-results.xml

.PHONY: ci-quality
ci-quality: ## Run CI quality checks
	$(BLACK) --check $(SRC_DIR) $(TEST_DIR)
	$(ISORT) --check-only $(SRC_DIR) $(TEST_DIR)
	$(FLAKE8) $(SRC_DIR) $(TEST_DIR)
	$(MYPY) $(SRC_DIR)

.PHONY: ci-security
ci-security: ## Run CI security checks
	bandit -r $(SRC_DIR) -f json -o bandit-report.json
	safety check --json --output safety-report.json

.PHONY: ci-build
ci-build: ## CI build process
	$(MAKE) ci-quality
	$(MAKE) ci-security
	$(MAKE) ci-test
	$(MAKE) build

# =============================================================================
# DEVELOPMENT UTILITIES
# =============================================================================

.PHONY: notebook
notebook: ## Start Jupyter notebook
	docker-compose --profile development up -d jupyter
	@echo "Jupyter notebook available at: http://localhost:8888"
	@echo "Token: mesh_notebook_token"

.PHONY: debug
debug: ## Start debug environment
	docker-compose -f docker-compose.yml -f docker-compose.debug.yml up -d

.PHONY: load-test
load-test: ## Run load tests
	docker-compose --profile testing run --rm load-tester

.PHONY: chaos-test
chaos-test: ## Run chaos engineering tests
	$(PYTHON) scripts/chaos_test.py

# =============================================================================
# NETWORK UTILITIES
# =============================================================================

.PHONY: network-create
network-create: ## Create test network with multiple nodes
	docker-compose up -d bootstrap-node mesh-node-1 mesh-node-2 mesh-node-3

.PHONY: network-scale
network-scale: ## Scale the mesh network
	docker-compose up -d --scale mesh-node-1=3

.PHONY: network-test
network-test: ## Test network connectivity
	$(PYTHON) scripts/network_test.py

# =============================================================================
# SPECIAL TARGETS
# =============================================================================

.DEFAULT_GOAL := help

# Make sure we can run multiple targets in parallel
.NOTPARALLEL:

# Declare phony targets
.PHONY: all install install-prod dev dev-logs dev-down shell format lint \
	format-check quality fix test test-unit test-integration test-e2e \
	test-performance test-coverage test-watch test-docker security \
	security-audit build build-docker build-all-stages build-multi-arch \
	docs docs-serve docs-clean docs-api docs-watch deploy-local deploy-dev \
	deploy-staging deploy-prod k8s-deploy k8s-delete monitor logs status \
	metrics db-start db-migrate db-reset db-backup clean clean-docker \
	clean-all requirements upgrade-deps check-deps pre-commit install-hooks \
	benchmark profile release-dry release tag ci-test ci-quality ci-security \
	ci-build notebook debug load-test chaos-test network-create network-scale \
	network-test help