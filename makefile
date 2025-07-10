.DEFAULT_GOAL := help
POETRY := $(shell command -v poetry 2> /dev/null)
CONDA_EXE := $(shell command -v mamba 2>/dev/null || command -v conda 2>/dev/null)
PYTHON := python3.12
ENV_NAME := warprec
INSTALL_STAMP := .install.stamp

# This is the fixed version of torch for WarpRec
TORCH_VERSION := 2.6.0

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  install-poetry - Install project dependencies"
	@echo "  install-venv   - Install dependencies via requirements.txt and venv"
	@echo "  install-conda	- Install dependencies via conda using environment.yml"
	@echo "  lint       	- Run code linters"
	@echo "  test       	- Run tests"


################################################################################
## Poetry install
.PHONY: install-poetry
install-poetry:
	@echo "🧠 Using Python 3.12 with poetry..."
	poetry env use ${PYTHON}
	poetry install

	@echo "🔍 Detecting CUDA version..."
	@CUDA_SUFFIX="cpu"; \
	if command -v nvcc &> /dev/null; then \
		CUDA_VERSION=$$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/'); \
		echo "✅ Found CUDA version: $$CUDA_VERSION"; \
		case "$$CUDA_VERSION" in \
			"11.8") CUDA_SUFFIX="cu118";; \
			"12.1") CUDA_SUFFIX="cu121";; \
			"12.4") CUDA_SUFFIX="cu124";; \
			"12.6") CUDA_SUFFIX="cu126";; \
			*) \
				echo "❌ ERROR: Unsupported CUDA version '$$CUDA_VERSION'. Supported versions are 11.8, 12.1, 12.4, 12.6."; \
				exit 1;; \
		esac; \
	else \
		echo "⚠️ WARNING: 'nvcc' not found. Installing CPU version of PyTorch."; \
	fi; \
	\
	TORCH_INSTALL_VERSION="$(TORCH_VERSION)+$$CUDA_SUFFIX"; \
	PYG_URL="https://data.pyg.org/whl/torch-$$TORCH_INSTALL_VERSION.html"; \
	\
	echo "⬇️ Installing torch==$(TORCH_VERSION) for $$CUDA_SUFFIX..."; \
	poetry run pip install torch==$(TORCH_VERSION) --index-url https://download.pytorch.org/whl/$$CUDA_SUFFIX; \
	\
	echo "⬇️ Installing PyG dependencies from $$PYG_URL..."; \
	poetry run pip install \
		torch-scatter \
		torch-sparse \
		torch-cluster \
		torch-spline-conv \
	-f $$PYG_URL; \
	\
	poetry run pip install torch-geometric torchmetrics; \
	echo "✅ PyTorch and PyG dependencies installed successfully."


################################################################################
## Venv + pip install
.PHONY: install-venv
install-venv:
	@echo "🧪 Creating virtual env and installing base dependencies..."
	$(PYTHON) -m venv .venv
	@. .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt && \
	\
	echo "🔍 Detecting CUDA version for venv environment..." && \
	CUDA_SUFFIX="cpu"; \
	if command -v nvcc &> /dev/null; then \
		CUDA_VERSION=$$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/'); \
		echo "✅ Found CUDA version: $$CUDA_VERSION"; \
		case "$$CUDA_VERSION" in \
			"11.8") CUDA_SUFFIX="cu118";; \
			"12.1") CUDA_SUFFIX="cu121";; \
			"12.4") CUDA_SUFFIX="cu124";; \
			"12.6") CUDA_SUFFIX="cu126";; \
			*) \
				echo "❌ ERROR: Unsupported CUDA version '$$CUDA_VERSION'. Supported versions are 11.8, 12.1, 12.4, 12.6."; \
				exit 1;; \
		esac; \
	else \
		echo "⚠️ WARNING: 'nvcc' not found. Installing CPU version of PyTorch."; \
	fi; \
	\
	TORCH_INSTALL_VERSION="$(TORCH_VERSION)+$$CUDA_SUFFIX"; \
	PYG_URL="https://data.pyg.org/whl/torch-$$TORCH_INSTALL_VERSION.html"; \
	\
	echo "⬇️ Installing torch==$(TORCH_VERSION) for $$CUDA_SUFFIX..."; \
	pip install --no-cache-dir torch==$(TORCH_VERSION) --index-url https://download.pytorch.org/whl/$$CUDA_SUFFIX; \
	\
	echo "⬇️ Installing PyG dependencies from $$PYG_URL..."; \
	pip install --no-cache-dir \
		torch-scatter \
		torch-sparse \
		torch-cluster \
		torch-spline-conv \
	-f $$PYG_URL; \
	\
	echo "⬇️ Installing torch-geometric and torchmetrics..."; \
	pip install --no-cache-dir torch-geometric torchmetrics; \
	\
	echo "✅ PyTorch ecosystem installed successfully for venv."


################################################################################
## Conda install
.PHONY: install-conda
install-conda:
ifndef CONDA_EXE
	$(error "ERROR: Neither mamba nor conda found in PATH")
endif
	@echo "🐍 Creating conda/mamba environment from environment.yml..."
	$(CONDA_EXE) env create --file environment.yml --name $(ENV_NAME) || $(CONDA_EXE) env update --file environment.yml --name $(ENV_NAME)

	@echo "🔍 Detecting CUDA version for Conda environment..."
	@CUDA_SUFFIX="cpu"; \
	if command -v nvcc &> /dev/null; then \
		CUDA_VERSION=$$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/'); \
		echo "✅ Found CUDA version: $$CUDA_VERSION"; \
		case "$$CUDA_VERSION" in \
			"11.8") CUDA_SUFFIX="cu118";; \
			"12.1") CUDA_SUFFIX="cu121";; \
			"12.4") CUDA_SUFFIX="cu124";; \
			"12.6") CUDA_SUFFIX="cu126";; \
			*) \
				echo "❌ ERROR: Unsupported CUDA version '$$CUDA_VERSION'. Supported versions are 11.8, 12.1, 12.4, 12.6."; \
				exit 1;; \
		esac; \
	else \
		echo "⚠️ WARNING: 'nvcc' not found. Installing CPU version of PyTorch."; \
	fi; \
	\
	TORCH_INSTALL_VERSION="$(TORCH_VERSION)+$$CUDA_SUFFIX"; \
	PYG_URL="https://data.pyg.org/whl/torch-$$TORCH_INSTALL_VERSION.html"; \
	\
	echo "⬇️ Installing torch==$(TORCH_VERSION) for $$CUDA_SUFFIX..."; \
	$(CONDA_EXE) run -n $(ENV_NAME) pip install --no-cache-dir torch==$(TORCH_VERSION) --index-url https://download.pytorch.org/whl/$$CUDA_SUFFIX; \
	\
	echo "⬇️ Installing PyG dependencies from $$PYG_URL..."; \
	$(CONDA_EXE) run -n $(ENV_NAME) pip install --no-cache-dir \
		torch-scatter \
		torch-sparse \
		torch-cluster \
		torch-spline-conv \
	-f $$PYG_URL; \
	\
	echo "⬇️ Installing torch-geometric and torchmetrics..."; \
	$(CONDA_EXE) run -n $(ENV_NAME) pip install --no-cache-dir torch-geometric torchmetrics; \
	\
	echo "✅ PyTorch ecosystem installed successfully for Conda. Activate with: $(CONDA_EXE) activate $(ENV_NAME)"

.PHONY: lint
lint:
	poetry run pre-commit run -a

.PHONY: test
test:
	poetry run pytest --junit-xml=junit_result.xml --cov-report=xml:coverage.xml --cov=src
