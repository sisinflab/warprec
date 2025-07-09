.DEFAULT_GOAL := help
POETRY := $(shell command -v poetry 2> /dev/null)
CONDA_EXE := $(shell command -v mamba 2>/dev/null || command -v conda 2>/dev/null)
PYTHON := python3.12
ENV_NAME := warprec
INSTALL_STAMP := .install.stamp

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  install    	- Install project dependencies"
	@echo "  install-pip    - Install dependencies via requirements.txt and venv"
	@echo "  install-conda	- Install dependencies via conda using environment.yml"
	@echo "  lint       	- Run code linters"
	@echo "  test       	- Run tests"


################################################################################
## Poetry install
.PHONY: install
install:
	@echo "🧠 Using Python 3.12 with poetry..."
	poetry env use python3.12
	poetry install

	@echo "🔍 Detecting torch version..."
	@torch_version_full=$$(poetry run python -c "import torch; print(torch.__version__)") && \
	torch_version_base=$$(echo $$torch_version_full | cut -d+ -f1) && \
	echo "✅ Detected torch version: $$torch_version_full" && \
	echo "⬇ Installing torch==$$torch_version_base via pip..." && \
	poetry run pip install torch==$$torch_version_base && \
	echo "⬇ Installing PyG dependencies for torch==$$torch_version_base with CUDA suffix..." && \
	poetry run pip install \
		torch-scatter \
		torch-sparse \
		torch-cluster \
		torch-spline-conv \
		-f https://data.pyg.org/whl/torch-$$torch_version_full.html && \
	poetry run pip install torch-geometric

	@echo "✅ Poetry environment created."


################################################################################
## Venv + pip install
.PHONY: install-venv
install-venv:
	@echo "🧪 Creating virtual env and installing via pip..."
	$(PYTHON) -m venv .venv
	@. .venv/bin/activate && \
		pip install --upgrade pip && \
		pip install -r requirements.txt && \
		echo "🔍 Detecting torch version..." && \
		torch_version_full=$$(python -c "import torch; print(torch.__version__)") && \
		torch_version_base=$$(echo $$torch_version_full | cut -d+ -f1) && \
		echo "✅ Detected torch version: $$torch_version_full" && \
		echo "⬇ Installing torch==$$torch_version_base via pip..." && \
		pip install torch==$$torch_version_base && \
		echo "⬇ Installing PyG dependencies for torch==$$torch_version_base with CUDA suffix..." && \
		pip install \
			torch-scatter \
			torch-sparse \
			torch-cluster \
			torch-spline-conv \
			-f https://data.pyg.org/whl/torch-$$torch_version_full.html && \
		pip install torch-geometric && \
		echo "✅ Venv environment created."


################################################################################
## Conda install
.PHONY: install-conda
install-conda:
ifndef CONDA_EXE
	$(error "ERROR: Neither mamba nor conda found in PATH")
endif
	@echo "🐍 Creating conda/mamba environment from environment.yml..."
	$(CONDA_EXE) env create --file environment.yml --name $(ENV_NAME)

	@echo "🔍 Detecting torch version..."
	torch_version_full=$$($(CONDA_EXE) run -n $(ENV_NAME) python -c "import torch; print(torch.__version__)") && \
	torch_version_base=$$(echo $$torch_version_full | cut -d+ -f1) && \
	echo "✅ Detected torch version: $$torch_version_full" && \
	echo "⬇ Installing torch==$$torch_version_base via pip..." && \
	$(CONDA_EXE) run -n $(ENV_NAME) pip install torch==$$torch_version_base && \
	echo "⬇ Installing PyG dependencies for torch==$$torch_version_full..." && \
	$(CONDA_EXE) run -n $(ENV_NAME) pip install \
		torch-scatter \
		torch-sparse \
		torch-cluster \
		torch-spline-conv \
		-f https://data.pyg.org/whl/torch-$$torch_version_full.html && \
	$(CONDA_EXE) run -n $(ENV_NAME) pip install torch-geometric

	@echo "✅ Environment '$(ENV_NAME)' created. Activate it with '$(CONDA_EXE) activate $(ENV_NAME)'"

.PHONY: lint
lint:
	poetry run pre-commit run -a

.PHONY: test
test:
	poetry run pytest --junit-xml=junit_result.xml --cov-report=xml:coverage.xml --cov=src
