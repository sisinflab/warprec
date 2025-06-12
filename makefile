.DEFAULT_GOAL := help
POETRY := $(shell command -v poetry 2> /dev/null)
INSTALL_STAMP := .install.stamp

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  install    - Install project dependencies"
	@echo "  lint       - Run code linters"
	@echo "  test       - Run tests"


.PHONY: install
install-cpu:
	@echo "🧠 Using Python 3.12 with poetry..."
	poetry env use python3.12
	poetry install

	@echo "🧠 Detecting torch version..."
	@torch_version=$$(poetry run python -c "import torch; print(torch.__version__)") && \
	echo "✅ Detected torch version: $$torch_version" && \
	echo "⬇ Installing PyG dependencies for torch==$$torch_version..." && \
	poetry run pip install \
		torch-scatter \
		torch-sparse \
		torch-cluster \
		torch-spline-conv \
		-f https://data.pyg.org/whl/torch-$$torch_version.html && \
	poetry add torch-geometric


# .PHONY: clean
# clean:
# 	find . -type d -name "__pycache__" | xargs rm -rf {};
# 	rm -rf .coverage .mypy_cache .pytest_cache .ruff_cache
# 	rm -rf dist build *.egg-info

.PHONY: lint
lint:
	poetry run pre-commit run -a

.PHONY: test
test:
	poetry run pytest --junit-xml=junit_result.xml --cov-report=xml:coverage.xml --cov=src

# .PHONY: build
# build: clean
# 	$(POETRY) build
