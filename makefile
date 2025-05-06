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
install:
	poetry env use python3.12
	poetry install


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
