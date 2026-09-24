.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  install-hooks	- Install the commit and push git hooks"
	@echo "  lint       	- Run code linters"
	@echo "  test       	- Run tests"
	@echo "  coverage   	- Run tests and report coverage (fails under 80%)"

# pylint and mypy run on push rather than on commit, so the push hook has to
# be installed as well as the commit one.
.PHONY: install-hooks
install-hooks:
	poetry run pre-commit install --hook-type pre-commit --hook-type pre-push

.PHONY: lint
lint:
	poetry run pre-commit run -a
	poetry run pre-commit run -a --hook-stage pre-push

.PHONY: test
test:
	poetry run pytest tests/ -q

.PHONY: coverage
coverage:
	poetry run pytest tests/ -q --cov=warprec --cov-report=term-missing
