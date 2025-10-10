.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  lint       	- Run code linters"
	@echo "  test       	- Run tests"

.PHONY: lint
lint:
	poetry run pre-commit run -a

.PHONY: test
test:
	poetry run pytest --junit-xml=junit_result.xml --cov-report=xml:coverage.xml --cov=src
