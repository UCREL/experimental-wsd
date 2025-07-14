SHELL=/bin/bash

.PHONY: check
check:
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting with Ruff"
	@uv run ruff check --exit-zero
	@echo "🚀 Formatting with Ruff"
	@uv run ruff format
	@echo "🚀 Checking types with pyrefly"
	@uv run pyrefly check ./src

.PHONY: test
test:
	@uv run pytest -vvv