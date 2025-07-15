SHELL=/bin/bash

.PHONY: format
format: lock
	@echo "🚀 Fixing linting with Ruff"
	@uv run ruff check --fix
	@echo "🚀 Running formatting with Ruff"
	@uv run ruff format

.PHONY: check
check: lock
	@echo "🚀 Checking linting with Ruff"
	@uv run ruff check
	@echo "🚀 Checking formatting with Ruff"
	@uv run ruff format --check
#	@echo "🚀 Checking types with pyrefly"
#	@uv run pyrefly check src tests

.PHONY: test
test: lock
	@uv run pytest -vvv

.PHONY: lock
lock:
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked