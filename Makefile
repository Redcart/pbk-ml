.PHONY: pre-commit
pre-commit:
	@echo "Running pre-commit hooks..."
	@uv run pre-commit run --all-files
	@echo "pre-commit hooks completed."

.PHONY: tests
tests:
	@echo "Running tests..."
	@uv run pytest tests
	@echo "tests completed."

.PHONY: ruff
ruff:
	@echo "Running ruff..."
	@uv run ruff check .
	@echo "ruff completed."

.PHONY: install
install:
	@echo "Installing dependencies..."
	@uv sync --dev
	@echo "Dependencies installed."
