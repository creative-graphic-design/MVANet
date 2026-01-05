.PHONY: format
format:
	uv run ruff format --check --diff .

.PHONY: lint
lint:
	uv run ruff check --output-format=github .

.PHONY: typecheck
typecheck:
	uv run mypy .

.PHONY: test
test:
	uv run pytest -vs
