FUNCTIONS ?= ./data/input/functions_definition.json
INPUT ?= ./data/input/function_calling_tests.json
OUTPUT ?= ./data/output/function_calls.json

install:
	pip install uv
	uv sync

run:
	uv run python -m src --functions_definition $(FUNCTIONS) --input $(INPUT) --output $(OUTPUT)

debug:
	uv run python -m pdb -m src --functions_definition $(FUNCTIONS) --input $(INPUT) --output $(OUTPUT)

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 .
	mypy . --strict
