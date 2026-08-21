# FinSight developer convenience commands.
# Usage: make <target>
# Requires: pip, streamlit, pytest, docker (for docker-* targets).

.PHONY: install install-dev test eval run app docker-build docker-run clean

## install: install the package and all runtime dependencies
install:
	pip install -e .

## install-dev: install with dev extras (pytest etc.)
install-dev:
	pip install -e ".[dev]"

## test: run all unit tests (zero API calls — fully mocked/pure)
test:
	pytest tests/ -v

## eval: run the full eval suite against data/sample_statement.csv
eval:
	python -m evals.run_evals

## run: start the CLI (defaults to data/sample_statement.csv)
run:
	python -m finsight.main

## app: start the Streamlit UI (then open http://localhost:8501)
app:
	streamlit run app.py

## docker-build: build the Docker image tagged 'finsight'
docker-build:
	docker build -t finsight .

## docker-run: run the Docker image (reads API keys from .env)
docker-run:
	docker run --rm -p 8501:8501 --env-file .env finsight

## clean: remove Python bytecode and __pycache__ dirs
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
