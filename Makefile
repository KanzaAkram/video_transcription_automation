.PHONY: install install-worker install-all dry-run api test lint fmt typecheck clean

install:              ## Tests + dry run. No ML dependencies, any Python 3.11+
	pip install -e ".[dev]"

install-worker:       ## Adds ASR and embeddings. Needs Python 3.11-3.13 + ffmpeg
	pip install -e ".[dev,asr,embeddings]"

install-all:          ## Everything
	pip install -e ".[all]"

dry-run:              ## End-to-end pipeline with a fake Drive and no credentials
	python scripts/dry_run.py

api:                  ## Webhook receiver + admin API
	uvicorn pipeline.api.main:app --host 0.0.0.0 --port 8080 --reload

test:
	pytest -q

lint:
	ruff check src tests scripts

fmt:
	ruff format src tests scripts && ruff check --fix src tests scripts

typecheck:
	mypy

clean:
	rm -rf .dryrun .pytest_cache .ruff_cache .mypy_cache .runs
