# Developer task runner. Mirrors the CI quality gate (.github/workflows/ci.yml).
# Usage: `make check` before pushing.
.PHONY: install lint format typecheck test test-all check clean reproduce-thesis

install:  ## editable install with dev + deep-learning extras
	pip install -e ".[dev,dl]"

lint:  ## static lint (ruff)
	ruff check src scripts tests

format:  ## apply formatting (ruff format) — not gated; adopt gradually
	ruff format src scripts tests

typecheck:  ## type-check the library (pyright, basic mode)
	pyright

test:  ## data-free test suite (runs anywhere)
	pytest -m "not requires_data"

test-all:  ## full suite incl. data-dependent tests (needs data/)
	pytest

check: lint typecheck test  ## everything CI runs

reproduce-thesis:  ## full multi-seed pipeline over the canonical thesis seeds (needs data/; see REPRODUCING.md)
	python -m src.modeling.orchestration.multi_seed

clean:  ## remove caches and build artefacts
	python -c "import shutil,glob,os; [shutil.rmtree(p,ignore_errors=True) for p in glob.glob('**/__pycache__',recursive=True)+['.pytest_cache','.ruff_cache','build','dist']]"
