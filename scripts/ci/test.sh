#!/usr/bin/env bash
set -euo pipefail

export UV_SYSTEM_PYTHON="${UV_SYSTEM_PYTHON:-1}"

uv sync --locked --group ci

uv run --locked --group ci ruff check .
uv run --locked --group ci ruff format --check .
uv run --locked --group ci pyrefly check
uv run --locked --group ci geninit --check

uv run --locked --group ci pytest --cov --cov-report=xml --cov-report=html --cov-report=term
