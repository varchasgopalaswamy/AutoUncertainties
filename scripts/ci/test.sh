#!/usr/bin/env bash
set -euo pipefail

export UV_SYSTEM_PYTHON="${UV_SYSTEM_PYTHON:-1}"

uv sync --extra CI

# Run ruff to check for linting issues
uv run ruff check .

uv run coverage run -m pytest --cov --cov-report=xml --cov-report=html --cov-report=term
