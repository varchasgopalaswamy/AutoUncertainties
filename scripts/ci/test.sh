#!/usr/bin/env bash
set -euo pipefail

export UV_SYSTEM_PYTHON="${UV_SYSTEM_PYTHON:-1}"

uv sync --extra CI

# Run ruff to check for linting issues
uv tool install "ruff==0.15.16"
ruff check .

uv run coverage run -m pytest --cov --cov-report=xml --cov-report=html --cov-report=term
