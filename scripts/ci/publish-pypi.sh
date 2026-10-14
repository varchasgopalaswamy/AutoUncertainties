#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYPI_TOKEN:-}" ]]; then
  echo "PYPI_TOKEN is required" >&2
  exit 1
fi

export UV_SYSTEM_PYTHON="${UV_SYSTEM_PYTHON:-1}"

rm -rf dist
uv sync --locked --group release
uv run --locked --group release python -m build
uv run --locked --group release twine check dist/*
TWINE_USERNAME=__token__ TWINE_PASSWORD="${PYPI_TOKEN}" \
  uv run --locked --group release twine upload --non-interactive dist/*
