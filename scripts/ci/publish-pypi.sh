#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYPI_TOKEN:-}" ]]; then
  echo "PYPI_TOKEN is required" >&2
  exit 1
fi

export UV_SYSTEM_PYTHON="${UV_SYSTEM_PYTHON:-1}"

rm -rf dist
uv sync
uv run --with build python -m build
uv run --with twine twine upload --verbose -u __token__ -p "${PYPI_TOKEN}" dist/*
