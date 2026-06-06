#!/usr/bin/env bash
set -euo pipefail

export UV_SYSTEM_PYTHON="${UV_SYSTEM_PYTHON:-1}"

uv sync
uv run --with build python -m build
