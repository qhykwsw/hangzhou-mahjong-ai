#!/usr/bin/env bash
set -euo pipefail
MAHJONG_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MAHJONG_REPO_ROOT"
unset CONTROL_PLANE_API_KEY OPENAI_API_KEY OPENAI_ADMIN_KEY
exec "$MAHJONG_REPO_ROOT/.venv/bin/python" -m mahjong_ai.project_mcp "$@"
