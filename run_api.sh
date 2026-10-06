#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec uvicorn inference_api:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
