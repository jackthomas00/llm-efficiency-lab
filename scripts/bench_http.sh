#!/usr/bin/env bash
set -euo pipefail

PROMPT=${1:-"Write a 3-line haiku about caching. Use vivid imagery.\nHaiku:\n"}
TOKENS=${2:-60}

curl -s -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"${PROMPT//\"/\\\"}\", \"max_new_tokens\": ${TOKENS} }" | jq
