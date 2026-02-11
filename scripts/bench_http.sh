#!/usr/bin/env bash
set -euo pipefail

PROMPT=${1:-$'Write a 3-line haiku about caching. Use vivid imagery.\nHaiku:\n'}
TOKENS_SPEC=${2:-60}                  # "60" or "16,32,64,128"
REPETITION_PENALTY=${3:-1.15}
NO_REPEAT_NGRAM_SIZE=${4:-3}
USE_KV_CACHE=${5:-true}

run_once () {
  local TOKENS="$1"
  curl -s -X POST http://127.0.0.1:8000/generate \
    -H "Content-Type: application/json" \
    -d "$(jq -nc \
      --arg prompt "$PROMPT" \
      --argjson max_new_tokens "$TOKENS" \
      --argjson repetition_penalty "$REPETITION_PENALTY" \
      --argjson no_repeat_ngram_size "$NO_REPEAT_NGRAM_SIZE" \
      --argjson use_kv_cache "$USE_KV_CACHE" \
      '{prompt:$prompt, max_new_tokens:$max_new_tokens, repetition_penalty:$repetition_penalty, no_repeat_ngram_size:$no_repeat_ngram_size, use_kv_cache:$use_kv_cache}')" \
  | jq -r --argjson tok "$TOKENS" '
      [
        $tok,
        .timing.prefill_s,
        .timing.decode_s,
        .timing.total_s,
        .timing.tokens_per_second,
        .generation.used_kv_cache
      ] | @tsv'
}

# Header
echo -e "tokens\tprefill_s\tdecode_s\ttotal_s\ttokens_per_second\tused_kv_cache"

IFS=',' read -ra TOK_ARR <<< "$TOKENS_SPEC"
for t in "${TOK_ARR[@]}"; do
  run_once "$t"
done
