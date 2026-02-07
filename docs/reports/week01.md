# Week 01 — Baseline single-request engine

## Goal
Get a working end-to-end path: load model → prefill → decode → API response.

## What works
- FastAPI server with /health and /generate
- Prefill and decode loop split (baseline, no KV cache)
- Basic timing: prefill_s, decode_s, tokens_per_second

## Known limitations (intentional)
- Decode re-feeds full context every step (slow baseline)
- No batching
- No KV-cache reuse
- No streaming

## Next week
- Introduce use_cache=True and store past_key_values
- Implement decode_step that appends one token without recomputing full prompt
- Add basic request queue + round-robin scheduling
