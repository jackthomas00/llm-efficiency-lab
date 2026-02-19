# Week 3 Handoff: Microbatching -> Batched Decode

## Scope
Week 3 focused on moving from async queue-based microbatching scaffolding to real batched KV-cache decode.

## Environment
- Model: `distilgpt2`
- Device: CPU
- Server: FastAPI `POST /generate`
- Worker model: single Uvicorn worker

## Starting Point (before Week 3.2)
- Async request queue and batch collection existed:
  - `max_batch_size=8`
  - `batch_timeout_ms=10`
- Batched prefill was implemented.
- Decode was still effectively serialized (`for item in batch: decode(...)`).
- Observed behavior under 8 concurrent requests:
  - similar completion times per request
  - aggregate throughput worse than single-request KV decode

## What Was Implemented

### 1) True batched KV decode (step-wise)
Implemented `decode_batch_with_kv(...)` in `InferenceRunner`:
- One shared model forward per decode step for active requests.
- Per-request sampling and history penalties (repetition / no-repeat ngram).
- Per-request output assembly and result packaging.

File:
- `src/efflab/engine/runner.py`

### 2) Batcher integration for KV requests
Updated `Batcher` loop to use batched decode for KV requests:
- KV requests are grouped by compatible generation params:
  - `max_new_tokens`
  - `temperature`
  - `repetition_penalty`
  - `no_repeat_ngram_size`
- Each subgroup uses one batched decode run.
- Non-KV requests still use existing per-request decode path.

File:
- `src/efflab/engine/batcher.py`

### 3) Attention-mask timing fix in batched decode
Adjusted mask update timing:
- Extend `attention_mask` at end of each decode iteration (after sampling), so next step forward sees correct `past_len + 1` semantics.
- Important for padded batch prompts with different prompt lengths.

File:
- `src/efflab/engine/runner.py`

### 4) Cache API modernization (`transformers` Cache)
Removed dependence on passing legacy tuple cache into model forward:
- Forward calls now use cache objects directly.
- Added split/stack helpers for per-item and batched cache handling:
  - `_split_past_key_values(...)`
  - `_stack_past_key_values(...)`
- Added compatibility fallback for legacy tuple-style cache for non-HF/test model paths.

File:
- `src/efflab/engine/runner.py`

## Bug Fixes During Integration
- Fixed missing cache symbol definitions/imports (`Cache`, `DynamicCache`) and legacy-cache type guard helper.
- Updated batched prefill state extraction to use cache splitting helper instead of direct tuple slicing.

File:
- `src/efflab/engine/runner.py`

## Validation Completed
- Static compile check passed:
  - `.venv/bin/python -m compileall -q src scripts`
- Tests passed:
  - `.venv/bin/pytest -q`
  - Result: `2 passed`

## Benchmark Status
- The concurrent bench script is ready:
  - `scripts/bench_concurrent.py`
- In this sandbox session, live bench could not be completed because model startup attempted Hugging Face download and DNS/network was unavailable.
- Re-run locally with model artifacts available/cached to validate Week 3 throughput gains.

## How To Run Locally
1. Start server:
   - `.venv/bin/python -m uvicorn efflab.engine.server_fastapi:app --host 127.0.0.1 --port 8000`
2. Run concurrent benchmark:
   - `.venv/bin/python scripts/bench_concurrent.py`

## Current Constraints (intentional for this phase)
- Batched decode currently assumes compatible generation params for batching subgroup.
- Early EOS removal from active set is not yet implemented (all stay active through `max_new_tokens` in current simplification).
- Single-worker scheduling model.

## Recommended Week 3.3 Follow-ups
1. Add early-EOS masking/removal in batched decode to avoid wasted compute.
2. Add regression test for batched KV decode equivalence under deterministic settings (`temperature=0`).
3. Add benchmark table in repo docs with:
   - wall time
   - p50/p95 latency
   - aggregate tokens/sec
   across batch sizes/concurrency levels.
4. Add lightweight metrics (batch size, decode step count, queue wait) to server responses/logs for easier profiling.
