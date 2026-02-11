# Week 02 — KV cache and last-token decode

## What changed
- **KV cache**: `use_cache=True`; store and reuse `past_key_values` across decode steps.
- **Last-token feeding**: Decode step only feeds the new token (and past KV), not the full context.

## Why it’s faster
Avoids redoing attention over the full context on every step. Prefill runs once; decode only attends over the new token + cached keys/values.

## Benchmark (distilgpt2, CPU)

**Week 2 — KV cache on** (`used_kv_cache: true`)

| max_new_tokens | prefill_s | decode_s | total_s | tokens_per_second |
|----------------|-----------|----------|---------|-------------------|
| 16             | 0.128     | 0.620    | 0.748   | 21.38             |
| 32             | 0.087     | 1.224    | 1.311   | 24.41             |
| 64             | 0.097     | 2.561    | 2.658   | 24.08             |
| 128            | 0.108     | 5.279    | 5.387   | 23.76             |

**Week 1 baseline — KV cache off** (`used_kv_cache: false`)

| max_new_tokens | prefill_s | decode_s | total_s | tokens_per_second |
|----------------|-----------|----------|---------|-------------------|
| 16             | 0.103     | 1.254    | 1.357   | 11.79             |
| 32             | 0.097     | 2.575    | 2.672   | 11.98             |
| 64             | 0.115     | 6.040    | 6.155   | 10.40             |
| 128            | 0.085     | 5.463    | 5.549   | 11.35             |

KV cache gives ~2× higher TPS across token counts.

## Known limitations (still)
- No batching
- No streaming
- Single request only
