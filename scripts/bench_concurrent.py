#!/usr/bin/env python3
"""Concurrency benchmark: 8 parallel HTTP requests to /generate.
Reports wall time, p50/p95 latency, avg latency, and aggregate tokens/sec.
"""
from __future__ import annotations

import asyncio
from collections import Counter
import statistics
import time

import httpx

URL = "http://127.0.0.1:8000/generate"
PROMPT = "Write a 3-line haiku about caching. Use vivid imagery.\nHaiku:\n"
MAX_NEW_TOKENS = 64
N_REQUESTS = 8


def _pctl(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int(q * (len(xs) - 1))
    return xs[idx]


def _summary(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (
        f"n={len(values)} "
        f"avg={statistics.mean(values):.3f} "
        f"p50={_pctl(values, 0.50):.3f} "
        f"p95={_pctl(values, 0.95):.3f} "
        f"min={min(values):.3f} "
        f"max={max(values):.3f}"
    )


async def generate(client: httpx.AsyncClient, i: int) -> tuple[int, float, int, dict]:
    """Single request. Returns (request_idx, latency_s, tokens_generated, batching)."""
    t0 = time.perf_counter()
    resp = await client.post(
        URL,
        json={
            "prompt": PROMPT,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 1,
            "use_kv_cache": True,
        },
    )
    resp.raise_for_status()
    latency = time.perf_counter() - t0
    data = resp.json()
    tokens = data["generation"]["generated_new_tokens"]
    return i, latency, tokens, data["batching"]


async def main() -> None:
    t_wall_start = time.perf_counter()

    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [generate(client, i) for i in range(N_REQUESTS)]
        results = await asyncio.gather(*tasks)

    t_wall = time.perf_counter() - t_wall_start

    latencies = [r[1] for r in results]
    total_tokens = sum(r[2] for r in results)
    batching_metrics = [r[3] for r in results]
    tokens_per_sec = total_tokens / t_wall if t_wall > 0 else 0

    queue_wait_ms = [m["queue_wait_ms"] for m in batching_metrics]
    batch_sizes = [m["batch_size"] for m in batching_metrics]
    batch_formation_ms = [m["batch_formation_time"] * 1000.0 for m in batching_metrics]
    active_sizes_flat = [x for m in batching_metrics for x in m["active_batch_size_per_decode_step"]]
    tokens_per_step_flat = [x for m in batching_metrics for x in m["tokens_generated_per_step"]]
    step_time_ms_flat = [x * 1000.0 for m in batching_metrics for x in m["time_per_decode_step"]]

    batch_size_hist = Counter(batch_sizes)
    batch_size_hist_text = ", ".join(
        f"{size}:{count}" for size, count in sorted(batch_size_hist.items(), key=lambda kv: kv[0])
    )

    print("=== 8 parallel /generate requests ===")
    print(f"Total wall time:    {t_wall:.3f}s")
    print(f"Latency (s):        {_summary(latencies)}")
    print(f"Total tokens:       {total_tokens}")
    print(f"Throughput:         {tokens_per_sec:.1f} tokens/sec (aggregate)")
    print("")
    print("Batching metrics (across request responses):")
    print(f"Queue wait (ms):    {_summary(queue_wait_ms)}")
    print(f"Batch size:         avg={statistics.mean(batch_sizes):.2f} dist={batch_size_hist_text}")
    print(f"Batch form (ms):    {_summary(batch_formation_ms)}")
    print(f"Active batch/step:  {_summary([float(x) for x in active_sizes_flat])}")
    print(f"Tokens/step:        {_summary([float(x) for x in tokens_per_step_flat])}")
    print(f"Step time (ms):     {_summary(step_time_ms_flat)}")
    print("")
    print("Per-request snapshot:")
    for req_idx, latency_s, tokens, metrics in sorted(results, key=lambda r: r[0]):
        q_ms = metrics["queue_wait_ms"]
        bsz = metrics["batch_size"]
        step_count = len(metrics["time_per_decode_step"])
        print(
            f"  req={req_idx:02d} "
            f"lat={latency_s:.3f}s "
            f"tokens={tokens:3d} "
            f"q_wait={q_ms:7.2f}ms "
            f"batch={bsz} "
            f"steps={step_count}"
        )


if __name__ == "__main__":
    asyncio.run(main())
