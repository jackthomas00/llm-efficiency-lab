#!/usr/bin/env python3
"""Concurrency benchmark: 8 parallel HTTP requests to /generate.
Reports wall time, p50/p95 latency, avg latency, and aggregate tokens/sec.
"""
from __future__ import annotations

import asyncio
import statistics
import time

import httpx

URL = "http://127.0.0.1:8000/generate"
PROMPT = "Write a 3-line haiku about caching. Use vivid imagery.\nHaiku:\n"
MAX_NEW_TOKENS = 64
N_REQUESTS = 8


async def generate(client: httpx.AsyncClient, i: int) -> tuple[float, int]:
    """Single request. Returns (latency_s, tokens_generated)."""
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
    return latency, tokens


async def main():
    t_wall_start = time.perf_counter()

    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [generate(client, i) for i in range(N_REQUESTS)]
        results = await asyncio.gather(*tasks)

    t_wall = time.perf_counter() - t_wall_start

    latencies = [r[0] for r in results]
    total_tokens = sum(r[1] for r in results)

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    p50 = latencies_sorted[int(0.50 * (n - 1))] if n > 0 else 0
    p95 = latencies_sorted[int(0.95 * (n - 1))] if n > 1 else latencies_sorted[-1]

    avg_latency = statistics.mean(latencies) if latencies else 0
    tokens_per_sec = total_tokens / t_wall if t_wall > 0 else 0

    print("=== 8 parallel /generate requests ===")
    print(f"Total wall time:    {t_wall:.3f}s")
    print(f"Avg latency:        {avg_latency:.3f}s")
    print(f"P50 latency:        {p50:.3f}s")
    print(f"P95 latency:        {p95:.3f}s")
    print(f"Total tokens:       {total_tokens}")
    print(f"Throughput:         {tokens_per_sec:.1f} tokens/sec (aggregate)")


if __name__ == "__main__":
    asyncio.run(main())
