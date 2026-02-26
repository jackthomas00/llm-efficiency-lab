#!/usr/bin/env python3
"""Scheduling experiment: sweep batch_timeout_ms, max_batch_size, short-job-first.
Optimizes latency vs throughput tradeoff.

Deliverable table: | batch_timeout | max_batch | short_job_first | p50 | p95 | tokens/sec |
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import time

from efflab.common.config import ModelConfig
from efflab.engine.model_loader import load_model_and_tokenizer
from efflab.engine.runner import InferenceRunner
from efflab.engine.batcher import Batcher, WorkItem

PROMPT = "Write a 3-line haiku about caching. Use vivid imagery.\nHaiku:\n"
MAX_NEW_TOKENS = 64
N_REQUESTS = 8


def _pctl(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int(q * (len(xs) - 1))
    return xs[idx]


async def _run_config(
    runner: InferenceRunner,
    *,
    batch_timeout_ms: int,
    max_batch_size: int,
    enable_short_job_first: bool,
    n_requests: int = N_REQUESTS,
) -> tuple[float, float, float]:
    """Run n_requests concurrent requests, return (p50_s, p95_s, tokens_per_sec)."""
    batcher = Batcher(
        runner,
        max_batch_size=max_batch_size,
        batch_timeout_ms=batch_timeout_ms,
        enable_short_job_first=enable_short_job_first,
        enable_round_robin=True,
    )
    await batcher.start()

    loop = asyncio.get_running_loop()
    t_wall_start = time.perf_counter()

    async def one_request(i: int) -> tuple[int, float, int]:
        fut = loop.create_future()
        item = WorkItem(
            prompt=PROMPT,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=1,
            use_kv_cache=True,
            future=fut,
        )
        await batcher.enqueue(item)
        result = await fut
        latency = time.perf_counter() - item.enqueued_at_s
        tokens = result["generation"]["generated_new_tokens"]
        return i, latency, tokens

    tasks = [one_request(i) for i in range(n_requests)]
    results = await asyncio.gather(*tasks)
    t_wall = time.perf_counter() - t_wall_start

    await batcher.stop()

    latencies = [r[1] for r in results]
    total_tokens = sum(r[2] for r in results)
    tokens_per_sec = total_tokens / t_wall if t_wall > 0 else 0.0

    return _pctl(latencies, 0.50), _pctl(latencies, 0.95), tokens_per_sec


async def _run_sweep(
    runner: InferenceRunner,
    configs: list[tuple[int, int, bool]],
    n_requests: int,
) -> list[tuple[int, int, bool, float, float, float]]:
    rows: list[tuple[int, int, bool, float, float, float]] = []
    for batch_timeout_ms, max_batch_size, short_job_first in configs:
        print(f"  batch_timeout={batch_timeout_ms} max_batch={max_batch_size} short_job_first={short_job_first}...", end=" ", flush=True)
        p50, p95, tps = await _run_config(
            runner,
            batch_timeout_ms=batch_timeout_ms,
            max_batch_size=max_batch_size,
            enable_short_job_first=short_job_first,
            n_requests=n_requests,
        )
        rows.append((batch_timeout_ms, max_batch_size, short_job_first, p50, p95, tps))
        print(f"p50={p50:.3f}s p95={p95:.3f}s {tps:.1f} tok/s")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheduling sweep: batch_timeout, max_batch, short_job_first")
    parser.add_argument(
        "--batch-timeout",
        type=int,
        nargs="+",
        default=[5, 10, 25, 50],
        help="batch_timeout_ms values to sweep",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="max_batch_size values to sweep",
    )
    parser.add_argument(
        "--short-job-first",
        type=str,
        nargs="+",
        default=["true", "false"],
        help="enable_short_job_first: true and/or false",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=N_REQUESTS,
        help="Number of concurrent requests per config",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID (default: from ModelConfig)",
    )
    args = parser.parse_args()
    n_requests = args.requests
    short_job_first_vals = [s.lower() == "true" for s in args.short_job_first]

    cfg = ModelConfig()
    if args.model:
        cfg.model_id = args.model

    print("Loading model...")
    model, tokenizer, cfg = load_model_and_tokenizer(cfg)
    runner = InferenceRunner(model, tokenizer, cfg.device)

    configs = list(
        itertools.product(
            args.batch_timeout,
            args.max_batch,
            short_job_first_vals,
        )
    )

    rows = asyncio.run(_run_sweep(runner, configs, n_requests))

    # Print deliverable table
    sweep_sjf = len(short_job_first_vals) > 1
    print()
    if sweep_sjf:
        print("| batch_timeout | max_batch | short_job_first | p50 | p95 | tokens/sec |")
        print("|--------------|----------|----------------|-----|-----|------------|")
        for batch_timeout_ms, max_batch_size, short_job_first, p50, p95, tps in rows:
            print(f"| {batch_timeout_ms:12} | {max_batch_size:8} | {str(short_job_first):14} | {p50:.3f} | {p95:.3f} | {tps:10.1f} |")
    else:
        print("| batch_timeout | max_batch | p50 | p95 | tokens/sec |")
        print("|--------------|----------|-----|-----|------------|")
        for batch_timeout_ms, max_batch_size, _, p50, p95, tps in rows:
            print(f"| {batch_timeout_ms:12} | {max_batch_size:8} | {p50:.3f} | {p95:.3f} | {tps:10.1f} |")


if __name__ == "__main__":
    main()
