#!/usr/bin/env python3
"""Multi-worker scaling benchmark: measure tokens/sec, p50/p95 latency, CPU % across uvicorn --workers 1,2,4."""
from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import threading
import time

import httpx
import psutil

PROMPT = "Write a 3-line haiku about caching. Use vivid imagery.\nHaiku:\n"
MAX_NEW_TOKENS = 64
N_REQUESTS = 8
HEALTH_TIMEOUT_S = 120
SHUTDOWN_TIMEOUT_S = 10


def _pctl(values: list[float], q: float) -> float:
    """Percentile with linear interpolation (numpy-style)."""
    if not values:
        return 0.0
    xs = sorted(values)
    n = len(xs)
    idx = (n - 1) * q
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    alpha = idx - lo
    return (1 - alpha) * xs[lo] + alpha * xs[hi]


def _run_uvicorn(port: int, workers: int) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "efflab.engine.server_fastapi:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        str(workers),
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _wait_health(base_url: str, timeout_s: float) -> bool:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            resp = httpx.get(f"{base_url}/health", timeout=5.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _cpu_sampler(samples: list[float], stop: threading.Event) -> None:
    """Sample system CPU % every 0.5s until stop is set."""
    while not stop.is_set():
        pct = psutil.cpu_percent(interval=0.5)
        samples.append(pct)


async def _run_benchmark(base_url: str, n_requests: int) -> tuple[list[float], int, float]:
    """Run n_requests parallel /generate, return (latencies, total_tokens, wall_time)."""
    payload = {
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 1,
        "use_kv_cache": True,
    }

    async def generate(client: httpx.AsyncClient, i: int) -> tuple[int, float, int]:
        t0 = time.perf_counter()
        resp = await client.post(f"{base_url}/generate", json=payload)
        resp.raise_for_status()
        latency = time.perf_counter() - t0
        data = resp.json()
        tokens = data["generation"]["generated_new_tokens"]
        return i, latency, tokens

    t_wall_start = time.perf_counter()
    async with httpx.AsyncClient(timeout=180.0) as client:
        tasks = [generate(client, i) for i in range(n_requests)]
        results = await asyncio.gather(*tasks)
    t_wall = time.perf_counter() - t_wall_start

    latencies = [r[1] for r in results]
    total_tokens = sum(r[2] for r in results)
    return latencies, total_tokens, t_wall


def _terminate_proc(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=SHUTDOWN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _run_one_config(
    port: int,
    workers: int,
    n_requests: int,
) -> tuple[int, float, float, float, float]:
    """Spawn server, run benchmark, return (workers, tokens_per_sec, p50, p95, cpu_pct)."""
    proc = _run_uvicorn(port, workers)
    base_url = f"http://127.0.0.1:{port}"

    try:
        if not _wait_health(base_url, HEALTH_TIMEOUT_S):
            raise RuntimeError(f"Server did not become ready within {HEALTH_TIMEOUT_S}s")

        samples: list[float] = []
        stop = threading.Event()
        sampler = threading.Thread(target=_cpu_sampler, args=(samples, stop))
        sampler.daemon = True
        sampler.start()

        # Warm-up: first cpu_percent returns 0, so let one sample happen before benchmark
        time.sleep(0.6)

        latencies, total_tokens, t_wall = asyncio.run(
            _run_benchmark(base_url, n_requests)
        )
        stop.set()
        sampler.join(timeout=2.0)

        tokens_per_sec = total_tokens / t_wall if t_wall > 0 else 0.0
        p50 = _pctl(latencies, 0.50)
        p95 = _pctl(latencies, 0.95)
        cpu_pct = sum(samples) / len(samples) if samples else 0.0

        return workers, tokens_per_sec, p50, p95, cpu_pct
    finally:
        _terminate_proc(proc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-worker scaling: tokens/sec, p50/p95, CPU % across uvicorn workers"
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Worker counts to test (default: 1 2 4)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for server (default: 8000)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=N_REQUESTS,
        help=f"Concurrent requests per run (default: {N_REQUESTS})",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Trials per worker config; report mean (default: 1)",
    )
    args = parser.parse_args()

    rows: list[tuple[int, float, float, float, float]] = []

    for config_idx, workers in enumerate(args.workers):
        tps_list: list[float] = []
        p50_list: list[float] = []
        p95_list: list[float] = []
        cpu_list: list[float] = []

        for trial in range(args.trials):
            # Unique port per (config, trial) to avoid TIME_WAIT reuse
            port = args.port + config_idx * 100 + trial
            print(
                f"  workers={workers} trial={trial + 1}/{args.trials} port={port}...",
                end=" ",
                flush=True,
            )
            w, tps, p50, p95, cpu = _run_one_config(port, workers, args.requests)
            tps_list.append(tps)
            p50_list.append(p50)
            p95_list.append(p95)
            cpu_list.append(cpu)
            print(f"tps={tps:.1f} p50={p50:.3f}s p95={p95:.3f}s cpu={cpu:.1f}%")

        tps_mean = sum(tps_list) / len(tps_list)
        p50_mean = sum(p50_list) / len(p50_list)
        p95_mean = sum(p95_list) / len(p95_list)
        cpu_mean = sum(cpu_list) / len(cpu_list)
        rows.append((workers, tps_mean, p50_mean, p95_mean, cpu_mean))

    print()
    print("| workers | tokens/sec | p50 | p95 | CPU % |")
    print("|---------|------------|-----|-----|-------|")
    for workers, tps, p50, p95, cpu in rows:
        print(f"| {workers:7} | {tps:10.1f} | {p50:.3f} | {p95:.3f} | {cpu:5.1f} |")


if __name__ == "__main__":
    main()
