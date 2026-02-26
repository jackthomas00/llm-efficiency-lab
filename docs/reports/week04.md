# Week 4: Scheduling Experiment — Latency vs Throughput

## Scope
Week 4 focused on optimizing the latency vs throughput tradeoff via scheduling parameter sweeps.

## Environment
- Model: `distilgpt2`
- Device: CPU
- Workload: 8 concurrent requests, 64 max new tokens each
- Script: `scripts/bench_scheduling_sweep.py` (in-process, no HTTP)

## Experiments

### 1) batch_timeout_ms sweep
Values: 5, 10, 25, 50 ms — controls how long the batcher waits for more requests before forming a batch.

### 2) max_batch_size sweep
Values: 2, 4, 8 — maximum number of requests per batch.

### 3) short-job-first grouping
`enable_short_job_first`: True vs False — whether to prioritize shorter jobs (by `max_new_tokens` bucket) when selecting batch items.

## Results

| batch_timeout | max_batch | short_job_first | p50 | p95 | tokens/sec |
|--------------|----------|----------------|-----|-----|------------|
|            5 |        2 | True           | 10.973 | 10.973 |       46.7 |
|            5 |        2 | False          | 13.050 | 13.050 |       39.2 |
|            5 |        4 | True           | 5.795 | 5.795 |       88.3 |
|            5 |        4 | False          | 5.539 | 5.539 |       92.4 |
|            5 |        8 | True           | 5.876 | 5.876 |       87.1 |
|            5 |        8 | False          | 6.920 | 6.920 |       74.0 |
|           10 |        2 | True           | 8.899 | 8.899 |       57.5 |
|           10 |        2 | False          | 8.591 | 8.591 |       59.6 |
|           10 |        4 | True           | 4.811 | 4.811 |      106.4 |
|           10 |        4 | False          | 5.016 | 5.016 |      102.1 |
|           10 |        8 | True           | 4.266 | 4.266 |      120.0 |
|           10 |        8 | False          | 4.493 | 4.493 |      113.9 |
|           25 |        2 | True           | 9.422 | 9.422 |       54.3 |
|           25 |        2 | False          | 13.445 | 13.445 |       38.1 |
|           25 |        4 | True           | 6.022 | 6.022 |       85.0 |
|           25 |        4 | False          | 5.704 | 5.704 |       89.8 |
|           25 |        8 | True           | 4.812 | 4.812 |      106.4 |
|           25 |        8 | False          | 5.831 | 5.831 |       87.8 |
|           50 |        2 | True           | 12.821 | 12.821 |       39.9 |
|           50 |        2 | False          | 10.663 | 10.663 |       48.0 |
|           50 |        4 | True           | 4.657 | 4.657 |      109.9 |
|           50 |        4 | False          | 5.438 | 5.438 |       94.2 |
|           50 |        8 | True           | 4.326 | 4.326 |      118.3 |
|           50 |        8 | False          | 4.482 | 4.482 |      114.2 |

## Observations

- **Best throughput**: 120.0 tokens/sec at `batch_timeout=10`, `max_batch=8`, `short_job_first=True`.
- **Best p50 latency**: 4.266s at the same config.
- **Large batch size (8)** consistently outperforms smaller batches for both latency and throughput.
- **Short-job-first** helps when batch size is small (2) or timeout is high (25–50 ms): lower p50 and higher throughput in several cases.
- **batch_timeout=10** appears to be a sweet spot for this workload; longer timeouts (25, 50) can increase queue wait and hurt latency.

## Multi-Worker Scaling (uvicorn)

### Scope
Measure aggregate tokens/sec, p50/p95 latency, and CPU utilization across uvicorn worker counts (1, 2, 4). Each worker loads its own model copy; requests are distributed across workers.

### Environment
- Model: `distilgpt2`
- Device: CPU
- Workload: 8 concurrent HTTP requests to `/generate`, 64 max new tokens each
- Script: `scripts/bench_workers.py` (spawns server per config, samples CPU during run)

### Results

| workers | tokens/sec | p50 | p95 | CPU % |
|---------|------------|-----|-----|-------|
|       1 |       15.6 | 32.784 | 32.807 |  95.3 |
|       2 |        6.7 | 76.004 | 76.006 |  98.0 |
|       4 |        4.9 | 103.192 | 104.000 |  98.4 |

### Observations

- **Single worker (1)** is best for this CPU-bound workload: highest throughput (15.6 tok/s), lowest latency (p50 ~33s).
- **More workers (2, 4)** degrade performance: throughput drops to 6.7 and 4.9 tok/s; p50 latency roughly doubles and triples.
- CPU utilization stays high (95–98%) across all configs, so the bottleneck is not underutilization.
- **Context switching penalty** dominates: multiple workers compete for CPU cores, each loading a full model copy. With 8 concurrent requests spread across workers, batching efficiency is lost and latency increases.

### How To Run

```bash
# Default: workers 1, 2, 4
python scripts/bench_workers.py

# Custom worker counts
python scripts/bench_workers.py --workers 1 2 4 8

# Multiple trials per config
python scripts/bench_workers.py --trials 3
```

---

## How To Run (Scheduling Sweep)

```bash
# Full sweep (batch_timeout × max_batch × short_job_first)
python scripts/bench_scheduling_sweep.py

# Custom sweep
python scripts/bench_scheduling_sweep.py --batch-timeout 5 10 25 --max-batch 4 8 --short-job-first true
```
