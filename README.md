# LLM Efficiency Lab

A hands-on lab for building **efficient LLM systems** across the full stack:
- inference engines (KV cache, batching, streaming)
- model distillation & quantization
- inference cost profiling
- structured / constrained decoding

This repo focuses on **latency, throughput, memory, and cost tradeoffs** rather than model hype.

---

## Projects

### 1. Minimal Inference Engine
- Prefill vs decode separation
- KV-cache reuse & eviction
- Batched decoding
- Streaming token API (FastAPI)

### 2. Distillation + Quantization
- Teacher → student distillation
- INT8 / INT4 quantization
- Accuracy vs latency benchmarks

### 3. Inference Cost Profiler
- Prefill vs decode latency breakdown
- Throughput vs batch size
- GPU memory usage tracking

### 4. Structured Outputs
- Regex / JSON constrained decoding
- Output repair & retry minimization

---

## Quickstart
```bash
make setup
make serve
```

---

## Philosophy

Model quality is plateauing.
Efficiency is the bottleneck.

This repo explores how real systems ship LLMs at scale.