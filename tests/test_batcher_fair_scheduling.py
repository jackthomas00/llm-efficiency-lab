from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import torch

from efflab.engine.batcher import Batcher, WorkItem


class _SchedulingRunner:
    def __init__(self, *, sleep_per_token_s: float = 0.0):
        self.model = SimpleNamespace(name_or_path="sched-toy")
        self.device = "cpu"
        self.sleep_per_token_s = sleep_per_token_s
        self.decode_order: list[int] = []

    def prefill_batch(self, prompts: list[str], use_kv_cache: bool = True):
        n = len(prompts)
        del use_kv_cache
        return [
            {
                "input_ids": torch.tensor([[65]], dtype=torch.long),
                "attention_mask": torch.tensor([[1]], dtype=torch.long),
                "prefill_s": 0.001,
            }
            for _ in range(n)
        ]

    def decode(
        self,
        state,
        max_new_tokens: int,
        temperature: float = 0.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        use_kv_cache: bool = False,
    ):
        del state, temperature, repetition_penalty, no_repeat_ngram_size, use_kv_cache
        self.decode_order.append(max_new_tokens)
        if self.sleep_per_token_s > 0:
            time.sleep(max_new_tokens * self.sleep_per_token_s)
        return {
            "output_ids": torch.tensor([[65, 66]], dtype=torch.long),
            "new_token_ids": [66],
            "decode_s": 0.001,
            "stopped_early": False,
            "used_kv_cache": False,
        }

    def decode_text(self, output_ids):
        del output_ids
        return "ok"

    def decode_token(self, token_id: int) -> str:
        del token_id
        return "x"


def _make_item(loop: asyncio.AbstractEventLoop, *, tokens: int) -> tuple[WorkItem, asyncio.Future]:
    fut = loop.create_future()
    item = WorkItem(
        prompt=f"p-{tokens}",
        max_new_tokens=tokens,
        temperature=0.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        use_kv_cache=False,
        future=fut,
    )
    return item, fut


def test_short_job_first_reorders_mixed_batch():
    async def _run():
        runner = _SchedulingRunner(sleep_per_token_s=0.0005)
        batcher = Batcher(
            runner,
            max_batch_size=2,
            batch_timeout_ms=40,
            token_buckets=(16, 64),
            enable_short_job_first=True,
            enable_round_robin=False,
        )
        await batcher.start()
        try:
            loop = asyncio.get_running_loop()
            long_item, long_fut = _make_item(loop, tokens=64)
            short_item, short_fut = _make_item(loop, tokens=8)

            await batcher.enqueue(long_item)
            await batcher.enqueue(short_item)
            await asyncio.gather(long_fut, short_fut)
        finally:
            await batcher.stop()

        assert runner.decode_order == [8, 64]

    asyncio.run(_run())


def test_round_robin_alternates_short_long_across_batches():
    async def _run():
        runner = _SchedulingRunner()
        batcher = Batcher(
            runner,
            max_batch_size=2,
            batch_timeout_ms=40,
            token_buckets=(16, 64),
            enable_short_job_first=True,
            enable_round_robin=True,
        )
        await batcher.start()
        try:
            loop = asyncio.get_running_loop()
            items = []
            futures = []
            for tokens in (8, 64, 8, 64):
                item, fut = _make_item(loop, tokens=tokens)
                items.append(item)
                futures.append(fut)
            for item in items:
                await batcher.enqueue(item)
            await asyncio.gather(*futures)
        finally:
            await batcher.stop()

        assert runner.decode_order == [8, 64, 64, 8]

    asyncio.run(_run())


def test_queue_depth_trigger_beats_batch_wait_timeout():
    async def _run():
        runner = _SchedulingRunner()
        batcher = Batcher(
            runner,
            max_batch_size=8,
            batch_timeout_ms=300,
            max_queue_depth=2,
            token_buckets=(16, 64),
            enable_short_job_first=True,
            enable_round_robin=True,
        )
        await batcher.start()
        try:
            loop = asyncio.get_running_loop()
            item_a, fut_a = _make_item(loop, tokens=8)
            item_b, fut_b = _make_item(loop, tokens=8)

            await batcher.enqueue(item_a)
            t0 = time.perf_counter()
            await batcher.enqueue(item_b)
            await asyncio.gather(fut_a, fut_b)
            elapsed = time.perf_counter() - t0
        finally:
            await batcher.stop()

        assert elapsed < 0.2

    asyncio.run(_run())


def test_non_stream_response_includes_batch_metrics():
    async def _run():
        runner = _SchedulingRunner()
        batcher = Batcher(
            runner,
            max_batch_size=2,
            batch_timeout_ms=20,
            token_buckets=(16, 64),
        )
        await batcher.start()
        try:
            loop = asyncio.get_running_loop()
            item, fut = _make_item(loop, tokens=8)
            await batcher.enqueue(item)
            out = await fut
        finally:
            await batcher.stop()

        assert "batching" in out
        metrics = out["batching"]
        assert metrics["queue_wait_ms"] >= 0.0
        assert metrics["batch_size"] == 1
        assert isinstance(metrics["active_batch_size_per_decode_step"], list)
        assert isinstance(metrics["tokens_generated_per_step"], list)
        assert isinstance(metrics["time_per_decode_step"], list)
        assert metrics["batch_formation_time"] >= 0.0

    asyncio.run(_run())
