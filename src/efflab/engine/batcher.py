from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional

from efflab.engine.runner import InferenceRunner


@dataclass
class WorkItem:
    prompt: str
    max_new_tokens: int
    temperature: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    use_kv_cache: bool
    future: asyncio.Future


class Batcher:
    """
    Simple microbatcher:
      - collects WorkItems up to max_batch_size
      - waits up to batch_timeout_ms for more items
      - processes each item (initially sequential decode; later batched decode)
    """

    def __init__(
        self,
        runner: InferenceRunner,
        *,
        max_batch_size: int = 8,
        batch_timeout_ms: int = 10,
    ):
        self.runner = runner
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms

        self._q: asyncio.Queue[WorkItem] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    async def start(self):
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task is None:
            return
        self._stop.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def enqueue(self, item: WorkItem):
        await self._q.put(item)

    async def _collect_batch(self) -> list[WorkItem]:
        # Always block for 1 item
        first = await self._q.get()
        items = [first]

        # Then opportunistically gather more until timeout or max size
        deadline = time.perf_counter() + (self.batch_timeout_ms / 1000.0)
        while len(items) < self.max_batch_size:
            timeout = deadline - time.perf_counter()
            if timeout <= 0:
                break
            try:
                nxt = await asyncio.wait_for(self._q.get(), timeout=timeout)
                items.append(nxt)
            except asyncio.TimeoutError:
                break

        return items

    async def _loop(self):
        while not self._stop.is_set():
            batch = await self._collect_batch()

            kv_items = [item for item in batch if item.use_kv_cache]
            base_items = [item for item in batch if not item.use_kv_cache]

            for group_items, group_use_kv in ((kv_items, True), (base_items, False)):
                if not group_items:
                    continue

                prompts = [item.prompt for item in group_items]
                try:
                    states = self.runner.prefill_batch(prompts, use_kv_cache=group_use_kv)
                    if len(states) != len(group_items):
                        raise RuntimeError("prefill_batch returned unexpected number of states")
                except Exception as e:
                    for item in group_items:
                        if not item.future.cancelled():
                            item.future.set_exception(e)
                    continue

                if group_use_kv:
                    param_groups = defaultdict(list)
                    for idx, item in enumerate(group_items):
                        key = (
                            item.max_new_tokens,
                            item.temperature,
                            item.repetition_penalty,
                            item.no_repeat_ngram_size,
                        )
                        param_groups[key].append(idx)

                    for key, idxs in param_groups.items():
                        max_new_tokens, temperature, repetition_penalty, no_repeat_ngram_size = key
                        subgroup_items = [group_items[i] for i in idxs]
                        subgroup_states = [states[i] for i in idxs]

                        try:
                            decs = self.runner.decode_batch_with_kv(
                                subgroup_states,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                repetition_penalty=repetition_penalty,
                                no_repeat_ngram_size=no_repeat_ngram_size,
                            )
                        except Exception as e:
                            for item in subgroup_items:
                                if not item.future.cancelled():
                                    item.future.set_exception(e)
                            continue

                        for item, state, dec in zip(subgroup_items, subgroup_states, decs):
                            if item.future.cancelled():
                                continue
                            try:
                                text = self.runner.decode_text(dec["output_ids"])
                                total_s = state["prefill_s"] + dec["decode_s"]
                                gen_n = len(dec["new_token_ids"])
                                tps = (gen_n / total_s) if total_s > 0 else None

                                item.future.set_result({
                                    "model": getattr(self.runner.model, "name_or_path", "unknown"),
                                    "device": self.runner.device,
                                    "timing": {
                                        "prefill_s": state["prefill_s"],
                                        "decode_s": dec["decode_s"],
                                        "total_s": total_s,
                                        "tokens_per_second": tps,
                                    },
                                    "generation": {
                                        "requested_new_tokens": item.max_new_tokens,
                                        "generated_new_tokens": gen_n,
                                        "new_token_ids_preview": dec["new_token_ids"][:10],
                                        "stopped_early": dec.get(
                                            "stopped_early",
                                            gen_n < item.max_new_tokens,
                                        ),
                                        "repetition_penalty": item.repetition_penalty,
                                        "no_repeat_ngram_size": item.no_repeat_ngram_size,
                                        "used_kv_cache": dec.get("used_kv_cache", item.use_kv_cache),
                                    },
                                    "text": text,
                                })
                            except Exception as e:
                                item.future.set_exception(e)
                else:
                    for item, state in zip(group_items, states):
                        if item.future.cancelled():
                            continue

                        try:
                            dec = self.runner.decode(
                                state,
                                max_new_tokens=item.max_new_tokens,
                                temperature=item.temperature,
                                repetition_penalty=item.repetition_penalty,
                                no_repeat_ngram_size=item.no_repeat_ngram_size,
                                use_kv_cache=item.use_kv_cache,
                            )
                            text = self.runner.decode_text(dec["output_ids"])

                            total_s = state["prefill_s"] + dec["decode_s"]
                            gen_n = len(dec["new_token_ids"])
                            tps = (gen_n / total_s) if total_s > 0 else None

                            item.future.set_result({
                                "model": getattr(self.runner.model, "name_or_path", "unknown"),
                                "device": self.runner.device,
                                "timing": {
                                    "prefill_s": state["prefill_s"],
                                    "decode_s": dec["decode_s"],
                                    "total_s": total_s,
                                    "tokens_per_second": tps,
                                },
                                "generation": {
                                    "requested_new_tokens": item.max_new_tokens,
                                    "generated_new_tokens": gen_n,
                                    "new_token_ids_preview": dec["new_token_ids"][:10],
                                    "stopped_early": dec.get("stopped_early", gen_n < item.max_new_tokens),
                                    "repetition_penalty": item.repetition_penalty,
                                    "no_repeat_ngram_size": item.no_repeat_ngram_size,
                                    "used_kv_cache": dec.get("used_kv_cache", item.use_kv_cache),
                                },
                                "text": text,
                            })
                        except Exception as e:
                            item.future.set_exception(e)
