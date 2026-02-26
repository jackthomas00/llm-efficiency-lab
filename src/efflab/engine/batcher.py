from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Any, Optional

import torch

from efflab.engine.runner import InferenceRunner
from efflab.engine.runner import _apply_no_repeat_ngram, _apply_repetition_penalty, _sample_next_id
from efflab.engine.runner import _stack_past_key_values
from efflab.engine.runner import _select_past_key_values_rows


@dataclass
class WorkItem:
    prompt: str
    max_new_tokens: int
    temperature: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    use_kv_cache: bool

    future: asyncio.Future | None = None
    stream: bool = False
    out_q: asyncio.Queue | None = None # queue of dict messages (token/done/error)
    cancelled: bool = False
    enqueued_at_s: float = field(default_factory=time.perf_counter)

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
        token_buckets: tuple[int, ...] = (32, 128),
        max_queue_depth: int = 32,
        enable_short_job_first: bool = True,
        enable_round_robin: bool = True,
    ):
        self.runner = runner
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.token_buckets = tuple(sorted(token_buckets))
        self.max_queue_depth = max_queue_depth
        self.enable_short_job_first = enable_short_job_first
        self.enable_round_robin = enable_round_robin

        self._q: asyncio.Queue[WorkItem] = asyncio.Queue()
        self._bucketed_pending: dict[int, deque[WorkItem]] = defaultdict(deque)
        self._rr_long_first = False
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

    def _bucket_for_tokens(self, max_new_tokens: int) -> int:
        for upper in self.token_buckets:
            if max_new_tokens <= upper:
                return upper
        # Sentinel bucket for "very long" jobs.
        return 10**9

    def _pending_depth(self) -> int:
        return sum(len(bucket_q) for bucket_q in self._bucketed_pending.values())

    async def _drain_incoming(self, *, block: bool, timeout_s: float | None = None) -> bool:
        got_item = False
        if block:
            try:
                if timeout_s is None:
                    first = await self._q.get()
                else:
                    first = await asyncio.wait_for(self._q.get(), timeout=timeout_s)
            except asyncio.TimeoutError:
                return False
            self._bucketed_pending[self._bucket_for_tokens(first.max_new_tokens)].append(first)
            got_item = True

        while True:
            try:
                nxt = self._q.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._bucketed_pending[self._bucket_for_tokens(nxt.max_new_tokens)].append(nxt)
            got_item = True
        return got_item

    def _pop_from_bucket(self, bucket: int) -> WorkItem | None:
        bucket_q = self._bucketed_pending.get(bucket)
        if not bucket_q:
            return None
        while bucket_q:
            item = bucket_q.popleft()
            if not self._item_cancelled(item):
                if not bucket_q:
                    del self._bucketed_pending[bucket]
                return item
        del self._bucketed_pending[bucket]
        return None

    def _select_batch_items(self) -> list[WorkItem]:
        batch: list[WorkItem] = []
        while len(batch) < self.max_batch_size:
            non_empty = [bucket for bucket, q in self._bucketed_pending.items() if q]
            if not non_empty:
                break

            non_empty.sort()
            if not self.enable_short_job_first:
                non_empty.reverse()

            if self.enable_round_robin and len(non_empty) > 1:
                if self._rr_long_first:
                    order = list(reversed(non_empty))
                else:
                    order = non_empty
            else:
                order = non_empty

            picked_any = False
            for bucket in order:
                item = self._pop_from_bucket(bucket)
                if item is None:
                    continue
                batch.append(item)
                picked_any = True
                if len(batch) >= self.max_batch_size:
                    break
            if not picked_any:
                break

        if self.enable_round_robin and len(batch) > 0:
            self._rr_long_first = not self._rr_long_first
        return batch

    @staticmethod
    def _item_cancelled(item: WorkItem) -> bool:
        return item.cancelled or (item.future is not None and item.future.cancelled())

    async def _publish_error(self, item: WorkItem, e: Exception):
        if item.stream and item.out_q is not None:
            await item.out_q.put({"type": "error", "message": str(e)})
            return
        if item.future is not None and not item.future.cancelled():
            item.future.set_exception(e)

    async def _publish_done(self, item: WorkItem):
        if item.stream and item.out_q is not None and not self._item_cancelled(item):
            await item.out_q.put({"type": "done"})

    async def _decode_kv_subgroup(
        self,
        subgroup_items: list[WorkItem],
        subgroup_states: list[dict],
        *,
        max_new_tokens: int,
        temperature: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
    ) -> tuple[list[dict], dict[str, list[float] | list[int]]]:
        t0 = time.perf_counter()
        batch_size = len(subgroup_states)
        eos_id = self.runner.tokenizer.eos_token_id
        active_batch_size_per_decode_step: list[int] = []
        tokens_generated_per_step: list[int] = []
        time_per_decode_step: list[float] = []

        generated: list[list[int]] = [[] for _ in range(batch_size)]
        histories: list[list[int]] = [s["input_ids"][0].tolist() for s in subgroup_states]
        last_logits = torch.cat([s["last_logits"] for s in subgroup_states], dim=0)  # [B, vocab]

        attn_full = None
        L0 = None
        if subgroup_states[0].get("attention_mask") is not None:
            attn = torch.cat([s["attention_mask"] for s in subgroup_states], dim=0)
            L0 = attn.shape[1]
            attn_full = torch.ones(
                (attn.shape[0], L0 + max_new_tokens),
                device=attn.device,
                dtype=attn.dtype,
            )
            attn_full[:, :L0] = attn

        past = _stack_past_key_values([s["past_key_values"] for s in subgroup_states])
        active_to_orig = list(range(batch_size))
        active_last_logits = last_logits
        active_attn_full = attn_full
        active_past = past
        active_last_ids = None

        use_penalties = (repetition_penalty is not None and repetition_penalty > 1.0) or (
            no_repeat_ngram_size is not None and no_repeat_ngram_size > 1
        )

        for step in range(max_new_tokens):
            if not active_to_orig:
                break

            if all(self._item_cancelled(subgroup_items[i]) for i in active_to_orig):
                break

            step_t0 = time.perf_counter()
            active_batch_size_per_decode_step.append(len(active_to_orig))
            active_past, active_last_logits = self.runner.decode_step_batch_with_kv(
                last_ids=active_last_ids,
                past=active_past,
                last_logits=active_last_logits,
                attn_full=active_attn_full,
                L0=L0,
                step=step,
            )

            next_ids: list[int] = []
            alive_rows: list[int] = []
            next_active_to_orig: list[int] = []
            for active_i, orig_i in enumerate(active_to_orig):
                item = subgroup_items[orig_i]
                if self._item_cancelled(item):
                    continue

                row_logits = active_last_logits[active_i : active_i + 1, :]
                if use_penalties:
                    history_t = torch.tensor(
                        [histories[orig_i]],
                        device=row_logits.device,
                        dtype=subgroup_states[orig_i]["input_ids"].dtype,
                    )
                    if repetition_penalty is not None and repetition_penalty > 1.0:
                        row_logits = _apply_repetition_penalty(row_logits, history_t, repetition_penalty)
                    if no_repeat_ngram_size is not None and no_repeat_ngram_size > 1:
                        row_logits = _apply_no_repeat_ngram(row_logits, history_t, no_repeat_ngram_size)

                next_id = _sample_next_id(row_logits, temperature=temperature, top_p=0.95)
                tok = int(next_id.item())
                if eos_id is not None and tok == eos_id:
                    continue

                generated[orig_i].append(tok)
                histories[orig_i].append(tok)
                if item.stream and item.out_q is not None and not self._item_cancelled(item):
                    await item.out_q.put(
                        {
                            "type": "token",
                            "token_id": tok,
                            "text": self.runner.decode_token(tok),
                        }
                    )
                next_ids.append(tok)
                alive_rows.append(active_i)
                next_active_to_orig.append(orig_i)

            step_t1 = time.perf_counter()
            tokens_generated_per_step.append(len(next_ids))
            time_per_decode_step.append(step_t1 - step_t0)

            if not next_active_to_orig:
                break

            active_last_ids = torch.tensor(
                next_ids,
                device=subgroup_states[0]["input_ids"].device,
                dtype=subgroup_states[0]["input_ids"].dtype,
            ).view(len(next_ids), 1)

            if len(next_active_to_orig) != len(active_to_orig):
                alive_rows_t = torch.tensor(
                    alive_rows,
                    device=subgroup_states[0]["input_ids"].device,
                    dtype=torch.long,
                )
                if active_attn_full is not None:
                    active_attn_full = active_attn_full.index_select(0, alive_rows_t)
                active_past = _select_past_key_values_rows(active_past, alive_rows_t)
            active_to_orig = next_active_to_orig

        t1 = time.perf_counter()
        decode_s = t1 - t0

        decs = []
        for i, state in enumerate(subgroup_states):
            prompt_ids = state["input_ids"]
            gen_ids = generated[i]
            if gen_ids:
                gen_t = torch.tensor([gen_ids], device=prompt_ids.device, dtype=prompt_ids.dtype)
                output_ids = torch.cat([prompt_ids, gen_t], dim=1)
            else:
                output_ids = prompt_ids
            decs.append(
                {
                    "output_ids": output_ids,
                    "new_token_ids": gen_ids,
                    "tokens_generated": len(gen_ids),
                    "decode_s": decode_s,
                    "stopped_early": len(gen_ids) < max_new_tokens,
                    "used_kv_cache": True,
                }
            )
        return decs, {
            "active_batch_size_per_decode_step": active_batch_size_per_decode_step,
            "tokens_generated_per_step": tokens_generated_per_step,
            "time_per_decode_step": time_per_decode_step,
        }

    async def _collect_batch(self) -> tuple[list[WorkItem], float]:
        formation_t0 = time.perf_counter()
        # Always block for at least 1 item unless pending is already populated.
        if self._pending_depth() == 0:
            await self._drain_incoming(block=True)

        # Opportunistically gather more until timeout or queue-depth trigger.
        deadline = time.perf_counter() + (self.batch_timeout_ms / 1000.0)
        while self._pending_depth() < self.max_batch_size:
            if self._pending_depth() >= self.max_queue_depth:
                break
            timeout = deadline - time.perf_counter()
            if timeout <= 0:
                break
            try:
                await self._drain_incoming(block=True, timeout_s=timeout)
            except asyncio.TimeoutError:
                break

        formation_t1 = time.perf_counter()
        return self._select_batch_items(), formation_t1 - formation_t0

    def _set_result(
        self,
        *,
        item: WorkItem,
        state: dict[str, Any],
        dec: dict[str, Any],
        queue_wait_ms: float,
        batch_size: int,
        batch_formation_time: float,
        active_batch_size_per_decode_step: list[int],
        tokens_generated_per_step: list[int],
        time_per_decode_step: list[float],
    ):
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
            "batching": {
                "queue_wait_ms": queue_wait_ms,
                "batch_size": batch_size,
                "active_batch_size_per_decode_step": active_batch_size_per_decode_step,
                "tokens_generated_per_step": tokens_generated_per_step,
                "time_per_decode_step": time_per_decode_step,
                "batch_formation_time": batch_formation_time,
            },
            "text": text,
        })

    async def _loop(self):
        while not self._stop.is_set():
            batch, batch_formation_time = await self._collect_batch()
            if not batch:
                continue
            batch_dispatched_at_s = time.perf_counter()
            collected_batch_size = len(batch)

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
                        await self._publish_error(item, e)
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
                            decs, kv_step_metrics = await self._decode_kv_subgroup(
                                subgroup_items,
                                subgroup_states,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                repetition_penalty=repetition_penalty,
                                no_repeat_ngram_size=no_repeat_ngram_size,
                            )
                        except Exception as e:
                            for item in subgroup_items:
                                await self._publish_error(item, e)
                            continue

                        for item, state, dec in zip(subgroup_items, subgroup_states, decs):
                            if self._item_cancelled(item):
                                continue
                            try:
                                if item.stream:
                                    await self._publish_done(item)
                                else:
                                    self._set_result(
                                        item=item,
                                        state=state,
                                        dec=dec,
                                        queue_wait_ms=(batch_dispatched_at_s - item.enqueued_at_s) * 1000.0,
                                        batch_size=collected_batch_size,
                                        batch_formation_time=batch_formation_time,
                                        active_batch_size_per_decode_step=list(
                                            kv_step_metrics["active_batch_size_per_decode_step"]
                                        ),
                                        tokens_generated_per_step=list(
                                            kv_step_metrics["tokens_generated_per_step"]
                                        ),
                                        time_per_decode_step=list(kv_step_metrics["time_per_decode_step"]),
                                    )
                            except Exception as e:
                                await self._publish_error(item, e)
                else:
                    for item, state in zip(group_items, states):
                        if self._item_cancelled(item):
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
                            gen_n = len(dec["new_token_ids"])

                            if item.stream and item.out_q is not None:
                                for token_id in dec["new_token_ids"]:
                                    await item.out_q.put(
                                        {
                                            "type": "token",
                                            "token_id": token_id,
                                            "text": self.runner.decode_token(token_id),
                                        }
                                    )
                                await self._publish_done(item)
                            else:
                                self._set_result(
                                    item=item,
                                    state=state,
                                    dec=dec,
                                    queue_wait_ms=(batch_dispatched_at_s - item.enqueued_at_s) * 1000.0,
                                    batch_size=collected_batch_size,
                                    batch_formation_time=batch_formation_time,
                                    active_batch_size_per_decode_step=[1],
                                    tokens_generated_per_step=[gen_n],
                                    time_per_decode_step=[dec["decode_s"]],
                                )
                        except Exception as e:
                            await self._publish_error(item, e)
