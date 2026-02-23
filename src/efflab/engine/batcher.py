from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional

import torch

from efflab.engine.runner import InferenceRunner
from efflab.engine.runner import _apply_no_repeat_ngram, _apply_repetition_penalty, _sample_next_id
from efflab.engine.runner import _stack_past_key_values


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
    ) -> list[dict]:
        t0 = time.perf_counter()
        batch_size = len(subgroup_states)

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
        last_ids = None

        use_penalties = (repetition_penalty is not None and repetition_penalty > 1.0) or (
            no_repeat_ngram_size is not None and no_repeat_ngram_size > 1
        )

        for step in range(max_new_tokens):
            if all(self._item_cancelled(item) for item in subgroup_items):
                break

            past, last_logits = self.runner.decode_step_batch_with_kv(
                last_ids=last_ids,
                past=past,
                last_logits=last_logits,
                attn_full=attn_full,
                L0=L0,
                step=step,
            )

            next_ids: list[int] = []
            for i in range(batch_size):
                row_logits = last_logits[i : i + 1, :]
                if use_penalties:
                    history_t = torch.tensor(
                        [histories[i]],
                        device=row_logits.device,
                        dtype=subgroup_states[i]["input_ids"].dtype,
                    )
                    if repetition_penalty is not None and repetition_penalty > 1.0:
                        row_logits = _apply_repetition_penalty(row_logits, history_t, repetition_penalty)
                    if no_repeat_ngram_size is not None and no_repeat_ngram_size > 1:
                        row_logits = _apply_no_repeat_ngram(row_logits, history_t, no_repeat_ngram_size)

                next_id = _sample_next_id(row_logits, temperature=temperature, top_p=0.95)
                next_ids.append(int(next_id.item()))

            for i, tok in enumerate(next_ids):
                generated[i].append(tok)
                histories[i].append(tok)
                item = subgroup_items[i]
                if item.stream and item.out_q is not None and not self._item_cancelled(item):
                    await item.out_q.put(
                        {
                            "type": "token",
                            "token_id": tok,
                            "text": self.runner.decode_token(tok),
                        }
                    )

            last_ids = torch.tensor(
                next_ids,
                device=subgroup_states[0]["input_ids"].device,
                dtype=subgroup_states[0]["input_ids"].dtype,
            ).view(batch_size, 1)

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
        return decs

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
                            decs = await self._decode_kv_subgroup(
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
                            text = self.runner.decode_text(dec["output_ids"])

                            total_s = state["prefill_s"] + dec["decode_s"]
                            gen_n = len(dec["new_token_ids"])
                            tps = (gen_n / total_s) if total_s > 0 else None

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
                            await self._publish_error(item, e)
