from __future__ import annotations
import time
import torch
from transformers.cache_utils import Cache, DynamicCache


def _is_legacy_kv_cache(past_key_values) -> bool:
    if not isinstance(past_key_values, tuple) or len(past_key_values) == 0:
        return False
    first = past_key_values[0]
    return (
        isinstance(first, tuple)
        and len(first) == 2
        and hasattr(first[0], "shape")
        and hasattr(first[1], "shape")
    )

class InferenceRunner:
    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.inference_mode()
    def prefill(self, prompt: str, use_kv_cache: bool = True):
        t0 = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attn = inputs.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(self.device)

        max_ctx = getattr(self.model.config, "n_positions", None) or getattr(
            self.model.config, "max_position_embeddings", 1024
        )
        if input_ids.shape[1] > max_ctx:
            input_ids = input_ids[:, -max_ctx:]
            if attn is not None:
                attn = attn[:, -max_ctx:]

        out = self.model(
            input_ids=input_ids,
            attention_mask=attn,
            use_cache=use_kv_cache,
            return_dict=True,
        )
        t1 = time.perf_counter()

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "prefill_s": t1 - t0,
            "last_logits": out.logits[:, -1, :],  # (1, vocab)
            "past_key_values": out.past_key_values if use_kv_cache else None,
            "used_kv_cache": use_kv_cache,
        }

    @torch.inference_mode()
    def prefill_batch(self, prompts: list[str], use_kv_cache: bool = True):
        if not prompts:
            return []

        t0 = time.perf_counter()

        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attn = inputs.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(self.device)

        max_ctx = getattr(self.model.config, "n_positions", None) or getattr(
            self.model.config, "max_position_embeddings", 1024
        )
        if input_ids.shape[1] > max_ctx:
            input_ids = input_ids[:, -max_ctx:]
            if attn is not None:
                attn = attn[:, -max_ctx:]

        out = self.model(
            input_ids=input_ids,
            attention_mask=attn,
            use_cache=use_kv_cache,
            return_dict=True,
        )
        t1 = time.perf_counter()
        prefill_each = (t1 - t0) / input_ids.shape[0]

        if attn is None:
            lengths = torch.full(
                (input_ids.shape[0],),
                input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            )
        else:
            lengths = attn.sum(dim=1).to(dtype=torch.long)

        lengths = torch.clamp(lengths, min=1)
        last_positions = lengths - 1
        batch_idx = torch.arange(input_ids.shape[0], device=input_ids.device)
        last_logits = out.logits[batch_idx, last_positions, :]

        states = []
        for i in range(input_ids.shape[0]):
            seq_len = int(lengths[i].item())
            item_input_ids = input_ids[i : i + 1, :seq_len]
            item_attn = None
            if attn is not None:
                # KV decode needs mask length to match cached sequence length.
                item_attn = attn[i : i + 1, :] if use_kv_cache else attn[i : i + 1, :seq_len]

            item_past = None
            if use_kv_cache and out.past_key_values is not None:
                item_past = _split_past_key_values(out.past_key_values, i)

            states.append(
                {
                    "input_ids": item_input_ids,
                    "attention_mask": item_attn,
                    "prefill_s": prefill_each,
                    "last_logits": last_logits[i : i + 1, :],
                    "past_key_values": item_past,
                    "prompt_len": seq_len,
                    "used_kv_cache": use_kv_cache,
                }
            )

        return states

    @torch.inference_mode()
    def decode_batch_with_kv(
        self,
        states: list[dict],
        max_new_tokens: int,
        temperature: float = 0.0,
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 3,
    ):
        if not states:
            return []

        t0 = time.perf_counter()
        batch_size = len(states)
        eos_id = self.tokenizer.eos_token_id

        # Per-request outputs (kept as Python lists for fast appends)
        generated: list[list[int]] = [[] for _ in range(batch_size)]
        histories: list[list[int]] = [s["input_ids"][0].tolist() for s in states]

        # Start from prefill-provided last logits
        last_logits = torch.cat([s["last_logits"] for s in states], dim=0)  # [B, vocab]

        # Attention mask (note: in prefill_batch you keep full padded attn for KV)
        attn_full = None
        L0 = None
        if states[0].get("attention_mask") is not None:
            attn = torch.cat([s["attention_mask"] for s in states], dim=0)  # [B, L]

            # attn is [B, L0] for the (padded) prompt length
            L0 = attn.shape[1]
            # Preallocate the full mask for prompt + max_new_tokens
            attn_full = torch.ones(
                (attn.shape[0], L0 + max_new_tokens),
                device=attn.device,
                dtype=attn.dtype,
            )
            # Copy the prompt mask into the front
            attn_full[:, :L0] = attn

        past = _stack_past_key_values([s["past_key_values"] for s in states])

        # Active rows are compacted as requests finish on EOS.
        active_to_orig = list(range(batch_size))
        active_last_logits = last_logits
        active_attn_full = attn_full
        active_past = past
        active_last_ids = None  # type: ignore

        # Fast path: if penalites are off, skip expensive per-row history tensors
        use_penalties = (repetition_penalty is not None and repetition_penalty > 1.0) or (
            no_repeat_ngram_size is not None and no_repeat_ngram_size > 1
        )

        for step in range(max_new_tokens):
            if not active_to_orig:
                break

            active_batch_size = len(active_to_orig)
            # For step 0, we already have the last logits from prefill.
            # For step >= 1, run a single batched forward on the last sampled ids.
            if step > 0:
                assert active_last_ids is not None and active_last_ids.shape == (active_batch_size, 1)

                # For step > 0, we've already sampled `step` tokens, so total length is L0 + step
                cur_attn = None
                if active_attn_full is not None:
                    assert L0 is not None, "L0 must be set before step > 0"
                    cur_attn = active_attn_full[:, : (L0 + step)]

                out = self.model(
                    input_ids=active_last_ids,
                    attention_mask=cur_attn,
                    past_key_values=active_past,
                    use_cache=True,
                    return_dict=True,
                )
                active_past = out.past_key_values
                active_last_logits = out.logits[:, -1, :]  # [B, vocab]

            # Sample next ids (still per-row because penalites/top-p are per-row here)
            next_ids: list[int] = []
            alive_rows: list[int] = []
            next_active_to_orig: list[int] = []
            for active_i, orig_i in enumerate(active_to_orig):
                row_logits = active_last_logits[active_i : active_i + 1, :]  # [1, vocab]

                if use_penalties:
                    history_t = torch.tensor(
                        [histories[orig_i]],
                        device=row_logits.device,
                        dtype=states[orig_i]["input_ids"].dtype,
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
                next_ids.append(tok)
                alive_rows.append(active_i)
                next_active_to_orig.append(orig_i)

            if not next_active_to_orig:
                break

            # Update last_ids tensor for the NEXT step's forward
            active_last_ids = torch.tensor(
                next_ids,
                device=states[0]["input_ids"].device,
                dtype=states[0]["input_ids"].dtype,
            ).view(len(next_ids), 1)

            if len(next_active_to_orig) != active_batch_size:
                alive_rows_t = torch.tensor(
                    alive_rows,
                    device=states[0]["input_ids"].device,
                    dtype=torch.long,
                )
                if active_attn_full is not None:
                    active_attn_full = active_attn_full.index_select(0, alive_rows_t)
                active_past = _select_past_key_values_rows(active_past, alive_rows_t)
            active_to_orig = next_active_to_orig

        t1 = time.perf_counter()
        decode_s = t1 - t0

        results = []
        for i, state in enumerate(states):
            prompt_ids = state["input_ids"]
            gen_ids = generated[i]
            if gen_ids:
                gen_t = torch.tensor([gen_ids], device=prompt_ids.device, dtype=prompt_ids.dtype)
                output_ids = torch.cat([prompt_ids, gen_t], dim=1)
            else:
                output_ids = prompt_ids
            results.append(
                {
                    "output_ids": output_ids,
                    "new_token_ids": gen_ids,
                    "tokens_generated": len(gen_ids),
                    "decode_s": decode_s,
                    "stopped_early": len(gen_ids) < max_new_tokens,
                    "used_kv_cache": True,
                }
            )
        return results

    @torch.inference_mode()
    def decode_step_batch_with_kv(self, *, last_ids, past, last_logits, attn_full, L0, step):
        # If step == 0, we already have the last logits from prefill.
        # For step >= 1, forward on last_ids with cur_attn slice.
        if step > 0:
            assert last_ids is not None, "last_ids must be set before step > 0"
            cur_attn = None
            if attn_full is not None:
                assert L0 is not None, "L0 must be set before step > 0"
                cur_attn = attn_full[:, : (L0 + step)]

            out = self.model(
                input_ids=last_ids, # [B, 1]
                attention_mask=cur_attn,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values
            last_logits = out.logits[:, -1, :] # [B, vocab]

        return past, last_logits

    @torch.inference_mode()
    def decode(
        self,
        state,
        max_new_tokens: int,
        temperature: float = 0.0,
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 3,
        use_kv_cache: bool = True,
    ):
        if use_kv_cache:
            return self._decode_with_kv_cache(
                state=state,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        return self._decode_without_kv_cache(
            state=state,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

    def _decode_with_kv_cache(
        self,
        state,
        max_new_tokens: int,
        temperature: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
    ):
        # Decode using prefill output: past + last_logits. No re-forward of prompt.
        prompt_ids = state["input_ids"]
        attn = state["attention_mask"]
        past = state["past_key_values"]

        t0 = time.perf_counter()
        generated = []

        eos_id = self.tokenizer.eos_token_id
        if max_new_tokens <= 0:
            t1 = time.perf_counter()
            return {
                "output_ids": prompt_ids,
                "new_token_ids": generated,
                "tokens_generated": 0,
                "decode_s": t1 - t0,
                "stopped_early": True,
                "used_kv_cache": True,
            }

        # Helper: build "history" tensor for penalties (prompt + generated)
        def _history_ids_tensor():
            if not generated:
                return prompt_ids
            gen_t = torch.tensor([generated], device=prompt_ids.device, dtype=prompt_ids.dtype)
            return torch.cat([prompt_ids, gen_t], dim=1)

        cur_input = None
        for step in range(max_new_tokens):
            if step == 0:
                logits = state["last_logits"]
            else:
                assert cur_input is not None and cur_input.shape == (1, 1), (
                    "cached decode must feed only last token (1, 1)"
                )
                out = self.model(
                    input_ids=cur_input,
                    attention_mask=attn,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = out.past_key_values
                logits = out.logits[:, -1, :]

            history_ids = _history_ids_tensor()
            logits = _apply_repetition_penalty(logits, history_ids, repetition_penalty)
            logits = _apply_no_repeat_ngram(logits, history_ids, no_repeat_ngram_size)

            next_id = _sample_next_id(logits, temperature=temperature, top_p=0.95)
            if next_id.item() == eos_id:
                break

            generated.append(next_id.item())

            # advance loop state
            cur_input = next_id.view(1, 1)
            if attn is not None:
                attn = torch.cat(
                    [attn, torch.ones((attn.shape[0], 1), device=attn.device, dtype=attn.dtype)],
                    dim=1,
                )

        t1 = time.perf_counter()

        # Build full output_ids = prompt + generated
        if generated:
            gen_t = torch.tensor([generated], device=prompt_ids.device, dtype=prompt_ids.dtype)
            output_ids = torch.cat([prompt_ids, gen_t], dim=1)
        else:
            output_ids = prompt_ids

        return {
            "output_ids": output_ids,
            "new_token_ids": generated,
            "tokens_generated": len(generated),
            "decode_s": t1 - t0,
            "stopped_early": len(generated) < max_new_tokens,
            "used_kv_cache": True,
        }

    def _decode_without_kv_cache(
        self,
        state,
        max_new_tokens: int,
        temperature: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
    ):
        # Baseline decode that re-feeds the full sequence each step.
        input_ids = state["input_ids"]
        attn = state["attention_mask"]

        t0 = time.perf_counter()
        generated = []

        for _ in range(max_new_tokens):
            out = self.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            logits = out.logits[:, -1, :]
            logits = _apply_repetition_penalty(logits, input_ids, repetition_penalty)
            logits = _apply_no_repeat_ngram(logits, input_ids, no_repeat_ngram_size)
            next_id = _sample_next_id(logits, temperature=temperature, top_p=0.95)

            if next_id.item() == self.tokenizer.eos_token_id:
                break

            generated.append(next_id.item())
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if attn is not None:
                attn = torch.cat(
                    [attn, torch.ones((attn.shape[0], 1), device=attn.device, dtype=attn.dtype)],
                    dim=1,
                )

        t1 = time.perf_counter()
        return {
            "output_ids": input_ids,
            "new_token_ids": generated,
            "tokens_generated": len(generated),
            "decode_s": t1 - t0,
            "stopped_early": len(generated) < max_new_tokens,
        }

    def decode_text(self, output_ids):
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], skip_special_tokens=False)


def _sample_next_id(logits, temperature=0.8, top_p=0.95, top_k=50):
    # logits: (1, vocab)
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / max(temperature, 1e-6)

    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, k=min(top_k, logits.shape[-1]))
        kth = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < kth, torch.full_like(logits, -1e10), logits)

    probs = torch.softmax(logits, dim=-1)

    # nucleus (top-p) sampling
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cum > top_p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False
    sorted_probs[cutoff] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    next_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
    next_id = sorted_idx.gather(-1, next_in_sorted)
    return next_id


def _apply_repetition_penalty(logits, input_ids, penalty: float):
    if penalty <= 1.0:
        return logits

    adjusted = logits.clone()
    seen_ids = torch.unique(input_ids[0])
    seen_logits = adjusted[:, seen_ids]
    adjusted[:, seen_ids] = torch.where(
        seen_logits < 0,
        seen_logits * penalty,
        seen_logits / penalty,
    )
    return adjusted


def _apply_no_repeat_ngram(logits, input_ids, ngram_size: int):
    if ngram_size < 2:
        return logits

    sequence = input_ids[0].tolist()
    if len(sequence) < ngram_size - 1:
        return logits

    prefix_len = ngram_size - 1
    ngram_prefix_to_next = {}
    for i in range(len(sequence) - ngram_size + 1):
        prefix = tuple(sequence[i : i + prefix_len])
        next_token = sequence[i + prefix_len]
        if prefix not in ngram_prefix_to_next:
            ngram_prefix_to_next[prefix] = set()
        ngram_prefix_to_next[prefix].add(next_token)

    current_prefix = tuple(sequence[-prefix_len:])
    banned = ngram_prefix_to_next.get(current_prefix, set())
    if not banned:
        return logits

    adjusted = logits.clone()
    adjusted[:, list(banned)] = float("-inf")
    return adjusted


def _stack_past_key_values(per_item_past):
    if not per_item_past or per_item_past[0] is None:
        return None

    first = per_item_past[0]
    if isinstance(first, Cache):
        ddp_cache_data = []
        num_layers = len(first.layers)
        for layer_idx in range(num_layers):
            keys = torch.cat([item.layers[layer_idx].keys for item in per_item_past], dim=0)
            values = torch.cat([item.layers[layer_idx].values for item in per_item_past], dim=0)
            ddp_cache_data.append((keys, values))
        return DynamicCache(ddp_cache_data=ddp_cache_data)

    if not _is_legacy_kv_cache(first):
        if len(per_item_past) == 1:
            return first
        raise TypeError("Unsupported past_key_values type for batched KV decode")

    num_layers = len(first)
    ddp_cache_data = []
    for layer_idx in range(num_layers):
        keys = torch.cat([item_past[layer_idx][0] for item_past in per_item_past], dim=0)
        values = torch.cat([item_past[layer_idx][1] for item_past in per_item_past], dim=0)
        ddp_cache_data.append((keys, values))
    return DynamicCache(ddp_cache_data=ddp_cache_data)


def _split_past_key_values(past_key_values, item_idx: int):
    if past_key_values is None:
        return None

    if isinstance(past_key_values, Cache):
        ddp_cache_data = []
        for layer in past_key_values.layers:
            ddp_cache_data.append(
                (
                    layer.keys[item_idx : item_idx + 1, ...],
                    layer.values[item_idx : item_idx + 1, ...],
                )
            )
        return DynamicCache(ddp_cache_data=ddp_cache_data)

    if _is_legacy_kv_cache(past_key_values):
        ddp_cache_data = []
        for layer in past_key_values:
            ddp_cache_data.append(
                (
                    layer[0][item_idx : item_idx + 1, ...],
                    layer[1][item_idx : item_idx + 1, ...],
                )
            )
        return DynamicCache(ddp_cache_data=ddp_cache_data)

    return past_key_values


def _select_past_key_values_rows(past_key_values, row_indices: torch.Tensor):
    if past_key_values is None:
        return None

    if isinstance(past_key_values, Cache):
        ddp_cache_data = []
        for layer in past_key_values.layers:
            ddp_cache_data.append(
                (
                    layer.keys.index_select(0, row_indices),
                    layer.values.index_select(0, row_indices),
                )
            )
        return DynamicCache(ddp_cache_data=ddp_cache_data)

    if _is_legacy_kv_cache(past_key_values):
        ddp_cache_data = []
        for layer in past_key_values:
            ddp_cache_data.append(
                (
                    layer[0].index_select(0, row_indices),
                    layer[1].index_select(0, row_indices),
                )
            )
        return DynamicCache(ddp_cache_data=ddp_cache_data)

    return past_key_values
