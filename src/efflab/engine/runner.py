from __future__ import annotations
import time
import torch

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
