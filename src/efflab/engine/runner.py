from __future__ import annotations
import time
import torch

class InferenceRunner:
    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.inference_mode()
    def prefill(self, prompt: str):
        t0 = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attn = inputs.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(self.device)

        max_ctx = getattr(self.model.config, "n_positions", None) or getattr(self.model.config, "max_position_embeddings", 1024)
        if input_ids.shape[1] > max_ctx:
            input_ids = input_ids[:, -max_ctx:]
            if attn is not None:
                attn = attn[:, -max_ctx:]

        # One forward pass just to align with "prefill" concept.
        out = self.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        t1 = time.perf_counter()

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "prefill_s": t1 - t0,
            "last_logits": out.logits[:, -1, :],  # (1, vocab)
        }

    @torch.inference_mode()
    def decode(self, state, max_new_tokens: int, temperature: float = 0.0):
        # Simple decode that re-feeds the whole sequence each step (slow baseline).
        # Week 2+ is when you introduce KV cache / use_cache=True.
        input_ids = state["input_ids"]
        attn = state["attention_mask"]

        t0 = time.perf_counter()
        generated = []

        for _ in range(max_new_tokens):
            out = self.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            logits = out.logits[:, -1, :]
            next_id = _sample_next_id(logits, temperature=temperature, top_p=0.95)

            if next_id.item() == self.tokenizer.eos_token_id:
                break
            generated.append(next_id.item())
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if attn is not None:
                attn = torch.cat([attn, torch.ones((attn.shape[0], 1), device=attn.device, dtype=attn.dtype)], dim=1)

        t1 = time.perf_counter()
        return {
            "output_ids": input_ids,
            "new_token_ids": generated,
            "tokens_generated": len(generated),
            "decode_s": t1 - t0,
        }

    def decode_text(self, output_ids):
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


def _sample_next_id(logits, temperature=0.8, top_p=0.95):
    # logits: (1, vocab)
    logits = logits / max(temperature, 1e-6)
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