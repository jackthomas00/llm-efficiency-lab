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

            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            generated.append(next_id.item())
            input_ids = torch.cat([input_ids, next_id], dim=1)

            if attn is not None:
                next_attn = torch.ones((attn.shape[0], 1), device=attn.device, dtype=attn.dtype)
                attn = torch.cat([attn, next_attn], dim=1)

        t1 = time.perf_counter()
        return {
            "output_ids": input_ids,
            "new_token_ids": generated,
            "decode_s": t1 - t0,
        }

    def decode_text(self, output_ids):
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
