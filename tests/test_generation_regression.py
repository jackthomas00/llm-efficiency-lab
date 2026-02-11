from __future__ import annotations

from types import SimpleNamespace

import torch

from efflab.engine.runner import InferenceRunner


class _ToyTokenizer:
    eos_token_id = 0

    def __call__(self, text: str, return_tensors: str = "pt"):
        assert return_tensors == "pt"
        ids = [ord(ch) if 0 < ord(ch) < 128 else 1 for ch in text]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens: bool = True):
        chars = []
        for token_id in ids:
            tid = int(token_id)
            if skip_special_tokens and tid == self.eos_token_id:
                continue
            if 0 <= tid < 128:
                chars.append(chr(tid))
        return "".join(chars)


class _ToyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(max_position_embeddings=512)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
    ):
        del attention_mask, use_cache, return_dict
        batch, seq_len = input_ids.shape
        prev_len = int(past_key_values[0]) if past_key_values is not None else 0

        logits = torch.full((batch, seq_len, self.vocab_size), -1e9, device=input_ids.device)
        for pos_in_step in range(seq_len):
            pos = prev_len + pos_in_step + 1
            token_id = 10 if pos % 5 == 0 else 65 + (pos % 26)
            logits[:, pos_in_step, token_id] = 0.0

        total_len = prev_len + seq_len
        return SimpleNamespace(logits=logits, past_key_values=(total_len,))


def _decode_baseline_no_cache(
    *,
    model,
    prompt_ids: torch.Tensor,
    prompt_attn: torch.Tensor | None,
    max_new_tokens: int,
):
    current_ids = prompt_ids
    current_attn = prompt_attn
    generated = []
    eos_id = 0

    for _ in range(max_new_tokens):
        out = model(
            input_ids=current_ids,
            attention_mask=current_attn,
            use_cache=False,
            return_dict=True,
        )
        logits = out.logits[:, -1, :]
        token = int(torch.argmax(logits, dim=-1).item())
        if token == eos_id:
            break

        generated.append(token)
        next_t = torch.tensor([[token]], dtype=current_ids.dtype, device=current_ids.device)
        current_ids = torch.cat([current_ids, next_t], dim=1)
        if current_attn is not None:
            one = torch.ones((1, 1), dtype=current_attn.dtype, device=current_attn.device)
            current_attn = torch.cat([current_attn, one], dim=1)

    return current_ids, generated


def test_kv_decode_regression_haiku_prompt():
    prompt = "Write a 3-line haiku about caching. Use vivid imagery."
    max_new_tokens = 32
    temperature = 0.0
    repetition_penalty = 1.0
    no_repeat_ngram_size = 0

    model = _ToyModel()
    tok = _ToyTokenizer()
    runner = InferenceRunner(model, tok, device="cpu")

    state = runner.prefill(prompt)
    prompt_len = state["input_ids"].shape[1]

    kv = runner.decode(
        state,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )

    output_ids_baseline, generated_baseline = _decode_baseline_no_cache(
        model=model,
        prompt_ids=state["input_ids"],
        prompt_attn=state["attention_mask"],
        max_new_tokens=max_new_tokens,
    )

    assert kv["tokens_generated"] > 0
    assert kv["tokens_generated"] == max_new_tokens
    assert kv["output_ids"].shape[1] == prompt_len + kv["tokens_generated"]
    assert kv["decode_s"] > 0

    assert len(generated_baseline) == kv["tokens_generated"]
    assert torch.equal(output_ids_baseline, kv["output_ids"])
