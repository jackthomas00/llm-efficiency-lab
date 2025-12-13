from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from efflab.common.config import ModelConfig

def _dtype_from_str(s: str):
    s = s.lower()
    if s in ("float16", "fp16"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32

def load_model_and_tokenizer(cfg: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    # Some GPT2-family tokenizers have no pad token; set it to eos for batching later.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _dtype_from_str(cfg.dtype)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=dtype)
    model.eval()

    if cfg.device == "cuda" and torch.cuda.is_available():
        model.to("cuda")
    else:
        cfg.device = "cpu"
        model.to("cpu")

    return model, tokenizer, cfg
