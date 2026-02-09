from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel

from efflab.common.config import ModelConfig
from efflab.engine.model_loader import load_model_and_tokenizer
from efflab.engine.runner import InferenceRunner

app = FastAPI()

cfg = ModelConfig()
_model, _tok, _cfg = load_model_and_tokenizer(cfg)
runner = InferenceRunner(_model, _tok, _cfg.device)

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = cfg.max_new_tokens_default
    temperature: float = 0.8

@app.get("/health")
def health():
    return {"status": "ok", "model": _cfg.model_id, "device": _cfg.device}

@app.post("/generate")
def generate(req: GenerateRequest):
    state = runner.prefill(req.prompt)
    dec = runner.decode(state, max_new_tokens=req.max_new_tokens, temperature=req.temperature)
    text = runner.decode_text(dec["output_ids"])

    total_s = state["prefill_s"] + dec["decode_s"]
    gen_n = len(dec["new_token_ids"])
    tps = (gen_n / dec["decode_s"]) if dec["decode_s"] > 0 else None

    return {
        "model": _cfg.model_id,
        "device": _cfg.device,
        "timing": {
            "prefill_s": state["prefill_s"],
            "decode_s": dec["decode_s"],
            "total_s": total_s,
            "tokens_per_second": tps,
        },
        "generation": {
            "requested_new_tokens": req.max_new_tokens,
            "generated_new_tokens": len(dec["new_token_ids"]),
            "new_token_ids_preview": dec["new_token_ids"][:10],
            "stopped_early": len(dec["new_token_ids"]) < req.max_new_tokens,
        },
        "text": text,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("efflab.engine.server_fastapi:app", host="127.0.0.1", port=8000, reload=True)

