from __future__ import annotations
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel

from efflab.common.config import ModelConfig
from efflab.engine.model_loader import load_model_and_tokenizer
from efflab.engine.runner import InferenceRunner
from efflab.engine.batcher import Batcher, WorkItem

from contextlib import asynccontextmanager

cfg = ModelConfig()
_model, _tok, _cfg = load_model_and_tokenizer(cfg)
runner = InferenceRunner(_model, _tok, _cfg.device)

batcher = Batcher(runner, max_batch_size=8, batch_timeout_ms=10)

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = cfg.max_new_tokens_default
    temperature: float = 0.8
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 3
    use_kv_cache: bool = True

@asynccontextmanager
async def lifespan(app: FastAPI):
    await batcher.start()
    yield
    await batcher.stop()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "model": _cfg.model_id, "device": _cfg.device}

@app.post("/generate")
async def generate(req: GenerateRequest):
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    work_item = WorkItem(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        repetition_penalty=req.repetition_penalty,
        no_repeat_ngram_size=req.no_repeat_ngram_size,
        use_kv_cache=req.use_kv_cache,
        future=future,
    )

    await batcher.enqueue(work_item)
    return await future

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("efflab.engine.server_fastapi:app", host="127.0.0.1", port=8000, reload=True)
