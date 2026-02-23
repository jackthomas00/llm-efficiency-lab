from __future__ import annotations
import asyncio
import json
import time
from fastapi import FastAPI, Request, Query
from fastapi.responses import StreamingResponse
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

@app.post("/generate_stream")
async def generate_stream(
    req: GenerateRequest,
    request: Request,
    delay_ms: int = Query(0, description="Delay between tokens (ms) for testing streaming"),
):
    q: asyncio.Queue[dict] = asyncio.Queue()

    work_item = WorkItem(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        repetition_penalty=req.repetition_penalty,
        no_repeat_ngram_size=req.no_repeat_ngram_size,
        use_kv_cache=req.use_kv_cache,
        stream=True,
        out_q=q,
    )

    await batcher.enqueue(work_item)

    async def event_gen():
        t0 = time.perf_counter()
        try:
            while True:
                # Client disconnected?
                if await request.is_disconnected():
                    work_item.cancelled = True
                    break

                msg = await q.get()
                typ = msg.get("type")
                msg["ts"] = round(time.perf_counter() - t0, 4)

                if typ == "token":
                    if delay_ms > 0:
                        await asyncio.sleep(delay_ms / 1000.0)
                    yield f"event: token\ndata: {json.dumps(msg)}\n\n"
                elif typ == "done":
                    yield f"event: done\ndata: {json.dumps(msg)}\n\n"
                    break
                elif typ == "error":
                    yield f"event: error\ndata: {json.dumps(msg)}\n\n"
                    break
                else:
                    # Unknown message type, still forward it
                    yield f"event: message\ndata: {json.dumps(msg)}\n\n"

        except asyncio.CancelledError:
            work_item.cancelled = True
            raise

    return StreamingResponse(event_gen(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("efflab.engine.server_fastapi:app", host="127.0.0.1", port=8000, reload=True)
