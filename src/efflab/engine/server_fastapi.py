from fastapi import FastAPI
from efflab.engine.request import InferenceRequest

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate(prompt: str, max_tokens: int = 32):
    req = InferenceRequest("demo", prompt, max_tokens)
    return {"request_id": req.request_id}
