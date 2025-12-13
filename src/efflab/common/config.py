from pydantic import BaseModel

class ModelConfig(BaseModel):
    model_id: str = "distilgpt2"
    device: str = "cpu"   # change to "cuda" if you have it
    dtype: str = "float32"
    max_new_tokens_default: int = 64
