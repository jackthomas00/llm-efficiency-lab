from dataclasses import dataclass

@dataclass
class InferenceRequest:
    request_id: str
    prompt: str
    max_tokens: int
    tokens_generated: int = 0
