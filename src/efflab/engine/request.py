from dataclasses import dataclass
from typing import List

@dataclass
class InferenceRequest:
    request_id: str
    prompt: str
    max_tokens: int
    tokens_generated: int = 0
