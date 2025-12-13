class InferenceRunner:
    def __init__(self, model, tokenizer, kv_cache):
        self.model = model
        self.tokenizer = tokenizer
        self.kv_cache = kv_cache

    def prefill(self, req):
        # TODO: tokenize + run initial forward pass
        pass

    def decode_step(self, req):
        # TODO: single-token decode using KV cache
        pass
