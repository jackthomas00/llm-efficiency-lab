class KVCache:
    def __init__(self):
        self.store = {}

    def get(self, request_id):
        return self.store.get(request_id)

    def set(self, request_id, value):
        self.store[request_id] = value

    def evict(self, request_id):
        self.store.pop(request_id, None)
