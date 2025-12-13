from collections import deque

class Scheduler:
    def __init__(self):
        self.queue = deque()

    def submit(self, req):
        self.queue.append(req)

    def next_batch(self, batch_size=4):
        batch = []
        while self.queue and len(batch) < batch_size:
            batch.append(self.queue.popleft())
        return batch
