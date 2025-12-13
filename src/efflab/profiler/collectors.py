import time

def measure(fn):
    start = time.time()
    fn()
    return time.time() - start
