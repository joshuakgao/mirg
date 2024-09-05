import time


class RateLimiter:
    def __init__(self, max_calls: int, period: int):
        self.max_calls = max_calls
        self.period = period
        self.calls = []

    def acquire(self, label=""):
        current_time = time.time()

        # Remove calls that are outside the period window
        self.calls = [call for call in self.calls if current_time - call < self.period]

        if len(self.calls) >= self.max_calls:
            sleep_time = self.period - (current_time - self.calls[0])
            print(
                f"Rate limit reached for {label}. Sleeping for {sleep_time:.2f} seconds..."
            )
            time.sleep(sleep_time)

        self.calls.append(time.time())
