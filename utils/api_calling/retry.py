import time


def retry(operation, retries=3, delay=1):
    """
    Retries the given operation if it raises an exception.

    Args:
        operation (callable): The operation to be retried.
        retries (int): The maximum number of retries.
        delay (int): The delay between retries in seconds.

    Returns:
        The result of the operation if it succeeds.

    Raises:
        Exception: The last exception raised if all retries fail.
    """
    for attempt in range(retries):
        try:
            return operation()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt + 1 == retries:
                raise  # Re-raise the last exception if out of retries
            time.sleep(delay)
