import os
from datetime import datetime

class Logger:
    def __init__(self, log_path="logs/logs.txt"):
        self.log_path = log_path
        self._ensure_log_directory_exists()
        self.log("================================================") # add line break on every run

    def _ensure_log_directory_exists(self):
        # Get the directory from the log path
        log_dir = os.path.dirname(self.log_path)
        # Check if the directory exists
        if not os.path.exists(log_dir):
            # Create the directory if it doesn't exist
            os.makedirs(log_dir)

    def log(self, message):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{current_time}] {message}"
        with open(self.log_path, 'a') as log_file:
            log_file.write(log_message + "\n")