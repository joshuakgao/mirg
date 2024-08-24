import re


def to_snake_case(s: str) -> str:
    # Replace spaces and hyphens with underscores
    s = re.sub(r"[\s\-]+", "_", s)
    # Convert to lowercase
    s = s.lower()
    # Remove any leading or trailing underscores
    s = s.strip("_")
    return s
