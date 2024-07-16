import os
import sys
from typing import Sequence
import ollama
from ollama import Message
from typing import Literal

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # for importing paths
from paths import ROOT_DIR
from utils.logger import logger


class Llava:
    def __init__(
        self,
        model: Literal[
            "llava", "llava:13b", "llava:34b", "llava-llama3", "bakllava", "llava-phi3"
        ] = "llava",
    ):
        self.model = model

        logger.log(
            f"Pulling {model} model. This may take a while if not already downloaded..."
        )
        ollama.pull(model=model)  # download model if not already downloaded

    def query(self, messages: Sequence[Message]):
        logger.log(f"Querying {self.model}...")

        response = ollama.chat(model="llava", messages=messages)
        return response["message"]["content"]


llava = Llava()

if __name__ == "__main__":
    image = os.path.join(ROOT_DIR, "assets/bridge_damage.jpg")
    response = llava.query(
        messages=[
            {
                "role": "user",
                "content": "What is in this image?",
                "images": [image],
            },
        ]
    )
    print(response)
