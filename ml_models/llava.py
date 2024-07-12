import os
import sys
from typing import Sequence
import ollama
from ollama import Message

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # for importing paths
from paths import ROOT_DIR


class Llava:
    def __init__(self, model="llava", verbose=True):
        self.v = verbose
        self.model = model

        if self.v:
            print(
                f"Pulling {model} model. This may take a while if not already downloaded..."
            )
        ollama.pull(model=model)  # download model if not already downloaded

    def query(self, messages: Sequence[Message]):
        if self.v:
            print(f"Querying {self.model}...")

        response = ollama.chat(model="llava", messages=messages)
        return response["message"]["content"]


if __name__ == "__main__":
    llava = Llava()
    response = llava.query(
        messages=[
            {
                "role": "user",
                "content": "What is in this image?",
                "images": [ROOT_DIR + "/assets/bridge_damage.jpg"],
            },
        ]
    )
    print(response)
