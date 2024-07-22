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


class Llama3:
    def __init__(
        self,
        model: Literal["llama3", "llama3:70b"] = "llama3",
    ):
        self.model = model

        logger.log(
            f"Pulling {model} model. This may take a while if not already downloaded..."
        )
        ollama.pull(model=model)  # download model if not already downloaded

    def query(self, messages: Sequence[Message]):
        logger.log(f"Querying {self.model}...")

        response = ollama.chat(model=self.model, messages=messages)
        return response["message"]["content"]


llama3 = Llama3("llama3")

if __name__ == "__main__":
    response = llama3.query(
        messages=[
            {
                "role": "user",
                "content": "How old is the sun?",
            },
        ]
    )
    print(response)
