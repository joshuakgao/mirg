import os
import sys
from typing import Sequence
import ollama
from ollama import Message, Options
from typing import Literal
from PIL import Image
from typing import Union

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # for importing paths
from paths import ROOT_DIR
from utils.logger import logger
from utils.media.image import load_image


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
        # ollama.pull(model=model)  # download model if not already downloaded

    def query(
        self,
        messages: Sequence[Message],
        options: Options,
    ):
        logger.log(f"Querying {self.model}...")

        response = ollama.chat(model=self.model, messages=messages, options=options)
        return response["message"]["content"]

    def caption_image(
        self,
        image: Union[str, Image.Image],
        context: str = "Describe in detail what is in this image",
        options: Options = {},
    ):
        image = load_image(image, to_bytes=True)

        caption = self.query(
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": "Image:", "images": [image]},
            ],
            options=options,
        )

        return caption


llava = Llava("llava")

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
