from transformers import pipeline
from PIL import Image
import requests
from typing import Union
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)  # for importing paths
from utils.media.image import load_image
from utils.device_selector import device_selector


class DeepAnything2:
    def __init__(self, device="auto"):
        self.device = device_selector(device, label="DeepAnything2")
        self.pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Large-hf",
            device=self.device,
        )

    def inference(self, image: Union[str, Image.Image]):
        image = load_image(image)
        depth = self.pipe(image)
        print(depth)
        depth["depth"].show()


deepAnything2 = DeepAnything2()


if __name__ == "__main__":
    img = "/home/jkgao/Documents/GitHub/mu-rag/assets/bridge_damage.jpg"
    deepAnything2.inference(img)
