import os
import sys
from typing import Union
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.gemini import Gemini
from utils.device_selector import device_selector
from utils.media.image import load_image
from utils.singleton_decorator import singleton


gemini = Gemini()


@singleton
class LlamaVision:
    def __init__(self, device="auto"):
        self.device = device_selector(device, label="Llama Vision")
        self.model = MllamaForConditionalGeneration.from_pretrained(
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.processor = AutoProcessor.from_pretrained(
            "meta-llama/Llama-3.2-11B-Vision-Instruct"
        )

    def query(self, text: str, images: list[Union[str, Image.Image]]) -> str:
        with torch.no_grad():
            images = [load_image(image) for image in images]
            image_entries = [{"type": "image"} for _ in images]
            prompt = [
                {
                    "role": "user",
                    "content": image_entries
                    + [
                        {
                            "type": "text",
                            "text": text,
                        },
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(
                prompt, add_generation_prompt=True
            )
            inputs = self.processor(images, prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(**inputs, max_new_tokens=500)
            response = self.processor.decode(output[0])
            response = response.split("\n")[4].replace("<|eot_id|>", "")
            return response


if __name__ == "__main__":
    llama = LlamaVision()
    image = load_image(
        "/home/jkgao/Documents/GitHub/mirg/data/public_inspection_reports/data/200030002008022/viewreport.ashx-5/images/dark_bridge_woo_graffiti_muddy_waterfront.png"
    )
    response1 = llama.query(
        "What is in this image?",
        [image],
    )
    print(response1, "\n")

    image2 = load_image(
        "/home/jkgao/Documents/GitHub/mirg/data/public_inspection_reports/data/200030002008022/viewreport.ashx-3/images/urban_overpass_graffiti_woon_muddy_ground.png"
    )
    response2 = llama.query("What is in this image?", [image2])
    print(response2, "\n")

    response = gemini.query(
        [
            "Description 1:",
            response1,
            "Description 2:",
            response2,
            "Given these two descriptions of two different images, generate a description of what changed between the two images. Output:",
        ]
    )
    print("Gemini:", response)

    print("\n=============\n")

    response = llama.query("What change between these two images?", [image, image2])
    print(response)
