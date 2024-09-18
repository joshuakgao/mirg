import os
import sys
from typing import Literal

import open_clip
import torch
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.device_selector import device_selector
from utils.media.image import load_image


class Clip:
    def __init__(self, model_id="G-14", device="auto"):
        self.device = device_selector(device, label="CLIP")
        model, preprocess, tokenizer = self._select_clip_model(model_id)
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    def _select_clip_model(self, model_id: Literal["B-32", "L-14", "H-14", "G-14"]):
        model_ids = {
            "B-32": "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
            "L-14": "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
            "H-14": "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            "G-14": "hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
        }
        model_id = model_ids[model_id]
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_id, device=self.device
        )
        tokenizer = open_clip.get_tokenizer(model_id)

        return model, preprocess_train, tokenizer

    def embed_image(self, image) -> torch.Tensor:
        pil_image = load_image(image)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        pil_image.close()
        with torch.no_grad():
            image_embedding = self.model.encode_image(image)
        return image_embedding.squeeze(0).to("cpu")

    def embed_text(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            text = self.tokenizer(text).to(self.device)
            text_embedding = self.model.encode_text(text)
        return text_embedding.to("cpu")

    def image_classification(self, image: Image.Image, classes: list[str]) -> str:
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        class_embeddings = self.tokenizer(classes).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(class_embeddings)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            classification = max(zip(text_probs[0], classes))[1]

        return classification


if __name__ == "__main__":
    clip = Clip()
    print(
        clip.embed_image(
            "/home/jkgao/Documents/GitHub/change-segmentation-database-creator/data/sa1b/data/sa_11187.jpg"
        )
    )
