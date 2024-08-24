import os
import sys
import numpy as np
import open_clip
import torch
from PIL import Image
from typing import List, List

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # for importing paths
from paths import ROOT_DIR
from utils.device_selector import device_selector
from utils.media.image import load_image


class Clip:
    def __init__(self, model_id="B-32", device="auto"):
        self.device = device_selector(device, label="CLIP")
        model, preprocess, tokenizer = self._select_clip_model(model_id)
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    def _select_clip_model(self, model_id):
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

    def encode_images(self, images: List[Image.Image] = []):
        embeddings = []
        for image in images:
            pil_image = load_image(image)
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            pil_image.close()
            with torch.no_grad():
                image_embedding = self.model.encode_image(image)
            embeddings.append(image_embedding)
        embeddings = torch.cat(embeddings, dim=0)

        return embeddings

    def encode_text(self, text: List[str] = []):
        text_input = self.tokenizer(text).to(self.device)
        with torch.no_grad():
            text_embedding = self.model.encode_text(text_input)
        return text_embedding

    def image_classification(self, image: Image.Image, classes: List[str]):
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


clip = Clip("G-14")

if __name__ == "__main__":
    img = Image.open(os.path.join(ROOT_DIR, "assets/bridge_damage.jpg"))
    image_features = clip.encode_images([img])
    text_features = clip.encode_text(["Some string here", "Another string"])
    print(image_features)
    print(text_features)

    classification = clip.image_classification(img, ["concrete", "document"])
    print(classification)
