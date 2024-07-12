import os
import sys
import numpy as np
import open_clip
import torch
from PIL import Image

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # for importing paths
from paths import ROOT_DIR


class Clip:
    def __init__(self, model_id="B-32"):
        model, preporcess, tokenizer = self._select_clip_model(model_id)
        self.model = model
        self.preprocess = preporcess
        self.tokenizer = tokenizer

    def _select_clip_model(self, model_id):
        """
        Return clip encoder and
        """
        model_ids = {
            "B-32": "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",  # smallest clip model
            "L-14": "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
            "H-14": "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            "G-14": "hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K",  # largest clip model
        }
        model_id = model_ids[model_id]
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_id
        )
        tokenizer = open_clip.get_tokenizer(model_id)

        return model, preprocess_train, tokenizer

    def encode(self, images=[], text=[]):
        """
        Encode list of images and list of text with clip
        """
        images = [self.preprocess(image) for image in images]

        images_input = torch.tensor(np.stack(images))
        text_input = self.tokenizer(text)
        with torch.no_grad():
            images_features = self.model.encode_image(images_input)
            text_features = self.model.encode_text(text_input)
        return images_features, text_features


if __name__ == "__main__":
    clip = Clip()
    img = Image.open(ROOT_DIR + "/assets/bridge_damage.jpg")
    images_features, text_features = clip.encode(
        images=[img], text=["Some string here", "Another string"]
    )
    print(images_features)  # list of 512x1 vectors
    print(text_features)  # list of 512x1 vectors
