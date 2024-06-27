import open_clip
import torch
import numpy as np

class Clip:
    def __init__(self, model_id="B-32"):
        model, prepocess, tokenizer = self._select_clip_model(model_id)
        self.model = model
        self.preprocess = prepocess
        self.tokenizer = tokenizer


    def _select_clip_model(self, model_id):
        model_ids = {
            "B-32": 'hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K',
            "L-14": 'hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K',
            "H-14": 'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
            "G-14": 'hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K',
        }
        
        model_id = model_ids[model_id]
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_id)
        tokenizer = open_clip.get_tokenizer(model_id)

        return model, preprocess_train, tokenizer


    def encode(self, images=[], text=[]):
        images = [self.preprocess(image) for image in images]

        images_input = torch.tensor(np.stack(images))
        text_input = self.tokenizer(text)
        with torch.no_grad():
            images_features = self.model.encode_image(images_input)
            text_features = self.model.encode_text(text_input)
        return images_features, text_features


if __name__ == '__main__':
    ...