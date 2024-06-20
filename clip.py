import open_clip
import torch
from PIL import Image
import requests
import numpy as np

def select_clip_model(model):
    model_ids = {
        "B-32": 'hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K',
        "L-14": 'hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K',
        "H-14": 'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
        "G-14": 'hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K',
    }

    model_id = model_ids[model]

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_id)
    tokenizer = open_clip.get_tokenizer(model_id)

    return model, preprocess_train, tokenizer

def get_features_from_image_urls(image_urls, model="B-32", count=None):
    model, preprocess, tokenizer = select_clip_model(model)

    images = []
    n = 0
    for image_url in image_urls:
        print(f"Downloading {image_url}")
        image = Image.open(requests.get(image_url, stream=True).raw)
        image = preprocess(image)
        images.append(image)

        n += 1
        if count is not None and count == n:
            break
        
    image_input = torch.tensor(np.stack(images))
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
    return image_features


if __name__ == '__main__':
    image_urls = ["https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/National_Museum_of_the_American_Indian_in_Washington%2C_D.C.jpg/800px-National_Museum_of_the_American_Indian_in_Washington%2C_D.C.jpg", "http://images.cocodataset.org/val2017/000000039769.jpg"]
    image_features = get_features_from_image_urls(image_urls)
    print(image_features)

    image_path = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/07/National_Museum_of_the_American_Indian%2C_Washington%2C_D.C_LCCN2011630892.tif/lossy-page1-1200px-National_Museum_of_the_American_Indian%2C_Washington%2C_D.C_LCCN2011630892.tif.jpg"
    image_search_embedding = get_features_from_image_urls([image_path])
    print(image_search_embedding)

    distance = []
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for image_feature in image_features:
        distance.append(cos(image_feature, image_search_embedding))
    print(distance)