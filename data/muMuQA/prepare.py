import json
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)  # for importing utils
from ml_models.clip import clip
from paths import ROOT_DIR
from utils.logger import logger
from utils.media.image import download_image


def pre_encode_split_until_done(split, path):
    """
    Pre-encode text and images in dataset until all examples are encoded, even through errors.
    """
    indexes = get_non_pre_encoded_indexes(split)
    while len(indexes) != 0:
        split = pre_encode_split(split, path, indexes)
        indexes = get_non_pre_encoded_indexes(split)
    return split


def get_non_pre_encoded_indexes(split):
    """
    This checks to see if a split is fully embedded by check if all dicts in array have the "image_embedding" and "text_embedding" key
    """
    return [
        index
        for index, d in enumerate(split)
        if "image_embedding" not in d and "text_embedding" not in d
    ]


def pre_encode_split(split, path, indexes):
    """
    Pre encode text and images in dataset of indexes parameter
    """
    # if split is entirely encoded, return
    if len(indexes) == 0:
        return

    # encode all non-encoded
    file_name = os.path.basename(path)
    for i in indexes:
        try:
            print(f"Encoding {i} of {len(split)} in {file_name}")
            doc = split[i]

            # encode image and text with clip
            image = download_image(doc["image"])
            text = doc["context"] + doc["caption"]  # build text data
            image_embeddings = clip.encode_images([image])
            text_embeddings = clip.encode_text([text])

            # save embeddings to dataset
            split[i]["image_embedding"] = image_embeddings[0].tolist()
            split[i]["text_embedding"] = text_embeddings[0].tolist()
        except Exception as e:
            logger.log(f"Error with doc {i} in {file_name}: {e}\n")

        # save progress every 500 steps
        if i % 500 == 0:
            save_split_to_file(split, path)

    # final save
    save_split_to_file(split, path)
    return split


def save_split_to_file(split, path):
    with open(path, "w") as f:
        json.dump(split, f, indent=4)


if __name__ == "__main__":
    # set data paths
    train_path = os.path.join(ROOT_DIR, "data/muMuQA/train.json")
    dev_path = os.path.join(ROOT_DIR, "data/muMuQA/dev.json")
    test_path = os.path.join(ROOT_DIR, "data/muMuQA/test.json")

    # open raw data
    train = json.loads(open(train_path, "r").read())
    dev = json.loads(open(dev_path, "r").read())
    test = json.loads(open(test_path, "r").read())

    # pre-embed dataset if not already embedded
    train = pre_encode_split_until_done(train, train_path)
    dev = pre_encode_split_until_done(dev, dev_path)
    test = pre_encode_split_until_done(test, test_path)
