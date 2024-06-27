import json
from utils.image import download_image
from clip import Clip
import os
from logger import Logger
import faiss
import numpy as np


clip = Clip()
logger = Logger()


class MuMuQA():
    def __init__(self):
        train_path = "./data/muMuQA/train/train.json"
        dev_path = "./data/muMuQA/eval/dev.json"
        test_path = "./data/muMuQA/eval/test.json"
        self.train = json.loads(open(train_path, 'r').read())
        self.dev = json.loads(open(dev_path, 'r').read())
        self.test = json.loads(open(test_path, 'r').read())

        # pre-embed dataset if not already embedded
        self.train = self._pre_encode_split_until_done(self.train, train_path)
        self.dev = self._pre_encode_split_until_done(self.dev, dev_path)
        self.test = self._pre_encode_split_until_done(self.test, test_path)

        # store embeddings into faiss index
        embeddings = np.array([np.concatenate((doc["image_embedding"], doc["text_embedding"])) for doc in self.dev]) # concat image with text embedding
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1]) # provide faiss index with dimension
        self.faiss_index.add(embeddings) # store embeddings
    

    def _pre_encode_split_until_done(self, split, path):
        """
        Pre-encode text and images in dataset until all examples are encoded, even through errors.
        """
        indexes = self._get_non_pre_encoded_indexes(split)
        while len(indexes) != 0:
            split = self._pre_encode_split(split, path, indexes)
            indexes = self._get_non_pre_encoded_indexes(split)
        return split


    def _get_non_pre_encoded_indexes(self, split):
        """
        This checks to see if a split is fully embedded by check if all dicts in array have the "image_embedding" and "text_embedding" key
        """
        return [index for index, d in enumerate(split) if "image_embedding" not in d and "text_embedding" not in d]

    
    def _pre_encode_split(self, split, path, indexes):
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
                image = download_image(doc['image'])
                text = doc['context'] + doc['caption'] # build text data
                image_embeddings, text_embeddings = clip.encode(images=[image], text=[text])

                # save embeddings to dataset
                split[i]['image_embedding'] = image_embeddings[0].tolist()
                split[i]['text_embedding'] = text_embeddings[0].tolist()
            except Exception as e:
                logger.log(f"Error with doc {i} in {file_name}: {e}\n")

            # save progress every 500 steps
            if i % 500 == 0:
                self._save_split_to_file(split, path)

        # final save
        self._save_split_to_file(split, path)
        return split


    def _save_split_to_file(self, split, path):
        with open(path, "w") as f:
            json.dump(split, f, indent=4)


    def get_examples_by_indexes(self, indexes):
        """
        Fetches examples from dataset by index
        """
        return [self.dev[i] for i in indexes]


if __name__ == "__main__":
    muMuQA = MuMuQA()