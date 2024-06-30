import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # for importing utils
from utils.image import download_image
from utils.logger import Logger
from ml_models.clip import Clip
import json
import faiss
import numpy as np
from config import DATASETS_DIR


# fixes a faiss error. See: https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
os.environ['KMP_DUPLICATE_LIB_OK']='True'


clip = Clip()
logger = Logger()


class MuMuQA():
    def __init__(self):
        # load raw data
        train_path = os.path.join(DATASETS_DIR, "muMuQA/train.json")
        dev_path = os.path.join(DATASETS_DIR, "muMuQA/dev.json")
        test_path = os.path.join(DATASETS_DIR, "muMuQA/test.json")
        self.train = json.loads(open(train_path, 'r').read())
        self.dev = json.loads(open(dev_path, 'r').read())
        self.test = json.loads(open(test_path, 'r').read())

        # store embeddings into faiss index
        embeddings = np.array([
            np.concatenate((doc["image_embedding"], doc["text_embedding"]))
            for doc in self.train
            if "image_embedding" in doc and "text_embedding" in doc
        ]) # concat image with text embedding
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1]) # provide faiss index with dimension
        self.faiss_index.add(embeddings) # store embeddings


    def get_examples_by_indexes(self, indexes):
        """
        Fetches examples from dataset by index
        """
        return [self.train[i] for i in indexes]


if __name__ == "__main__":
    muMuQA = MuMuQA()