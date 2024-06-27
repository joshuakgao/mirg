import numpy as np
from ml_models.clip import Clip


clip = Clip()


def muRag(query_image, query_text, faiss_index, k=3):
    # get image and text embeddings
    query_image_embeddings, query_text_embeddings = clip.encode(images=[query_image], text=[query_text])
    query_embedding = np.concatenate((query_image_embeddings[0], query_text_embeddings[0]))
    query_embedding = np.expand_dims(query_embedding, axis=0)

    # get top k most similar examples' indexes
    distances, indexes = faiss_index.search(query_embedding, k)
    return indexes[0]