import torch
import requests
from PIL import Image

from gemini import Gemini
from muMuQA import MuMuQA
from clip import get_features_from_image_urls

muMuQA = MuMuQA()

db_image_urls = muMuQA.get_image_urls()
db_image_features = get_features_from_image_urls(db_image_urls, count=10)

query_image = "https://imagez.tmz.com/image/b1/4by3/2012/01/20/b1b4990ab1b650fc8082838df4e5d75a_xl.jpg"
query_image_feature = get_features_from_image_urls([query_image])

k = 3
similarities = []
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
for db_image_feature in db_image_features:
    similarities.append(cos(db_image_feature, query_image_feature))

data_with_indices = [(tensor.item(), i) for i, tensor in enumerate(similarities)]
sorted_data = sorted(data_with_indices, key=lambda x: x[0], reverse=True)
top_k_indices = [index for value, index in sorted_data[:k]]

print(f"Example similarities:\n {similarities}")
print(f"Top indexes: {top_k_indices}")

# build context from MuRAG
docs = muMuQA.get_examples_by_indexes(top_k_indices)
context = "Conscisely answer the question by inferring an answer from the following knowledge: \n\n"
for doc in docs:
    context += doc['context'] + " " + doc['caption'] + "\n"


query_text = "Who did the person in this image help and how did he help?"
query_image = Image.open(requests.get(query_image, stream=True).raw)

gemini = Gemini()
response = gemini.query([query_text, query_image])
print(response)
response = gemini.query([context, query_text, query_image])
print(response)