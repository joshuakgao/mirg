import requests
from PIL import Image

from gemini import Gemini
from muMuQA import MuMuQA
from muRag import muRag

muMuQA = MuMuQA() # init db

# run MuRag on query image and images in database
query_image_url = "https://imagez.tmz.com/image/b1/4by3/2012/01/20/b1b4990ab1b650fc8082838df4e5d75a_xl.jpg"
db_image_urls = muMuQA.get_image_urls()
top_k_indexes = muRag(query_image_url, db_image_urls)

# build context from MuRAG
docs = muMuQA.get_examples_by_indexes(top_k_indexes)
context = "Conscisely answer the question by inferring an answer from the following knowledge: \n\n"
for doc in docs:
    context += doc['context'] + " " + doc['caption'] + "\n"

# query Gemini
gemini = Gemini()
query_text = "Who did the person in this image help and how did he help?"
query_image = Image.open(requests.get(query_image_url, stream=True).raw)

# without MuRAG
response = gemini.query([query_text, query_image])
print(f"\n{response}\n")

# with MuRAG
response = gemini.query([context, query_text, query_image])
print(f"{response}\n")