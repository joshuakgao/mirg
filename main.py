from gemini import Gemini
from muMuQA import MuMuQA
from muRag import muRag
from utils.image import download_image
import os

# fixes a faiss error. See: https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
os.environ['KMP_DUPLICATE_LIB_OK']='True'

muMuQA = MuMuQA() # init db
gemini = Gemini() # init Gemini LLM

while True:
    # query Gemini
    query_text = input("\nText input: ")
    query_image_url = input("Query image url: ")

    # run MuRag on query image and images in database
    query_image = download_image(query_image_url)
    top_k_indexes = muRag(query_image, query_text, muMuQA.faiss_index)
    print(top_k_indexes)

    # build context from MuRAG
    docs = muMuQA.get_examples_by_indexes(top_k_indexes)
    context = ""
    for doc in docs:
        context += doc['context'] + ". " + doc['caption'] + "\n"

    print(context)

    # query Gemini without MuRAG
    response = gemini.query([query_text, query_image])
    print(f"\n{response}\n")

    # query Gemini with MuRAG
    response = gemini.query([context, query_text, query_image])
    print(f"{response}\n")