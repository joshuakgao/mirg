from data.muMuQA.dataparser import muMuQA
from ml_models.gemini import gemini
from utils.media.image import download_image
from utils.mu_rag import mu_rag


while True:
    # get query from user
    query_text = input("\nText input: ")
    query_image_url = input("Query image url: ")

    # run MuRag on query image and images in database
    query_image = download_image(query_image_url)
    top_k_indexes = mu_rag(query_image, query_text, muMuQA.faiss_index)
    print(top_k_indexes)

    # build context from MuRAG
    docs = muMuQA.get_examples_by_indexes(top_k_indexes)
    context = "The following are mulitple documents that could be related to the question:\n\n"
    for doc in docs:
        context += doc["context"] + ". " + doc["caption"] + "\n\n"

    print(context)

    # query Gemini without MuRAG
    response = gemini.query([query_text, query_image])
    print(f"\nNO MURAG:\n{response}\n")

    # query Gemini with MuRAG
    response = gemini.query([context, query_text, query_image])
    print(f"WITH MURAG:\n{response}\n")
