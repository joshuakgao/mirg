import torch
from clip import get_features_from_image_urls


def muRag(query_image_url, db_image_urls, k=3):
    query_image_feature = get_features_from_image_urls([query_image_url])
    db_image_features = get_features_from_image_urls(db_image_urls, count=10)

    # get k most similar db examples
    similarities = []
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for db_image_feature in db_image_features:
        similarities.append(cos(db_image_feature, query_image_feature))

    # get top k indexes
    data_with_indices = [(tensor.item(), i) for i, tensor in enumerate(similarities)]
    sorted_data = sorted(data_with_indices, key=lambda x: x[0], reverse=True)
    top_k_indices = [index for value, index in sorted_data[:k]]

    print(f"Example similarities:\n {similarities}")
    print(f"Top indexes: {top_k_indices}")

    return top_k_indices