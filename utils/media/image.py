import base64
import io
import os
import sys
import requests
from PIL import Image, UnidentifiedImageError
import faiss
import pprint as pp

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)  # for importing paths
from ml_models.clip import clip
from utils.logger import logger


def download_image(url: str):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the request was successful
        image = Image.open(io.BytesIO(response.content))
        return image
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    except UnidentifiedImageError:
        print(f"UnidentifiedImageError: cannot identify image file at {url}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None


def save_base64_image(base64_str: str, output_dir: str, image_format="jpeg"):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Decode the base64 string
    image_data = base64.b64decode(base64_str)

    # Generate a unique filename
    img_filename = f"image_{hash(base64_str)}.{image_format}"
    img_path = os.path.join(output_dir, img_filename)

    # Save the image to the file
    with open(img_path, "wb") as img_file:
        img_file.write(image_data)

    return img_filename


def find_duplicate_images(images_dir: str, threshold=0.9):
    # read images
    images = []  # [( PIL_IMAGE, dimensions as (w,h), file_path)]
    for filename in os.listdir(images_dir):
        # ignore non-image files
        if not filename.endswith(("png", "jpg", "jpeg")):
            continue

        # build image paths
        image_path = os.path.join(images_dir, filename)

        # open images
        image = Image.open(image_path)
        images.append((image_path, image.size))
        image.close()

    # get embeddings for all images
    images_embeddings, _ = clip.encode(
        images=[Image.open(path) for path, dims in images]
    )
    images_embeddings = images_embeddings.to("cpu").numpy()
    # print(images_embeddings)
    faiss.normalize_L2(images_embeddings)

    # create faiss index
    faiss_index = faiss.IndexFlatIP(images_embeddings.shape[1])
    faiss_index.add(images_embeddings)

    # calc distances of each image to every other image
    all_similarities, all_indexes = faiss_index.search(images_embeddings, k=len(images))

    # identify duplicates
    # list of sets of tuples
    # [
    #     {(path1, dims1), (path2, dims2)},
    #     {(path3, dims3), ...}
    #     ...
    # ]
    duplicates = []
    for original_index, (similarities, indexes) in enumerate(
        zip(all_similarities, all_indexes)
    ):
        for j, (similarity, comparison_index) in enumerate(zip(similarities, indexes)):
            # skip first comparision, because it is image compared to itself
            if j == 0:
                continue

            # skip image comparisons that have a lower similarity than the threshold
            if similarity <= threshold:
                continue

            original_image = images[original_index]
            comparison_image = images[comparison_index]

            matching_duplicate_group_found = False
            for i, duplicate_group in enumerate(duplicates):
                for image in duplicate_group:
                    if image == original_image or image == comparison_image:
                        duplicates[i].add(original_image)
                        duplicates[i].add(comparison_image)
                        matching_duplicate_group_found = True
                        break
                if matching_duplicate_group_found:
                    break

            if not matching_duplicate_group_found:
                duplicates.append(set([original_image, comparison_image]))

    # get the path of largest image and delete the others
    duplicate_groups = {}
    for duplicate_group in duplicates:
        max_dimensions = max(dims for path, dims in duplicate_group)
        largest_image = next(img for img in duplicate_group if img[1] == max_dimensions)
        duplicate_group.remove(largest_image)
        to_be_deleted = [img[0] for img in duplicate_group]

        duplicate_groups[largest_image[0]] = to_be_deleted

    return duplicate_groups


if __name__ == "__main__":
    deletions = find_duplicate_images(
        "/home/jkgao/Documents/GitHub/mu-rag/data/inspection_reports/data/12-020-2105-02-005_RTInsp_2024-01/images"
    )
    pp.pprint(deletions)
