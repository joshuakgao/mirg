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
from ml_models.clip import Clip


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


def remove_duplicate_images(images_dir: str, confidence=0.8):
    clip = Clip(device="cpu")

    # read images
    images = []
    resolutions = []
    images_paths = []
    for filename in os.listdir(images_dir):
        # ignore non-image files
        if not filename.endswith(("png", "jpg", "jpeg")):
            continue

        # build image paths
        image_path = os.path.join(images_dir, filename)
        images_paths.append(image_path)

        # open images
        image = Image.open(image_path)
        resolutions.append(image.size)
        images.append(image)

    # get embeddings for all images
    images_embeddings, _ = clip.encode(images=images)
    images_embeddings = images_embeddings.numpy()
    # print(images_embeddings)
    faiss.normalize_L2(images_embeddings)

    # create faiss index
    faiss_index = faiss.IndexFlatIP(images_embeddings.shape[1])
    faiss_index.add(images_embeddings)

    # query faiss index
    distances, indexes = faiss_index.search(images_embeddings, k=2)

    # identify duplicates
    duplicates = {}
    for i, (distance, index) in enumerate(zip(distances, indexes)):
        # distance[0] is the image itself, so we check distance[1]
        if distance[1] > confidence:
            duplicate_path = images_paths[index[1]]
            original_path = images_paths[i]

            # compare resolutions
            if duplicate_path not in duplicates:
                if resolutions[i] > resolutions[index[1]]:
                    duplicates[duplicate_path] = original_path
                else:
                    duplicates[duplicate_path] = None  # mark for deletion
            else:
                if resolutions[i] > resolutions[index[1]]:
                    duplicates[duplicate_path] = original_path

    deleted = set()
    for duplicate, original in duplicates.items():
        if original is None:
            duplicate_path = os.path.join(images_dir, duplicate)
            os.remove(duplicate_path)
            print(f"Removed duplicate image: {duplicate_path}")
        elif original not in deleted:
            original_path = os.path.join(images_dir, original)
            deleted.add(original)
            print(f"Kept one copy of duplicate image: {original_path}")
        else:
            duplicate_path = os.path.join(images_dir, duplicate)
            os.remove(duplicate_path)
            print(f"Removed duplicate image: {duplicate_path}")


if __name__ == "__main__":
    duplicates = remove_duplicate_images(
        "/home/tugonbob/Documents/GitHub/UH/mu-rag/data/inspection_reports/data/12-020-M018-20-001_RTInsp_2023-08/images"
    )
    pp.pprint(duplicates)
