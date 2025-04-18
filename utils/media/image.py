import base64
import io
import os
import requests
from PIL import Image, UnidentifiedImageError
import faiss
from typing import Union
from io import BytesIO
import torch


def load_image(image: Union[str, Image.Image]):
    if isinstance(image, str):
        is_url = image.startswith(("http://", "https://"))
        if is_url:
            # If input_data is a URL, download the image
            response = requests.get(image)
            response.raise_for_status()  # Check for HTTP errors
            image = Image.open(BytesIO(response.content))
        elif os.path.isfile(image):
            # If image is a file path, open the image
            image = Image.open(image)
        else:
            raise ValueError("Input string must be a valid file path or URL")
    elif isinstance(image, Image.Image):
        # If image is already an Image object, use it directly
        image = image
    else:
        raise ValueError("Input must be a file path or a PIL Image object")

    image = image.copy()
    return image


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
    """
    Saves a base64 image to a file in output_dir
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Decode the base64 string
    image_data = base64.b64decode(base64_str)

    # Convert image data bytes to PIL Image
    image = Image.open(io.BytesIO(image_data))

    # Generate a unique filename
    img_filename = f"image_{hash(base64_str)}.{image_format}"
    img_path = os.path.join(output_dir, img_filename)

    # Save the image to the file
    with open(img_path, "wb") as img_file:
        img_file.write(image_data)

    return img_filename


def convert_image_to_base64(image: Image.Image):
    # Convert the image to bytes
    byte_arr = io.BytesIO()
    image.save(byte_arr, format="PNG")
    byte_arr = byte_arr.getvalue()

    # Convert to base64
    base64_image = base64.b64encode(byte_arr).decode("utf-8")
    return base64_image


def find_duplicate_images(images_dir: str, threshold=0.95):
    from models.clip import Clip

    clip = Clip()

    # read images
    images = []  # [(image_path, image_dimension)]
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
    print("Cuda memory used:", torch.cuda.memory_allocated() / (1024**2))
    images_embeddings = torch.stack([clip.embed_image(path) for path, _ in images])
    images_embeddings = images_embeddings.to("cpu").numpy()
    torch.cuda.empty_cache()
    print("Cuda memory used:", torch.cuda.memory_allocated() / (1024**2))

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


def get_overlapping_crops(pil_image, num_crops, overlap):
    # calculate stride based on num_crops and overlap
    image_width, image_height = pil_image.size
    num_crops_x, num_crops_y = num_crops

    # Calculate the effective area covered by the crops minus the overlap
    crop_width = image_width / (num_crops_x - overlap * (num_crops_x - 1))
    crop_height = image_height / (num_crops_y - overlap * (num_crops_y - 1))

    crop_width = int(crop_width)
    crop_height = int(crop_height)

    # Calculate the stride (step size between crops)
    stride_x = crop_width * (1 - overlap)
    stride_y = crop_height * (1 - overlap)

    crops = []

    # Iterate over the image based on the calculated strides
    for i in range(num_crops_y):
        for j in range(num_crops_x):
            left = min(int(j * stride_x), image_width - crop_width)
            top = min(int(i * stride_y), image_height - crop_height)

            # Define the box for the crop: (left, top, right, bottom)
            crop_box = (left, top, left + crop_width, top + crop_height)

            # Crop the image and append to the list
            crop = pil_image.crop(crop_box)
            crops.append(crop)

    return crops


if __name__ == "__main__":
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = load_image(url)
    image.show()
    image.close()

    image_path = "/home/jkgao/Documents/GitHub/mu-rag/assets/bridge_damage.jpg"
    image = load_image(image_path)
    image.show()
    image.close()
