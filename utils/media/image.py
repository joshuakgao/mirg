import base64
import io
import os
import requests
from PIL import Image, UnidentifiedImageError


def download_image(url):
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


def save_base64_image(base64_str, output_dir, img_format="jpeg"):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Decode the base64 string
    image_data = base64.b64decode(base64_str)

    # Generate a unique filename
    img_filename = f"image_{hash(base64_str)}.{img_format}"
    img_path = os.path.join(output_dir, img_filename)

    # Save the image to the file
    with open(img_path, "wb") as img_file:
        img_file.write(image_data)

    return img_filename
