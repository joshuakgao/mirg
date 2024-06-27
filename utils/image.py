from PIL import Image, UnidentifiedImageError
import io
import requests
import numpy as np

async def download_image_async(session, url):
    async with session.get(url) as response:
        if response.status == 200:
            image_data = await response.read()
            return Image.open(io.BytesIO(image_data))
        else:
            # Handle unsuccessful download (raise exception, log error, etc.)
            raise Exception(f"Failed to download image from {url}")

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