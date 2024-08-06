import os
import sys
import google.generativeai as genai

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # for importing paths
from paths import ROOT_DIR
from utils.logger import logger
from utils.api_calling.rate_limiter import RateLimiter

# for typing
from typing import List, Union
from PIL import Image


class Gemini:
    def __init__(self):
        self.rate_limiter = RateLimiter(max_calls=15, period=60)  # 15 calls per minute
        genai.configure(api_key=os.getenv("GEMINI_KEY"))
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    def query(self, query: List[Union[Image.Image, str]]):
        self.rate_limiter.acquire()
        logger.log("Querying gemini api...")

        response = self.model.generate_content(query)
        return response.text

    def caption_image(
        self,
        image: Image.Image,
        context: str = "Describe in detail what is in this image",
    ):
        self.rate_limiter.acquire()
        logger.log("Captioning image...")

        caption = self.model.generate_content([context, image])
        return caption.text


gemini = Gemini()


if __name__ == "__main__":
    img = Image.open(os.path(ROOT_DIR, "assets/bridge_damage.jpg"))
    response = gemini.query([img, "Describe the damages"])
    print(response)
