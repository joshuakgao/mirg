import os
import sys

import google.generativeai as genai

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Union

from google.generativeai.types import HarmBlockThreshold, HarmCategory
from PIL import Image

from utils.media.image import load_image
from utils.logger import logger
from utils.api_calling.rate_limiter import RateLimiter


class Gemini:
    def __init__(self):
        self.rate_limiter = RateLimiter(max_calls=15, period=60)  # 15 calls per minute
        genai.configure(api_key=os.getenv("GEMINI_KEY"))
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    def query(self, query: list[Union[Image.Image, str]]) -> str:
        response = None
        while response is None:
            try:
                self.rate_limiter.acquire(label="Gemini")
                response = self.model.generate_content(
                    query,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    },
                ).text.strip()
            except KeyboardInterrupt:
                print("KeyboardInterrupt received. Exiting...")
                sys.exit()
            except Exception as e:
                logger.log(f"Something went wrong with this Gemini call, retrying...")
                logger.log(e)
        return response

    def caption_image(
        self,
        image: Union[Image.Image, str],
        context: str = "Describe in detail what is in this image",
    ) -> str:
        try:
            self.rate_limiter.acquire(label="Gemini")

            image = load_image(image)
            caption = self.query([context, image])
            return caption
        except Exception as e:
            logger.log(e)


if __name__ == "__main__":
    gemini = Gemini()
    image1 = load_image(
        "/home/jkgao/Documents/GitHub/mirg/data/public_inspection_reports/data/images/start.png"
    )
    image2 = load_image(
        "/home/jkgao/Documents/GitHub/mirg/data/public_inspection_reports/data/images_other/image_4894597298081990202.png"
    )
    image3 = load_image(
        "/home/jkgao/Documents/GitHub/mirg/data/public_inspection_reports/data/images_other/concrete_bridge_with_green_railing_over_calm_water.png"
    )
    response = gemini.query(
        [
            image1,
            image1,
            "What has changed between these two images? Respond with 'na' if the images are not the same location. Output:",
        ]
    )
    print(response)
