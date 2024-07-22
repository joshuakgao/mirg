import os
import sys
import google.generativeai as genai

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # for importing paths
from paths import ROOT_DIR
from utils.logger import logger

# for typing
from typing import List, Union
from PIL import Image


class Gemini:
    def __init__(self):
        print(os.getenv("GEMINI_KEY"))
        genai.configure(api_key=os.getenv("GEMINI_KEY"))
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    def query(self, query: List[Union[Image.Image, str]]):
        logger.log("Querying gemini api...")

        response = self.model.generate_content(query)
        return response.text


gemini = Gemini()


if __name__ == "__main__":
    img = Image.open(os.path(ROOT_DIR, "assets/bridge_damage.jpg"))
    response = gemini.query([img, "Describe the damages"])
    print(response)
