import os
import sys
import google.generativeai as genai

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # for importing paths
from paths import ROOT_DIR

# for typing
from typing import List, Union
from PIL import Image


class Gemini:
    def __init__(self, verbose=True):
        self.v = verbose
        genai.configure(api_key=os.getenv("GEMINI_KEY"))
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    def query(self, query: List[Union[Image.Image, str]]):
        if self.v:
            print("Querying gemini api...")

        response = self.model.generate_content(query)
        return response.text


if __name__ == "__main__":
    gemini = Gemini()
    img = Image.open(ROOT_DIR + "/assets/bridge_damage.jpg")
    response = gemini.query([img, "Describe the damages"])
    print(response)
