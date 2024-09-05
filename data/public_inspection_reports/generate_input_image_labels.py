import os
import sys
import json
from PIL import Image
import pprint as pp
import google.generativeai as genai

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)  # for importing paths
from paths import ROOT_DIR
from ml_models.gemini import Gemini

gemini = Gemini()

input_images_dir = "/home/jkgao/Documents/GitHub/mirg/data/public_inspection_reports/data/input/VT15-00020-CAMBRIDGE"
report1_images_dir = "/home/jkgao/Documents/GitHub/mirg/data/public_inspection_reports/data/200030002008022/viewreport.ashx-3"
report2_images_dir = "/home/jkgao/Documents/GitHub/mirg/data/public_inspection_reports/data/200030002008022/viewreport.ashx-5"

all_image_dirs = [input_images_dir, report1_images_dir, report2_images_dir]

input_images = []
report1_images = []
report2_images = []


for i, images_dir in enumerate(all_image_dirs):
    for image_name in os.listdir(images_dir):
        if not image_name.endswith(".png"):
            continue

        image_path = os.path.join(images_dir, image_name)
        # image = Image.open(os.path.join(images_dir, image_name))
        print(f"uploading {image_name}")
        f = genai.upload_file(image_path)

        if i == 0:
            input_images.append(f)
        elif i == 1:
            report1_images.append(f)
        elif i == 2:
            report2_images.append(f)


with open("input_images.json", "w") as f:
    json.dump(f)

report1_images = report1_images[7:14]
report2_images = report2_images[11:28]

input_image = input_images[13]


response = gemini.query(
    query=["Image set 1:"]
    + report1_images
    + ["Image set 2:"]
    + report2_images
    + [
        "Label the following image based on what the previous engineers have labeled the above images of the same bridge:",
        input_image,
    ]
)
print(response)
