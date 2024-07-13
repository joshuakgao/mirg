import os
import sys
from PIL import Image
from ml_models.gemini import Gemini

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # for importing paths
from paths import DATASETS_DIR


gemini = Gemini()

print("Getting report images...")
IMAGES_DIR = (
    DATASETS_DIR + "/inspection_reports/data/14-246-AA03-51-003_FCReport_2023-06/images"
)

images = []
for file_name in os.listdir(IMAGES_DIR):
    # ignore non-pdf files
    if not file_name.endswith(".jpeg"):
        continue

    image_path = os.path.join(IMAGES_DIR, file_name)
    img = Image.open(image_path)
    images.append(file_name)
    images.append(img)

print("Generating markdown inspection report...")
instructions = "Generate an inspection report as a markdown document a civil engineer would need to create given the following images. Provide some maintenance suggestions based on each image. Include the images in the report by embedding it with the image's name."
response = gemini.query([instructions] + images)

f = open("generated_inspection_report.md", "w+")
f.write(response)
f.close()
