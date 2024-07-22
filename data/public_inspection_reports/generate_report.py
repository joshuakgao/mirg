import os
import sys
import json
from PIL import Image

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)  # for importing paths
from paths import ROOT_DIR

from ml_models.gemini import gemini

print("Getting report images...")
doc_dir = os.path.join(ROOT_DIR, "data/public_inspection_reports/data/2023-05-22_00004")
images_dir = os.path.join(doc_dir, "images")

refernce_dir = os.path.join(
    ROOT_DIR, "data/public_inspection_reports/data/2022-07-20_00020"
)
reference_images_dir = os.path.join(refernce_dir, "images")

with open(images_dir + "/metadata.json", "r") as f:
    images_metadata = json.loads(f.read())

with open(reference_images_dir + "/metadata.json", "r") as f:
    reference_images_metadata = json.loads(f.read())
with open(refernce_dir + "/page1.md") as f:
    reference_md = f.read()

image_data = []
for image_name, data in images_metadata.items():
    image = Image.open(images_dir + "/" + image_name)
    image_data.append(image)

    data_str = ""
    data_str = "Image Name: " + image_name + "\n" + "Image Caption: " + data["caption"]

    if data["image_type"] == "concrete":
        data_str += " " + data["damages"]["caption"]

    image_data.append(data_str)

context_data = []
for image_name, data in reference_images_metadata.items():
    data_str = ""
    data_str = "Image Name: " + image_name + "\n" + "Image Caption: " + data["caption"]
    context_data.append(data_str)


print(len(image_data))
print("Generating markdown inspection report...")
instructions = "Generate a new inspection report following the example as a markdown document a civil engineer would need to create given the following image names and their description. Embed the image names into the markdown file.\n\nExample:\n"
response = gemini.query(
    query=image_data
    + ["Generate the inspetion report in markdown with the image embeddings"]
)

f = open("generated_inspection_report.md", "w+")
f.write(response)
f.close()
