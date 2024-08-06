import os
import sys
import json

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # for importing paths
from paths import ROOT_DIR

# from ml_models.gemma2 import gemma2
from ml_models.gemini import gemini


print("Getting report images...")
doc_dir = os.path.join(ROOT_DIR, "data/public_inspection_reports/data/2022-07-20_00020")
images_dir = os.path.join(doc_dir, "images")

with open(images_dir + "/metadata.json", "r") as f:
    images_metadata = json.loads(f.read())


image_data = []
for image_name, data in images_metadata.items():
    if data["image_type"] != "concrete":
        continue

    data_str = ""
    data_str = "Image Name: " + image_name + "\n" + "Damages: " + data["damages"]

    image_data.append(data_str)


print("Generating markdown inspection report...")
instructions = "Generate a new inspection report as a markdown document a civil engineer would need to create given the following image names and their description."
response = gemini.query([instructions] + image_data)

f = open(os.path.join(images_dir, "generated_report.md"), "w+")
f.write(response)
f.close()
