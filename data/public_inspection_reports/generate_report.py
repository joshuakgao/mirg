import os
import sys
import json
from PIL import Image
import pprint as pp

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)  # for importing paths
from paths import ROOT_DIR
from data.national_bridge_inventory.dataparser import Nbi
from ml_models.gemini import gemini

print("Getting report images...")
doc_dir = os.path.join(ROOT_DIR, "data/public_inspection_reports/data/2023-09-27_00022")
images_dir = os.path.join(doc_dir, "images")

with open(images_dir + "/metadata.json", "r") as f:
    images_metadata = json.loads(f.read())

with open(ROOT_DIR + "/data/public_inspection_reports/template.txt") as f:
    template_md = f.read()

image_data = []
for image_name, data in images_metadata.items():
    image = Image.open(images_dir + "/" + image_name)
    image_data.append(image)

    data_str = ""

    if data["image_type"] == "concrete":
        data_str += image_name + "\n" + data["damages"] + "\n\n"
    else:
        data_str += image_name + "\n" + data["caption"] + "\n\n"

    image_data.append(data_str)

nbi = Nbi()
bridge_data = str(nbi.get_bridge_by_structure_number("200030002208022"))
pp.pprint(bridge_data)
print("Generating markdown inspection report...")
instructions = "Generate a new inspection report following the template as a markdown document a civil engineer would need to create given the following image names, their descriptions, and NBI data about the bridge.\n\n"
response = gemini.query(
    query=[instructions]
    + image_data
    + ["Example:\n", template_md, "\n\nNBI Data:\n", bridge_data]
)

f = open(doc_dir + "/generated_inspection_report.md", "w+")
f.write(response)
f.close()
