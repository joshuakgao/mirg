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
from models.gemini import Gemini

gemini = Gemini()

structure_number = "200030002008022"
print("Getting report images...")
doc_dir = os.path.join(
    ROOT_DIR, "data/public_inspection_reports/data/input/VT15-00020-CAMBRIDGE"
)
images_dir = os.path.join(doc_dir, "images")

with open(images_dir + "/metadata.json", "r") as f:
    images_metadata = json.loads(f.read())

with open(ROOT_DIR + "/data/public_inspection_reports/template.txt") as f:
    template_md = f.read()

image_data = ["New Images:\n"]
for image_name, data in images_metadata.items():
    image = Image.open(images_dir + "/" + image_name)
    image_data.append(image)

    data_str = ""

    if data["image_type"] == "concrete":
        data_str += image_name + "\n" + data["damages"] + "\n\n"
    else:
        data_str += image_name + "\n" + data["caption"] + "\n\n"

    image_data.append(data_str)

past_report_data = ["Past Reports:"]
past_reports_dir = ROOT_DIR + f"/data/public_inspection_reports/data/{structure_number}"
for report_name in os.listdir(past_reports_dir):
    report_path = os.path.join(past_reports_dir, report_name)
    if not os.path.isdir(report_path):
        continue

    for report_file in os.listdir(report_path):
        if not report_file.endswith(".md"):
            continue

        with open(os.path.join(report_path, report_file)) as f:
            md_page = f.read()

        past_report_data.append(md_page)


nbi = Nbi()
bridge_data = str(nbi.get_bridge_by_structure_number(structure_number))
pp.pprint(bridge_data)
print("Generating markdown inspection report...")
instructions = "Generate a new inspection report following the markdown template as a markdown document a civil engineer would need to create given the following image names, their descriptions, and NBI data about the bridge. Use information from past reports to get an understanding on what to focus on. Embed all images with this format: ![Image](images/image_path.png)\n\n"
response = gemini.query(
    query=[instructions]
    + image_data
    + ["Example:\n", template_md, "\n\nNBI Data:\n", bridge_data]
    + ["Past reports:"]
    + past_report_data
)

f = open(doc_dir + "/generated_inspection_report.md", "w+")
f.write(response)
f.close()
