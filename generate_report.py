import os
import sys
import json

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # for importing paths
from paths import ROOT_DIR

# from ml_models.gemma2 import gemma2
from ml_models.llama3 import llama3


print("Getting report images...")
doc_dir = os.path.join(
    ROOT_DIR, "data/inspection_reports/data/12-085-C002-08-052_RTInsp_2023-10"
)
images_dir = os.path.join(doc_dir, "images")

with open(images_dir + "/metadata.json", "r") as f:
    images_metadata = json.loads(f.read())


image_data = []
for image_name, data in images_metadata.items():
    data_str = ""
    data_str = "Image Name: " + image_name + "\n" + "Image Caption: " + data["caption"]

    if data["image_type"] == "concrete":
        data_str += " " + data["damages"]["caption"]

    image_data.append(data_str)


print(len(image_data))
print(image_data)
print("Generating markdown inspection report...")
instructions = "Generate a new inspection report as a markdown document a civil engineer would need to create given the following image names and their description. Add the image names into the markdown file."
response = llama3.query(
    messages=[
        # {
        #     "role": "system",
        #     "content": instructions,
        # },
        {
            "role": "user",
            "content": instructions + "\n\n".join(image_data),
        },
    ]
)

f = open("generated_inspection_report.md", "w+")
f.write(response)
f.close()
