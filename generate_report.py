import os
import sys
import json
from PIL import Image

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # for importing paths
from paths import ROOT_DIR
from ml_models.llava import llava
from utils.media.image import convert_image_to_base64


print("Getting report images...")
images_dir = os.path.join(
    ROOT_DIR, "data/inspection_reports/data/14-246-AA03-51-003_FCReport_2023-06/images"
)

with open(images_dir + "/metadata.json", "r") as f:
    images_metadata = json.loads(f.read())


image_names = []
images = []
for image_name, data in images_metadata.items():
    if images_metadata[image_name]["image_type"] == "concrete":
        # open and resize image
        image_path = os.path.join(images_dir, image_name)
        image = Image.open(image_path)
        image = image.resize((512, 512))

        # store image and image name
        if not os.path.exists("./temp/"):
            os.makedirs("./temp/")
        image.save("./temp/" + image_name)

        images.append("./temp/" + image_name)
        image_names.append(image_name)

image_names = image_names[:2]
images = images[:2]
print(len(images))
print("Generating markdown inspection report...")
instructions = "Generate an inspection report as a markdown document a civil engineer would need to create given the following images. Provide some maintenance suggestions based on each image. Include the images in the report by embedding it with the image's name."
response = llava.query(
    messages=[
        {
            "role": "system",
            "content": instructions,
        },
        {
            "role": "user",
            "content": "\n".join(image_names),
            "images": images,
        },
    ]
)

f = open("generated_inspection_report.md", "w+")
f.write(response)
f.close()
