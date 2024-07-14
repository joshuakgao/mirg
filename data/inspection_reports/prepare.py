import os
import sys
import json
from PIL import Image

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)  # for importing utils
from ml_models.clip import Clip
from ml_models.dacl import *
from paths import ROOT_DIR
from utils.media.pdf import convert_pdf_to_md
from utils.logger import logger

if __name__ == "__main__":
    clip = Clip(model_id="B-32")
    dacl = Dacl()

    DATA_DIR = os.path.join(ROOT_DIR, "data/inspection_reports/data")
    for i, file_name in enumerate(os.listdir(DATA_DIR)):
        # ignore non-pdf files
        if not file_name.endswith(".pdf"):
            continue

        logger.log(
            f"({i+1} of {len(os.listdir(DATA_DIR))}) Preparing inspection report: {file_name}"
        )

        file_path = os.path.join(DATA_DIR, file_name)
        report_dir = convert_pdf_to_md(file_path, paginate=True)

        image_dir = os.path.join(report_dir, "images")
        images_metadata = {}
        for i, image_filename in enumerate(os.listdir(image_dir)):
            image_filepath = os.path.join(
                image_dir, image_filename
            )  # report/images/image_name.jpeg

            logger.log(
                f"({i+1} of {len(os.listdir(image_dir))}) Checking if is a structure: {image_filepath}"
            )

            image = Image.open(image_filepath)

            if image.mode != "RGB":
                image = image.convert("RGB")

            classes = ["city", "map", "diagram", "logo"]
            probs = clip.image_classification(image, classes)

            # Find the class with the highest probability
            classification = max(zip(probs, classes))[1]
            print(classification)

            # add result to metadata
            images_metadata[image_filename] = {"image_type": classification}

            # execute the following code only if the classification is "city"
            if classification != "city":
                continue

            # get list of damage segmentation images by category
            damages = dacl.assess_damage(image)  # (damage_image, category)

            # add damage categories to image metadata
            images_metadata[image_filename]["damages"] = [
                damage_category for _, damage_category in damages
            ]

            # save damage images to new dir under images dir
            for damage_image, damage_category in damages:
                # create dir for damage assetsment images
                image_basename = os.path.basename(image_filename)
                image_name_without_extension = os.path.splitext(image_basename)[
                    0
                ]  # inspection_report
                print(image_name_without_extension)
                damage_images_dir = os.path.join(
                    image_dir, image_name_without_extension
                )
                if not os.path.exists(damage_images_dir):
                    os.makedirs(damage_images_dir)

                # save damage image
                damage_image_filepath = os.path.join(
                    image_dir, f"{damage_images_dir}/{damage_category}.png"
                )
                damage_image.save(damage_image_filepath)

        # write images metadata to file
        metadata_filepath = os.path.join(image_dir, "metadata.json")
        with open(metadata_filepath, "w") as f:
            json.dump(images_metadata, f, indent=4)
