import os
import sys
import json
from PIL import Image
import psutil

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)  # for importing utils
from ml_models.clip import clip
from ml_models.dacl import dacl
from ml_models.llava import llava
from paths import ROOT_DIR
from utils.media.pdf import convert_pdf_to_md
from utils.logger import logger
from utils.media.file_validation import check_files_and_directories

process = psutil.Process()


def check_if_downloads_exist():
    required_file_paths = [os.path.join(ROOT_DIR, "ml_models/model_weights/dacl.pth")]
    non_empty_dirs = [
        os.path.join(ROOT_DIR, "data/inspection_reports/data"),
    ]
    check_files_and_directories(required_file_paths, non_empty_dirs)


def remove_image_duplicates(): ...


if __name__ == "__main__":
    check_if_downloads_exist()

    reports_dir = os.path.join(ROOT_DIR, "data/inspection_reports/data")
    for i, file_name in enumerate(os.listdir(reports_dir)):
        # ignore non-pdf files
        if not file_name.endswith(".pdf"):
            continue

        logger.log(
            f"({i+1} of {len(os.listdir(reports_dir))}) Preparing inspection report: {file_name}"
        )

        # convert pdf inpsection report to md and save images to images/ dir
        file_path = os.path.join(reports_dir, file_name)
        report_dir = convert_pdf_to_md(file_path, paginate=True)

        image_dir = os.path.join(report_dir, "images")
        images_metadata = {}
        for i, image_filename in enumerate(os.listdir(image_dir)):
            print("Memory used:", process.memory_info().rss / 1000000000)

            image_filepath = os.path.join(
                image_dir, image_filename
            )  # report/images/image_name.jpeg

            logger.log(f"Checking if is a structure: {image_filepath}")

            image = Image.open(image_filepath)

            if image.mode != "RGB":
                image = image.convert("RGB")

            classes = [
                "concrete",
                "grass field",
                "map",
                "diagram",
                "logo",
                "blank",
            ]
            probs = clip.image_classification(image, classes)

            # Find the class with the highest probability
            classification = max(zip(probs, classes))[1]
            logger.log(classification)

            # delete images that are nothing
            if classification == "blank":
                os.remove(image_filepath)
                continue

            # add result to metadata
            images_metadata[image_filename] = {"image_type": classification}

            # add caption to image with llava
            caption = llava.caption_image(image_filepath)
            images_metadata[image_filename]["caption"] = caption

            # execute the following code only if the classification is "city"
            if classification != "concrete":
                continue

            # get list of damage segmentation images by category
            logger.log(f"Saving damage images:{image_filepath}")
            damages = dacl.assess_damage(
                image, confidence=0.4, min_mask_size=13107  # 5% of a 512x512 image
            )  # (damage_image, mask, category)

            # create damages dict
            images_metadata[image_filename]["damages"] = {}

            # add damage categories to image metadata
            images_metadata[image_filename]["damages"]["categories"] = [
                damage_category for _, _, damage_category in damages
            ]

            # add damage masks to image metadata
            for _, mask, damage_category in damages:
                images_metadata[image_filename]["damages"][
                    damage_category
                ] = mask.tolist()

            # save damage images to new dir under images dir
            for damage_image, _, damage_category in damages:
                # create dir for damage assetsment images
                image_basename = os.path.basename(image_filename)
                image_name_without_extension = os.path.splitext(image_basename)[
                    0
                ]  # inspection_report
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

            # describe damages in image with llava
            caption = llava.caption_image(
                image_filepath,
                context="Describe in detail any damages to the structure.",
            )
            images_metadata[image_filename]["damages"]["caption"] = caption

        # write images metadata to file
        metadata_filepath = os.path.join(image_dir, "metadata.json")
        with open(metadata_filepath, "w") as f:
            json.dump(images_metadata, f, indent=4)
