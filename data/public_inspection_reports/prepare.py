import os
import sys
import json
from PIL import Image
import psutil

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)  # for importing utils
from ml_models.clip import Clip
from ml_models.gemini import Gemini
from paths import ROOT_DIR
from utils.media.pdf import convert_pdf_to_md
from utils.logger import logger
from utils.media.file_validation import check_files_and_directories
from pdf2image import convert_from_path


process = psutil.Process()
clip = Clip()
gemini = Gemini()


def check_if_downloads_exist():
    required_file_paths = [os.path.join(ROOT_DIR, "ml_models/model_weights/dacl.pth")]
    non_empty_dirs = [
        os.path.join(ROOT_DIR, "data/public_inspection_reports/data"),
    ]
    check_files_and_directories(required_file_paths, non_empty_dirs)


def full_prepare():
    check_if_downloads_exist()

    reports_dir = os.path.join(ROOT_DIR, "data/public_inspection_reports/data")
    report_dirs = os.listdir(reports_dir)
    for report_dir in report_dirs:

        if report_dir == "input":
            continue

        report_dir = os.path.join(reports_dir, report_dir)
        if not os.path.isdir(report_dir):
            continue

        for i, file_name in enumerate(os.listdir(report_dir)):
            # ignore non-pdf files
            if not file_name.endswith(".pdf"):
                continue

            logger.log(
                f"({i+1} of {len(os.listdir(report_dir))}) Preparing inspection report: {file_name}"
            )

            # convert pdf inpsection report to md and save images to images/ dir
            file_path = os.path.join(report_dir, file_name)
            md_report_dir = convert_pdf_to_md(file_path, paginate=False)

            image_dir = os.path.join(md_report_dir, "images")
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
                classification = clip.image_classification(image, classes)

                logger.log(classification)

                # delete images that are nothing
                if classification == "blank":
                    os.remove(image_filepath)
                    continue

                # add result to metadata
                images_metadata[image_filename] = {"image_type": classification}

                # add short caption to image by parsing file name
                # we do this to avoid calling an llm twice for a short caption
                caption = image_filename.split("_")
                caption.pop()  # remove uuid and file extension
                caption = " ".join(caption)  # convert list of strings to string
                images_metadata[image_filename]["caption"] = caption

                # execute the following code only if the classification is "city"
                if classification != "concrete":
                    continue

                # describe damages in image with llava
                caption = gemini.caption_image(
                    image,
                    context="Describe in detail any damages to the structure.",
                )
                images_metadata[image_filename]["damages"] = caption

            # write images metadata to file
            metadata_filepath = os.path.join(image_dir, "metadata.json")
            with open(metadata_filepath, "w") as f:
                json.dump(images_metadata, f, indent=4)


def pdf_to_image_prepare():
    reports_dir = os.path.join(ROOT_DIR, "data/public_inspection_reports/data")
    report_dirs = os.listdir(reports_dir)
    for report_dir in report_dirs:

        # if report_dir == "input":
        #     continue

        report_dir = os.path.join(reports_dir, report_dir)
        if not os.path.isdir(report_dir):
            continue

        for i, file_name in enumerate(os.listdir(report_dir)):
            # ignore non-pdf files
            if not file_name.endswith(".pdf"):
                continue

            logger.log(
                f"({i+1} of {len(os.listdir(report_dir))}) Preparing inspection report: {file_name}"
            )

            # convert pdf inpsection report to md and save images to images/ dir
            file_path = os.path.join(report_dir, file_name)

            file_name = os.path.basename(file_path)  # inspection_report.pdf
            file_name_without_extension = os.path.splitext(file_name)[
                0
            ]  # inspection_report
            parent_dir = os.path.dirname(file_path)  # ../
            report_images_dir = os.path.join(
                parent_dir, file_name_without_extension
            )  # ../inspection_report/
            os.makedirs(report_images_dir, exist_ok=True)

            images = convert_from_path(file_path)

            # Save each page as an image
            for i, image in enumerate(images):
                image.save(os.path.join(report_images_dir, f"page_{i+1}.png"))


if __name__ == "__main__":
    pdf_to_image_prepare()
