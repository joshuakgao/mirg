import os
import sys
import json
from PIL import Image
import psutil
import ast


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

        if report_dir != "input":
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
            md_report_dir = convert_pdf_to_md(file_path, paginate=True)

            # extract the captions of images links
            images_metadata = {}
            for i, md_name in enumerate(os.listdir(md_report_dir)):
                if not md_name.endswith(".md"):
                    continue

                print(f"Extracting cpations of images links {md_name}")
                md_path = os.path.join(md_report_dir, md_name)
                f = open(md_path, "r")
                md_page = f.read()

                dict_str = (
                    gemini.query(
                        query=[
                            md_page,
                            "Extract all unique image links and their captions from the markdown file above. Only output as python dict:",
                        ]
                    )
                    .strip("```")
                    .replace("python", "")
                )

                parsed_dict = ast.literal_eval(dict_str)
                images_metadata = {**images_metadata, **parsed_dict}
                f.close()

            # remove all "images/" from image links, move report caption to dict
            images_metadata = {
                key.replace("![Image](", "")
                .replace(")", "")
                .replace("images/", ""): value
                for key, value in images_metadata.items()
            }
            images_metadata = {
                key: {"report_caption": value} for key, value in images_metadata.items()
            }
            print(images_metadata)

            image_dir = os.path.join(md_report_dir, "images")
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

                # add dict to image key if it doesn't exist
                if image_filename not in images_metadata:
                    images_metadata[image_filename] = {}

                # add result to metadata
                images_metadata[image_filename]["image_type"] = classification

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
    # pdf_to_image_prepare()
    full_prepare()
