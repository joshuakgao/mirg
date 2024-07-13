import os
import sys
import json
from PIL import Image

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)  # for importing utils
from ml_models.clip import Clip
from paths import DATASETS_DIR
from utils.media.pdf import convert_pdf_to_md
from utils.logger import logger

if __name__ == "__main__":
    clip = Clip(model_id="G-14")

    DATA_DIR = os.path.join(DATASETS_DIR, "inspection_reports/data")
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
            classes = ["city", "map", "diagram", "logo"]
            probs = clip.image_classification(image, classes)

            # Find the class with the highest probability
            classification = max(zip(probs, classes))[1]
            print(classification)

            # add result to metadata
            images_metadata[image_filename] = {"content_type": classification}

        # write images metadata to file
        metadata_filepath = os.path.join(image_dir, "metadata.json")
        with open(metadata_filepath, "w") as f:
            json.dump(images_metadata, f, indent=4)
