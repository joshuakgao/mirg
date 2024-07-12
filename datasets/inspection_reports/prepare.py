import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)  # for importing utils
from paths import DATASETS_DIR
from utils.media.pdf import convert_pdf_to_md

if __name__ == "__main__":
    DATA_DIR = os.path.join(DATASETS_DIR, "inspection_reports/data")

    for file_name in os.listdir(DATA_DIR):
        # ignore non-pdf files
        if not file_name.endswith(".pdf"):
            continue

        file_path = os.path.join(DATA_DIR, file_name)
        report_dir = convert_pdf_to_md(file_path, paginate=True)
