import os
import sys
import re
import markdownify
import pymupdf

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)  # for importing paths
from utils.media.md import convert_md_base64_images_to_filepath_images
from paths import ROOT_DIR
from utils.logger import Logger


logger = Logger()


def convert_pdf_to_md(file_path: str, paginate=False):
    logger.log(f"Converting to markdown: {os.path.basename(file_path)}")

    # convert pdf to html
    pdf = pymupdf.open(file_path)  # open pdf
    paginated_pdf_html = []
    for i in range(len(pdf)):
        page = pdf.load_page(i)
        page_text = page.get_text("xhtml")
        page_text = page_text.strip()
        page_text = page_text.strip("\n")
        page_text = re.sub("\s+", " ", page_text)
        paginated_pdf_html.append(page_text)
    pdf.close()

    if not paginate:
        paginated_pdf_html = ["".join(paginated_pdf_html)]  # merge all pages into one

    # conver html to md
    paginated_pdf_md = []  # to hold md of every page of pdf
    for page in paginated_pdf_html:
        md = markdownify.markdownify(page)
        paginated_pdf_md.append(md)

    # create dir with pdf name in given pdf dir
    file_name = os.path.basename(file_path)  # inspection_report.pdf
    file_name_without_extension = os.path.splitext(file_name)[0]  # inspection_report
    parent_dir = os.path.dirname(file_path)  # ../
    report_dir = os.path.join(
        parent_dir, file_name_without_extension
    )  # ../inspection_report/
    os.makedirs(report_dir, exist_ok=True)
    images_dir = os.path.join(report_dir, "images")  # ../inspection_report/images/

    for i, page in enumerate(paginated_pdf_md):
        # convert all base64 images in md to a file path image, since base64 images not supported in md
        new_md_content = convert_md_base64_images_to_filepath_images(page, images_dir)

        # write md to doc dir
        md_page_file_path = report_dir + f"/page{i+1}.md"
        with open(md_page_file_path, "w") as f:
            f.write(new_md_content)

    return report_dir


if __name__ == "__main__":
    pdf_file_path = os.path.join(
        ROOT_DIR, "data/inspection_reports/data/12-085-E000-12-004_RTInsp_2023-12.pdf"
    )
    convert_pdf_to_md(pdf_file_path, paginate=True)
