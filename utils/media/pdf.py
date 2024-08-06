import os
import sys
import re
import markdownify
import pymupdf
from PIL import Image

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)  # for importing paths
from utils.media.md import convert_md_base64_images_to_filepath_images
from utils.media.image import find_duplicate_images
from paths import ROOT_DIR
from utils.logger import logger
from ml_models.gemini import gemini


def replace_image_paths(md: str, old_image_path: str, new_image_path: str):
    new_md = md.replace(old_image_path, new_image_path)
    return new_md


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

    for i, page_md in enumerate(paginated_pdf_md):
        # convert all base64 images in md to a file path image, since base64 images not supported in md
        new_md_content = convert_md_base64_images_to_filepath_images(
            page_md, images_dir
        )
        paginated_pdf_md[i] = new_md_content

    # replace duplicate images
    duplicate_groups = find_duplicate_images(images_dir)
    for i, page_md in enumerate(paginated_pdf_md):
        for replacement_image, to_be_removed_images in duplicate_groups.items():
            replacement_image_relative_path = os.path.basename(replacement_image)
            for to_be_removed_image in to_be_removed_images:
                # delete duplicate image
                logger.log(f"deleting duplicate: {to_be_removed_image}")
                if os.path.exists(to_be_removed_image):
                    os.remove(to_be_removed_image)

                to_be_removed_image_relative_path = os.path.basename(
                    to_be_removed_image
                )
                page_md = replace_image_paths(
                    page_md,
                    to_be_removed_image_relative_path,
                    replacement_image_relative_path,
                )

        # write md to doc dir
        md_page_file_path = report_dir + f"/page{i+1}.md"
        with open(md_page_file_path, "w") as f:
            f.write(page_md)

    # rename images from hash to descriptive name
    for image_name in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_name)
        image = Image.open(image_path)
        new_image_filename = gemini.caption_image(
            image,
            context="Provide a short caption describing this image in snake case.",
        )
        new_image_path = os.path.join(images_dir, new_image_filename + ".png")
        os.rename(image_path, new_image_path)

    return report_dir


if __name__ == "__main__":
    pdf_file_path = os.path.join(
        ROOT_DIR, "data/inspection_reports/data/12-085-E000-12-004_RTInsp_2023-12.pdf"
    )
    convert_pdf_to_md(pdf_file_path, paginate=True)
