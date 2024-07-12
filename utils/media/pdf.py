import os
import re

import markdownify
import pymupdf

from utils.media.md import convert_md_base64_images_to_filepath_images


def convert_pdf_to_md(file_path):
    print(f"Converting to markdown: {os.path.basename(file_path)}")

    # convert pdf to html
    pdf = pymupdf.open(file_path)  # open a document
    pdf_html = ""
    for i in range(len(pdf)):
        page1 = pdf.load_page(i)
        page1text = page1.get_text("xhtml")
        page1text = page1text.strip()
        page1text = page1text.strip("\n")
        page1text = re.sub("\s+", " ", page1text)
        pdf_html += page1text

    # conver html to md
    md = markdownify.markdownify(pdf_html, heading_style="ATX")

    # create dir with pdf name
    file_name = os.path.basename(file_path)  # inspection_report.pdf
    file_name_without_extension = os.path.splitext(file_name)[0]  # inspection_report
    parent_dir = os.path.dirname(file_path)  # ../
    report_dir = os.path.join(
        parent_dir, file_name_without_extension
    )  # inspection_report/
    os.makedirs(report_dir, exist_ok=True)
    images_dir = report_dir + "/images"  # inspection_report/images/

    # convert all base64 images in md to it's own file
    new_md_content = convert_md_base64_images_to_filepath_images(md, images_dir)

    # write md to doc dir
    md_file_path = report_dir + "/report.md"
    f = open(md_file_path, "w")
    f.write(new_md_content)

    return report_dir
