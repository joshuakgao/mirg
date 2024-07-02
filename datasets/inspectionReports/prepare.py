import pymupdf
import re
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))) # for importing utils
from config import DATASETS_DIR
import markdownify 


RAWDATA_DIR = os.path.join(DATASETS_DIR, "inspectionReports/data")

for file_name in os.listdir(RAWDATA_DIR):
    # ignore non-pdf files
    if not file_name.endswith('.pdf'):
        continue

    # convert pdf to html
    file_path = os.path.join(RAWDATA_DIR, file_name)
    pdf = pymupdf.open(file_path) # open a document
    pdf_html = ""
    for i in range(len(pdf)):
        page1 = pdf.load_page(i)
        page1text = page1.get_text("xhtml")
        page1text = page1text.strip()
        page1text = page1text.strip('\n')
        page1text= re.sub('\s+', ' ', page1text)
        pdf_html += page1text

    # conver html to md
    md = markdownify.markdownify(pdf_html, heading_style="ATX") 

    # save md file
    OUTPUT_PATH = os.path.splitext(file_path)[0]
    f = open(OUTPUT_PATH + '.md', "w")
    f.write(md)

