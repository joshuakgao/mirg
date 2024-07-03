import pymupdf
import re
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))) # for importing utils
from config import DATASETS_DIR
import markdownify 
import base64


def save_base64_image(base64_str, output_dir, img_format="jpeg"):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Decode the base64 string
    image_data = base64.b64decode(base64_str)
    
    # Generate a unique filename
    img_filename = f"image_{hash(base64_str)}.{img_format}"
    img_path = os.path.join(output_dir, img_filename)
    
    # Save the image to the file
    with open(img_path, "wb") as img_file:
        img_file.write(image_data)
    
    return img_filename


def convert_base64_images_in_md(md_content, output_dir):
    # Regular expression to find base64 encoded images
    base64_img_regex = re.compile(r'!\[.*?\]\(data:image\/(.*?);base64, (.*?)\)')
    
    def replace_base64_with_path(match):
        img_format = match.group(1)
        base64_str = match.group(2)
        img_path = save_base64_image(base64_str, output_dir, img_format)
        return f"![Image](images/{img_path})"
    
    # Replace all base64 images with file paths
    updated_md_content = base64_img_regex.sub(replace_base64_with_path, md_content)
    
    return updated_md_content


if __name__ == "__main__":
    DATA_DIR = os.path.join(DATASETS_DIR, "inspectionReports/data")

    for file_name in os.listdir(DATA_DIR):
        # ignore non-pdf files
        if not file_name.endswith('.pdf'):
            continue

        # convert pdf to html
        file_path = os.path.join(DATA_DIR, file_name)
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

        # create dir with pdf name
        file_name_without_extension = os.path.splitext(file_name)[0]
        new_dir_path = os.path.join(DATA_DIR, file_name_without_extension)
        os.makedirs(new_dir_path, exist_ok=True)

        # convert all base64 images in md to it's own file
        new_md_content = convert_base64_images_in_md(md, new_dir_path + "/images")

        # write md to doc dir
        f = open(new_dir_path + '/doc.md', "w")
        f.write(new_md_content)