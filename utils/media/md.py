import re
from utils.media.image import save_base64_image


def convert_md_base64_images_to_filepath_images(md_content, output_dir):
    # Regular expression to find base64 encoded images
    base64_img_regex = re.compile(r"!\[.*?\]\(data:image\/(.*?);base64, (.*?)\)")

    def replace_base64_with_path(match):
        img_format = match.group(1)
        base64_str = match.group(2)
        img_path = save_base64_image(base64_str, output_dir, img_format)
        return f"![Image](images/{img_path})"

    # Replace all base64 images with file paths
    updated_md_content = base64_img_regex.sub(replace_base64_with_path, md_content)

    return updated_md_content
