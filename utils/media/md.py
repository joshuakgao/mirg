import os
import re
from utils.media.image import save_base64_image
from PIL import Image


def convert_md_base64_images_to_filepath_images(
    md_content: str, output_dir: str, min_required_wh=(100, 100)
):
    # Regular expression to find base64 encoded images
    base64_img_regex = re.compile(r"!\[.*?\]\(data:image\/(.*?);base64, (.*?)\)")

    def replace_base64_with_path(match):
        base64_str = match.group(2)
        image_name = save_base64_image(base64_str, output_dir, image_format="png")
        image_path = os.path.join(output_dir, image_name)

        # delete image if it is less than required width and height
        with Image.open(image_path) as img:
            width, height = img.size
            if width < min_required_wh[0] or height < min_required_wh[1]:
                os.remove(image_path)
                return ""

        return f"![Image](images/{image_name})"

    # Replace all base64 images with file paths
    updated_md_content = base64_img_regex.sub(replace_base64_with_path, md_content)

    return updated_md_content
