import os
import sys
from pathlib import Path
import json
from typing import Union
from PIL import Image
import shutil


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from packages.hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
)
from utils.singleton_decorator import singleton
from utils.media.image import load_image
from utils.logger import logger


@singleton
class Hloc:
    def __init__(
        self,
        outputs=Path("outputs/hloc"),
    ):
        self.outputs = Path(outputs)
        self.sfm_pairs = outputs / "pairs-sfm.txt"
        self.loc_pairs = outputs / "pairs-loc.txt"
        self.sfm_dir = outputs / "sfm"
        self.features = outputs / "features.h5"
        self.matches = outputs / "matches.h5"
        self.cropped_images = outputs / "images"

        self.feature_conf = extract_features.confs["disk"]
        self.matcher_conf = match_features.confs["disk+lightglue"]

    def is_same_location(
        self,
        image: Union[Image.Image, str],
        image_dir: str,
        min_inliers: int = 100,
    ):
        if os.path.isdir(self.outputs):
            shutil.rmtree(self.outputs)

        os.makedirs(self.cropped_images)

        # preprare images for sfm build
        images_dir = self.cropped_images
        self._generate_cropped_images(image)
        references = [
            p.relative_to(images_dir).as_posix() for p in (images_dir).iterdir()
        ]

        # get features from generated cropped images
        extract_features.main(
            self.feature_conf,
            images_dir,
            image_list=references,
            feature_path=self.features,
        )
        pairs_from_exhaustive.main(self.sfm_pairs, image_list=references)
        match_features.main(
            self.matcher_conf,
            self.sfm_pairs,
            features=self.features,
            matches=self.matches,
        )

        # create sfm model from image features
        model = reconstruction.main(
            self.sfm_dir,
            images_dir,
            self.sfm_pairs,
            self.features,
            self.matches,
            image_list=references,
        )  # sfm model

        # compare sfm model to images in image_dir
        matches = {}  # store matching images
        for name in os.listdir(image_dir):
            try:
                if not name.endswith(".png"):
                    continue
                query = os.path.join(image_dir, name)
                extract_features.main(
                    self.feature_conf,
                    images_dir,
                    image_list=[query],
                    feature_path=self.features,
                    overwrite=True,
                )
                pairs_from_exhaustive.main(
                    self.loc_pairs, image_list=[query], ref_list=references
                )
                match_features.main(
                    self.matcher_conf,
                    self.loc_pairs,
                    features=self.features,
                    matches=self.matches,
                    overwrite=True,
                )
                import pycolmap
                from packages.hloc.localize_sfm import QueryLocalizer, pose_from_cluster

                camera = pycolmap.infer_camera_from_image(images_dir / query)
                ref_ids = [model.find_image_with_name(r).image_id for r in references]
                conf = {
                    "estimation": {"ransac": {"max_error": 12}},
                    "refinement": {
                        "refine_focal_length": True,
                        "refine_extra_params": True,
                    },
                }
                localizer = QueryLocalizer(model, conf)
                ret, log = pose_from_cluster(
                    localizer, query, camera, ref_ids, self.features, self.matches
                )
                if ret:
                    logger.log(
                        f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.'
                    )
                    if len(ret["inliers"]) > min_inliers:
                        matches[query] = len(ret["inliers"])

                visualization.visualize_loc_from_log(images_dir, query, log, model)
            except:
                continue

        return matches

    def _generate_cropped_images(self, image: Union[Image.Image, str]):
        image = load_image(image)
        image.save(f"{self.cropped_images}/original.png")

        def zoom_image(image, zoom_factor=1.4, pattern_index=0, total_patterns=10):
            # Get the original size
            original_width, original_height = image.size

            # Calculate the size of the zoomed area
            crop_width = int(original_width / zoom_factor)
            crop_height = int(original_height / zoom_factor)

            # Calculate the grid size (total_patterns should define the grid)
            rows = cols = int(total_patterns**0.5)

            # Determine the position of the crop based on the pattern index
            row = pattern_index // cols
            col = pattern_index % cols

            # Calculate the left, top, right, bottom coordinates based on grid position
            left = int(col * (original_width - crop_width) / max(1, cols - 1))
            top = int(row * (original_height - crop_height) / max(1, rows - 1))
            right = left + crop_width
            bottom = top + crop_height

            # Crop the image to the specific area
            cropped_image = image.crop((left, top, right, bottom))

            # Resize the cropped image back to the original size
            zoomed_image = cropped_image.resize(
                (original_width, original_height), Image.LANCZOS
            )
            return zoomed_image

        # Apply patterned zoom to the image multiple times
        total_patterns = 9  # Define the number of patterns
        for i in range(total_patterns):
            zoomed_image = zoom_image(
                image, pattern_index=i, total_patterns=total_patterns
            )
            # Save the zoomed image
            zoomed_image.save(f"{self.cropped_images}/cropped_{i}.png")


if __name__ == "__main__":
    hloc = Hloc()
    with open(
        "/home/jkgao/Documents/GitHub/mirg/data/public_inspection_reports/data/200030002008022/viewreport.ashx-3/images/metadata.json",
        "r",
    ) as f:
        metadata = json.load(f)

    image_path = "/home/jkgao/Documents/GitHub/mirg/data/public_inspection_reports/data/input/VT15-00020-CAMBRIDGE/images/concrete_bridge_underpass_waterway.png"

    matches3 = hloc.is_same_location(
        image=image_path,
        image_dir="/home/jkgao/Documents/GitHub/mirg/data/public_inspection_reports/data/200030002008022/viewreport.ashx-3/images",
        min_inliers=500,
    )

    matches4 = hloc.is_same_location(
        image=image_path,
        image_dir="/home/jkgao/Documents/GitHub/mirg/data/public_inspection_reports/data/200030002008022/viewreport.ashx-5/images",
        min_inliers=500,
    )

    print("Report 3")
    for k, v in matches3.items():
        try:
            print(f'{v}: {metadata[os.path.basename(k)]["report_caption"]}')
        except:
            continue

    print("Report 5")
    for k, v in matches4.items():
        try:
            print(f'{v}: {metadata[os.path.basename(k)]["report_caption"]}')
        except:
            continue
