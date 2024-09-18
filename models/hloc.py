import os
import sys
from pathlib import Path
import numpy as np
import tqdm
from typing import Union
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from packages.hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
)
from packages.hloc.visualization import plot_images, read_image
from packages.hloc.utils import viz_3d
from utils.singleton_decorator import singleton
from utils.media.image import load_image


@singleton
class Hloc:
    def __init__(
        self,
        outputs=Path("outputs/hloc_demo"),
    ):
        self.outputs = Path(outputs)
        self.sfm_pairs = outputs / "pairs-sfm.txt"
        self.loc_pairs = outputs / "pairs-loc.txt"
        self.sfm_dir = outputs / "sfm"
        self.features = outputs / "features.h5"
        self.matches = outputs / "matches.h5"
        self.feature_conf = extract_features.confs["disk"]
        self.matcher_conf = match_features.confs["disk+lightglue"]

    def count_inliers(self, image: Union[Image.Image, str]):
        image = load_image(image)


if __name__ == "__main__":
    hloc = Hloc()
    hloc.match()
