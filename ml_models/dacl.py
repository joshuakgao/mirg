import io
import sys
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation
from typing import List
import __main__

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # for importing paths
from paths import ROOT_DIR
from utils.device_selector import device_selector


class SegModel(nn.Module):
    def __init__(self, segformer):
        super(SegModel, self).__init__()
        self.segformer = segformer
        self.upsample = nn.Upsample(scale_factor=4, mode="nearest")

    def forward(self, x):
        return self.upsample(self.segformer(x).logits)


__main__.SegModel = SegModel  # for importing into another script


class Dacl:
    def __init__(self, device="auto"):
        self.classes, self.label2id, self.id2label = self._define_classes()
        self.transforms = self._define_transforms()
        self.device = device_selector(device, label="Dacl")
        model_path = os.path.join(ROOT_DIR, "ml_models/model_weights/dacl.pth")
        self.model = self._load_model(model_path)

    def _define_classes(self):
        classes = [
            "Crack",
            "ACrack",
            "Wetspot",
            "Efflorescence",
            "Rust",
            "Rockpocket",
            "Hollowareas",
            "Cavity",
            "Spalling",
            "Graffiti",
            "Weathering",
            "Restformwork",
            "ExposedRebars",
            "Bearing",
            "EJoint",
            "Drainage",
            "PEquipment",
            "JTape",
            "WConccor",
        ]
        nclasses = len(classes)
        label2id = dict(zip(classes, range(nclasses)))
        id2label = dict(zip(range(nclasses), classes))

        return classes, label2id, id2label

    def _define_transforms(self):
        """
        Used to preprocess images
        """
        self.to_tensor = transforms.ToTensor()
        self.to_array = transforms.ToPILImage()
        self.resize = transforms.Resize((512, 512))
        self.resize_small = transforms.Resize((369, 369))
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def _load_model(self, model_path):
        print(f"Loading Dacl weights from {model_path}")
        segformer = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b1", id2label=self.id2label, label2id=self.label2id
        )
        model = SegModel(segformer)
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        return model

    def _preprocess_image(self, img: Image.Image):
        img = self.to_tensor(img)
        img = self.resize(img)
        img = self.normalize(img)
        return img

    def _resize_image(self, img: Image.Image) -> Image.Image:
        """
        Used to resize the original background image to the model output size
        """
        img = self.to_tensor(img)
        img = self.resize_small(img)
        img = self.to_array(img)
        img = img.convert("RGBA")
        return img

    def _create_composite_image(self, foreground, background, alpha_factor):
        """
        Combine the foreground (mask_all) and background (original image) to create one image
        """
        foreground = np.array(foreground)
        background = np.array(background)

        background = Image.fromarray(background)
        foreground = Image.fromarray(foreground)
        new_alpha_factor = int(255 * alpha_factor)
        foreground.putalpha(new_alpha_factor)
        background.paste(foreground, (0, 0), foreground)

        return background

    def _get_mask_area(self, mask: List[List[float]]):
        return sum(sum(row) for row in mask)

    def inference(self, image: Image.Image, confidence=0.7, alpha_factor=0.6):
        # preprocess images
        background = self._resize_image(image)
        image = self._preprocess_image(image)

        # we need a batch, hence we introduce an extra dimenation at position 0 (unsqueeze)
        mask = self.model(image.unsqueeze(0).to(self.device))
        mask = mask[0]

        # get probability values (logits to probs)
        mask_probs = torch.sigmoid(mask).to("cpu")
        mask_probs = mask_probs.detach().numpy()  # (1, 512, 512)

        # make binary mask
        mask_preds = mask_probs > confidence

        # all combined
        mask_all = mask_preds.sum(axis=0)
        mask_all = np.expand_dims(mask_all, axis=0)  # (1, 512, 512)

        # Concat all combined with normal preds
        mask_preds = np.concatenate((mask_all, mask_preds), axis=0)  # (20, 512, 512)
        labs = ["ALL"] + self.classes

        # get mask areas
        mask_areas = [self._get_mask_area(pred) for pred in mask_preds]

        fig, axes = plt.subplots(5, 4, figsize=(10, 10))

        # save all mask_preds in all_mask
        all_masks = []

        for i, ax in enumerate(axes.flat):
            label = labs[i]

            all_masks.append(mask_preds[i])

            ax.imshow(mask_preds[i])
            ax.set_title(label)

        plt.tight_layout()
        plt.close()

        # plt to PIL
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png")
        im = Image.open(img_buf)

        # Saved all masks combined with unvisible xaxis und yaxis and without a white
        # background.
        all_images: List[Image.Image] = []
        for i in range(len(all_masks)):
            plt.figure()
            fig = plt.imshow(all_masks[i], interpolation="none", cmap="Reds")
            plt.axis("off")
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            img_buf = io.BytesIO()
            plt.savefig(img_buf, bbox_inches="tight", pad_inches=0, format="png")
            foreground = Image.open(img_buf)
            all_images.append(
                self._create_composite_image(foreground, background, alpha_factor)
            )
            foreground.close()
            plt.close()
        im.close()

        # create image with all masks overlayed on original image
        composite = self._create_composite_image(
            all_images[0], background, alpha_factor
        )

        return im, all_images, background, composite, mask_probs, mask_areas

    def assess_damage(
        self,
        image: Image.Image,
        confidence=0.5,  # min probability to be included in mask
        min_mask_size=0,  # require number of pixels in mask
    ):
        _, all_images, _, _, mask_probs, mask_areas = self.inference(
            image, confidence=confidence
        )

        # we only get a certain range of all_images because the other categories aren't considered damages
        all_images = all_images[1:12]
        mask_areas = mask_areas[1:12]
        classes = self.classes[0:11]

        # ignore the damage categories that are empty
        damaged_images_list = [
            (img, mask, label)
            for img, area, mask, label in zip(
                all_images, mask_areas, mask_probs, classes
            )
            if area > min_mask_size
        ]
        return damaged_images_list


if __name__ == "__main__":
    dacl = Dacl()
    img = Image.open(os.path.join(ROOT_DIR, "assets/bridge_damage_2.png"))
    img.show()
    print(dacl.assess_damage(img))
