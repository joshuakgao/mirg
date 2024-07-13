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

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # for importing paths
from paths import ROOT_DIR

###################
# Setup label names
target_list = [
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
target_list_all = ["All"] + target_list
classes, nclasses = target_list, len(target_list)
label2id = dict(zip(classes, range(nclasses)))
id2label = dict(zip(range(nclasses), classes))

############
# Load model
device = torch.device("cpu")
segformer = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b1", id2label=id2label, label2id=label2id
)

# SegModel


class SegModel(nn.Module):
    def __init__(self, segformer):
        super(SegModel, self).__init__()
        self.segformer = segformer
        self.upsample = nn.Upsample(scale_factor=4, mode="nearest")

    def forward(self, x):
        return self.upsample(self.segformer(x).logits)


model = SegModel(segformer)
path = ROOT_DIR + "/ml_models/model_weights/dacl.pth"
print(f"Load Segformer weights from {path}")
# model = model.load_state_dict(torch.load(path, map_location=device))
model = torch.load(path)
model.eval()

##################
# Image preprocess
##################

to_tensor = transforms.ToTensor()
to_array = transforms.ToPILImage()
resize = transforms.Resize((512, 512))
resize_small = transforms.Resize((369, 369))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def process_pil(img):
    img = to_tensor(img)
    img = resize(img)
    img = normalize(img)
    return img


# the background of the image


def resize_pil(img):
    img = to_tensor(img)
    img = resize_small(img)
    img = to_array(img)
    img = img.convert("RGBA")
    return img


# combine the foreground (mask_all) and background (original image) to create one image


def transparent(foreground, background, alpha_factor):

    foreground = np.array(foreground)
    background = np.array(background)

    background = Image.fromarray(background)
    foreground = Image.fromarray(foreground)
    new_alpha_factor = int(255 * alpha_factor)
    foreground.putalpha(new_alpha_factor)
    background.paste(foreground, (0, 0), foreground)

    return background


def show_img(all_imgs, dropdown, bg, alpha_factor):
    idx = target_list_all.index(dropdown)
    fg = all_imgs[idx]["name"]

    foreground = Image.open(fg)
    background = np.array(bg)

    background = Image.fromarray(bg)
    new_alpha_factor = int(255 * alpha_factor)
    foreground.putalpha(new_alpha_factor)
    background.paste(foreground, (0, 0), foreground)

    return background


###########
# Inference


def inference(img: Image.Image, alpha_factor=0.4):
    background = resize_pil(img)

    img = process_pil(img)

    # we need a batch, hence we introduce an extra dimenation at position 0 (unsqueeze)
    mask = model(img.unsqueeze(0))
    mask = mask[0]

    # Get probability values (logits to probs)
    mask_probs = torch.sigmoid(mask)
    mask_probs = mask_probs.detach().numpy()
    mask_probs.shape

    # Make binary mask
    THRESHOLD = 0.5
    mask_preds = mask_probs > THRESHOLD

    # All combined
    mask_all = mask_preds.sum(axis=0)
    mask_all = np.expand_dims(mask_all, axis=0)
    mask_all.shape

    # Concat all combined with normal preds
    mask_preds = np.concatenate((mask_all, mask_preds), axis=0)
    labs = ["ALL"] + target_list

    fig, axes = plt.subplots(5, 4, figsize=(10, 10))

    # save all mask_preds in all_mask
    all_masks = []

    for i, ax in enumerate(axes.flat):
        label = labs[i]

        all_masks.append(mask_preds[i])

        ax.imshow(mask_preds[i])
        ax.set_title(label)

    plt.tight_layout()

    # plt to PIL
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    im = Image.open(img_buf)

    # Saved all masks combined with unvisible xaxis und yaxis and without a white
    # background.
    all_images = []
    for i in range(len(all_masks)):
        plt.figure()
        fig = plt.imshow(all_masks[i])
        plt.axis("off")
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        img_buf = io.BytesIO()
        plt.savefig(img_buf, bbox_inches="tight", pad_inches=0, format="png")
        all_images.append(Image.open(img_buf))

    # create image with all masks overlayed on original image
    composite = transparent(all_images[0], background, alpha_factor)

    return im, all_images, background, composite


if __name__ == "__main__":
    img = Image.open(ROOT_DIR + "/assets/bridge_damage.jpg")
    im, all_images, background, composite = inference(img, alpha_factor=0.4)
    background.show()
    composite.show()
    im.show()
