import torch
import numpy as np
import cv2
import albumentations as albu
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from cloths_segmentation.pre_trained_models import create_model

def infer(image):
    try:
        model = create_model("Unet_2020-10-30")
    except FileNotFoundError:
        print("model is empty!")
    model.eval()
    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    try:
        padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
    except FileNotFoundError:
        print("Image is empty!")
    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)
    with torch.no_grad():
        prediction = model(x)[0][0]
    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)
    return mask
