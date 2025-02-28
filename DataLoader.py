import cv2
import numpy as np
from PIL import Image

def load_image(image_path):
    img = Image.open(image_path)
    return img

def load_image_canny(image_path):
    img = load_image(image_path)
    LTH = 100
    HTH = 210

    np_image = np.array(img)

    canny_image = cv2.Canny(np_image, LTH, HTH)

    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)

    return canny_image

