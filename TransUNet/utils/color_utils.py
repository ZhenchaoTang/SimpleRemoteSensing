import numpy as np
import cv2
from PIL import Image, ImageFile

class ImageOperation:

    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")

    @staticmethod
    def saveImage(image, path):
        image.save(path)

# 给标签图上色

def color_annotation(img, output_path):
    '''
    给class图上色
    '''

    color = np.ones([img.shape[0], img.shape[1], 3])

    color[img == 0] = [255, 255, 255] # Impervious surfaces
    color[img == 1] = [0, 0, 255]     # Building
    color[img == 2] = [0, 255, 255]   # Low vegetation
    color[img == 3] = [0, 255, 0]     # Tree
    color[img == 4] = [255, 255, 0]   # Car
    color[img == 5] = [255, 0, 0]     # Clutter

    ImageOperation.saveImage(Image.fromarray(np.uint8(color)), output_path)