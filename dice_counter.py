import cv2 as cv
import skimage
import numpy as np
import random as rng
import copy
import matplotlib.pyplot as plt

def normalize_image(image):
    blur = skimage.filters.gaussian(image, sigma=2)
    x = round(np.std(image) + np.mean(image))
    t = skimage.filters.threshold_otsu(x)
    mask = (blur > t)
    return  mask.astype(np.uint8)
    

def remove_everything_above_std_and_mean(image):
    image = skimage.filters.gaussian(image, sigma=20)
    x = round(np.std(image) + np.mean(image))
    return np.uint8(x)

def draw_contours(image):
    copy_image = copy.deepcopy(image) 
    contours, hierarchy = cv.findContours(copy_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    color = (rng.randint(120,130), rng.randint(120,130), rng.randint(120,130))
    cv.drawContours(copy_image, contours, -1, color, 5, cv.LINE_8, hierarchy, 2)
    return copy_image

def process_image(image):
    image = cv.resize(image, (800, 600))
    image_grayscale = cv.cvtColor(255 - image, cv.COLOR_RGB2GRAY)
    temp = remove_everything_above_std_and_mean(image_grayscale)
    contoured_image1 = draw_contours(temp)
    plt.figure(figsize=(25,25))
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.subplot(1, 2, 2)
    plt.imshow(contoured_image1, cmap='gray')
    return contoured_image1

def count_contours(image):
    pass


if __name__ == "__main__" :
    img = cv.imread("images.png")
    cv.resize(img, (960, 720))
    contoured = normalize_image(img)
    contoured= draw_contours(contoured)
    plt.imshow(contoured, cmap='gray')
    plt.show()
