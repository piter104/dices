import cv2 as cv
import skimage
import numpy as np
import random as rng
import copy

def normalize_image(image):
    blur = skimage.filters.gaussian(image, sigma=20)
    t = skimage.filters.threshold_otsu(blur)
    mask = (blur > t)
    return  mask.astype(np.uint8)
    

def remove_everything_above_std_and_mean(image):
    x = round(np.std(image) + np.mean(image) )
    return np.array((image > x) * 255, dtype=np.uint8)



def draw_contours(image):
    copy_image = copy.deepcopy(image) 
    contours, hierarchy = cv.findContours(copy_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    color = (rng.randint(200,256), rng.randint(220,250), rng.randint(0,20))
    cv.drawContours(copy_image, contours, -1, color, 5, cv.LINE_8, hierarchy, 2)
    return copy_image

def count_contours(image):
    pass


if __name__ == "__main__" :
    import matplotlib.pyplot as plt

    img = cv.imread("images.png")
    cv.resize(img, (960, 720))
    contoured = normalize_image(img)
    contoured= draw_contours(contoured)
    plt.imshow(contoured, cmap='gray')
    plt.show()
