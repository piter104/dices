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
    # image = skimage.filters.gaussian(image, sigma=20)
    x = round(np.std(image) + np.mean(image))
    return np.array((image > x) * 255, dtype=np.uint8)


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
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(contoured_image1, cmap='gray')
    return contoured_image1

def count_contours(image):
    pass


def find_rectangle(contours, hierarchy):
    x = []
    y = []
    cubes = []

    for i in range(len(contours)):
        if len(contours[i])>30 and len(contours[i]) < 242 and hierarchy[0][i][2] != -1:
            for j in range(len(contours[i])):
                x.append(contours[i][j][0][0])
                y.append(contours[i][j][0][1])
                cubes.append([max(x), min(x), max(y), min(y)])
                x = []
                y = []

    return cubes


def fragment(image, cubes):
    return cv.resize(image[cubes[3]:cubes[2], cubes[1]:cubes[0]], (400, 400))


if __name__ == "__main__" :
    img = cv.imread("XD.jpg")
    image = cv.resize(img, (800, 600))
    image_grayscale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    
    
    temp = remove_everything_above_std_and_mean(image_grayscale)



    kernel = np.ones((3,3), np.uint8)
    temp = cv.erode(temp, kernel, iterations=2)
    temp = cv.dilate(temp, kernel, iterations=2)
            
    temp = cv.morphologyEx(temp, cv.MORPH_OPEN, kernel)
    
    contoursss, hierarchy = cv.findContours(temp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    images = []
    for j, con in enumerate(contoursss):
        x,y,w,h = cv.boundingRect(contoursss[j])
        if w < 80 or h < 80:
            continue
        cropped =  temp[y:y+h,x:x+w]
        rezised = cv.resize(cropped, (400, 400))
    
        
        color = (rng.randint(120,130), rng.randint(120,130), rng.randint(120,130))
        
        circles = cv.HoughCircles(rezised, cv.HOUGH_GRADIENT, 1, 30,
                            param1=100, param2=13,
                            minRadius=0, maxRadius=60)
    
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv.circle(rezised, center, 1, color, 4)
                # circle outline
                radius = i[2]
                cv.circle(rezised, center, radius, color, 5)
    
        images.append(rezised)

    fig = plt.figure()

    for i, img in enumerate(images):
        fig.add_subplot(len(images), 1, i+1)
        plt.imshow(img)


    plt.show()
