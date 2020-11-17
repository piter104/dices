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
    img = cv.imread("dice_back.jpg")
    image = cv.resize(img, (800, 600))
    image_grayscale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    temp = remove_everything_above_std_and_mean(image_grayscale)
    kernel = np.ones((3,3), np.uint8)
    temp = cv.erode(temp, kernel, iterations=3)
    temp = cv.dilate(temp, kernel, iterations=6)
    
    closing = cv.morphologyEx(temp, cv.MORPH_OPEN, kernel)
    
    contoursss, hierarchy = cv.findContours(temp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    

    # cv.drawContours(temp, contours, -1, color, 5, cv.LINE_8, hierarchy, 2)
    fig = plt.figure()
    for i, con in enumerate(contoursss):
    
    
        fig.add_subplot(len(contoursss), 1, i+1)
        x,y,w,h = cv.boundingRect(contoursss[i])
        cropped =  temp[y:y+h,x:x+w]
        rezised = cv.resize(cropped, (400, 400))
    
        contours = cv.findContours(rezised.copy(), cv.RETR_TREE ,
            cv.CHAIN_APPROX_SIMPLE)[0]
        # contours.sort(key=lambda x:cv.boundingRect(x)[0])
        color = (rng.randint(120,130), rng.randint(120,130), rng.randint(120,130))
        array = []
        ii = 1
            # Approximate contours to polygons + get bounding rects and circles
        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)
        centers = [None]*len(contours)
        radius = [None]*len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv.approxPolyDP(c, 3, True)
            boundRect[i] = cv.boundingRect(contours_poly[i])
            centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
      
                # Draw polygonal contour + bonding rects + circles
        for i in range(len(contours)):
            # cv.drawContours(rezised, contours_poly, i, color, 6)
            # cv.rectangle(rezised, (int(boundRect[i][0]), int(boundRect[i][1])), \
            # (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2, 4)
            # cv.circle(rezised, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2, 4)
            pass


        circles = cv.HoughCircles(rezised, cv.HOUGH_GRADIENT, 1, 20,
                            param1=100, param2=18,
                            minRadius=0, maxRadius=100)
    
    
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv.circle(rezised, center, 1, color, 4)
                # circle outline
                radius = i[2]
                cv.circle(rezised, center, radius, color, 5)
    
        plt.imshow(rezised)

    plt.show()
