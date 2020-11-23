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
    

def remove_everything_below_std_and_mean(image):
    treshold = round(np.std(image) + np.mean(image))
    return np.array((image > treshold) * 255, dtype=np.uint8)


def draw_contours(image):
    copy_image = copy.deepcopy(image) 
    contours, hierarchy = cv.findContours(copy_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    
    color = (rng.randint(120,130), rng.randint(120,130), rng.randint(120,130))
    cv.drawContours(copy_image, contours, -1, color, 5, cv.LINE_8, hierarchy, 2)
    return copy_image

def process_image(image):
    image = cv.resize(image, (800, 600))
    image_grayscale = cv.cvtColor(255 - image, cv.COLOR_RGB2GRAY)
    temp = remove_everything_below_std_and_mean(image_grayscale)
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




if __name__ == "__main__" :
    image = cv.imread("airpodsy.png")
    image = cv.resize(image, (1280, 720))
    
   
    bw_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    
    plt.imshow(bw_image, cmap="binary_r")
    plt.show()
   
    temp = remove_everything_below_std_and_mean(bw_image)
    kernel = np.ones((3,3), np.uint8)
    temp = cv.erode(temp, kernel, iterations=2)
            
    

    
    plt.imshow(temp, cmap="binary_r")
    plt.show()

    all_circles = 0
    image = np.array(image, dtype=np.uint8)
    contoursss, hierarchy = cv.findContours(temp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for j, con in enumerate(contoursss):
        x,y,w,h = cv.boundingRect(contoursss[j])

        peri = cv.arcLength(con, True)
        approx = cv.approxPolyDP(con, 0.03 * peri, True)
        print(approx)

        if len(approx) != 4 or w < 30 or w > 400 or h < 30 or h > 400:
            continue

        cropped =  temp[y:y+h,x:x+w]
        rezised = cv.resize(cropped, (400, 400))
    
        


        color = (rng.randint(120,130), rng.randint(120,130), rng.randint(120,130))
        minimum_distance_between_circles_centers = 80 
        
        circles = cv.HoughCircles(rezised, cv.HOUGH_GRADIENT, 1,
                            minimum_distance_between_circles_centers,
                            param1=100, param2=14,
                            minRadius=15, maxRadius=80)
    
        if circles is None:
            continue 
        circles = np.uint16(np.around(circles))
        circles_count = len(circles[0, :])
        all_circles += circles_count
        cv.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), color, 4)
        cv.putText(image, str(circles_count), (int(x+w+10), int(y+h+10)), cv.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv.LINE_AA)

    # fig = plt.figure()

    # for i, img in enumerate(images):
    #     fig.add_subplot(len(images), 1, i+1)
    #     plt.imshow(img)

    cv.putText(image, f'Wszystkich kropek: {all_circles}', (60, 60), cv.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 100), 4, cv.LINE_AA)
    

    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    plt.show()
