import cv2 as cv
import skimage
import numpy as np
import random as rng
import copy
import matplotlib.pyplot as plt


def remove_everything_below_std_and_mean(image):
    treshold = round(np.std(image) + np.mean(image))
    return np.array((image > treshold) * 255, dtype=np.uint8)


def __draw_contours(image):
    copy_image = copy.deepcopy(image) 
    contours, hierarchy = cv.findContours(copy_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    
    color = (rng.randint(120,130), rng.randint(120,130), rng.randint(120,130))
    cv.drawContours(copy_image, contours, -1, color, 5, cv.LINE_8, hierarchy, 2)
    return copy_image


def process_image_and_draw_fragments(filtered_image):
    min_width = 30
    max_width = 400
    color = (rng.randint(120,130), rng.randint(120,130), rng.randint(120,130))
    minimum_distance_between_circles_centers = 80 

    contoursss, hierarchy = cv.findContours(filtered_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    images = []
    for j, con in enumerate(contoursss):
        x,y,w,h = cv.boundingRect(contoursss[j])

        perimeter  = cv.arcLength(con, True)
        approx = cv.approxPolyDP(con, 0.03 * perimeter , True)
        
        if len(approx) != 4 or w < min_width or w > max_width or h < min_width or h > max_width:
            continue

        cropped =  filtered_image[y:y+h,x:x+w]
        rezised = cv.resize(cropped, (400, 400))

        circles = cv.HoughCircles(rezised, cv.HOUGH_GRADIENT, 1,
                            minimum_distance_between_circles_centers,
                            param1=100, param2=14,
                            minRadius=15, maxRadius=80)

        if circles is None:
            continue 

        circles = np.uint16(np.around(circles))
        for i, circle in enumerate(circles[0, :]):
            center = (circle[0],circle[1])
            radius = circle[2]
            cv.circle(rezised,center,radius,color,3)
        
        images.append(rezised)

    
    fig = plt.figure()
    for i, img in enumerate(images):
        fig.add_subplot(len(images), 1, i+1)
        plt.imshow(img)

    plt.show()


def __process(base_image, filtered_image):
    base_image = np.array(base_image, dtype=np.uint8)
    min_width = 30
    max_width = 400
    rectangle_thickness = 4
    color = (rng.randint(120,130), rng.randint(120,130), rng.randint(120,130))
    minimum_distance_between_circles_centers = 80 

    contoursss, hierarchy = cv.findContours(filtered_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    all_circles = 0

    for j, con in enumerate(contoursss):
        x,y,w,h = cv.boundingRect(contoursss[j])

        perimeter  = cv.arcLength(con, True)
        approx = cv.approxPolyDP(con, 0.03 * perimeter , True)
        
        if len(approx) != 4 or w < min_width or w > max_width or h < min_width or h > max_width:
            continue

        cropped =  filtered_image[y:y+h,x:x+w]
        rezised = cv.resize(cropped, (400, 400))

        circles = cv.HoughCircles(rezised, cv.HOUGH_GRADIENT, 1,
                            minimum_distance_between_circles_centers,
                            param1=100, param2=14,
                            minRadius=15, maxRadius=80)

        if circles is None:
            continue 

        circles = np.uint16(np.around(circles))
        circles_count = len(circles[0, :])
        all_circles += circles_count
        cv.rectangle(base_image, (x, y), (x+w, y+h), color, rectangle_thickness)
        cv.putText(base_image, str(circles_count), (x+w+10, y+h+10), cv.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv.LINE_AA)

    cv.putText(base_image, f'Wszystkich kropek: {all_circles}', (60, 60), cv.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 100), 4, cv.LINE_AA)
    

    plt.imshow(cv.cvtColor(base_image, cv.COLOR_BGR2RGB))

    plt.show()


def process_image(file_name):
    image = cv.imread(file_name)
    image = cv.resize(image, (1280, 720))
    bw_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    filtered = remove_everything_below_std_and_mean(bw_image)
    kernel = np.ones((3,3), np.uint8)
    eroded = cv.erode(filtered, kernel, iterations=2)
    __process(image, eroded)

if __name__ == "__main__" :
    image = cv.imread("airpodsy.png")
    image = cv.resize(image, (1280, 720))
    
   
    bw_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    
    # plt.imshow(bw_image, cmap="binary_r")
    # plt.show()
   
    temp = remove_everything_below_std_and_mean(bw_image)
    kernel = np.ones((3,3), np.uint8)
    temp = cv.erode(temp, kernel, iterations=2)
            
    process_image_and_draw_fragments(temp)


    # plt.imshow(temp, cmap="binary_r")
    # plt.show()

    image = np.array(image, dtype=np.uint8)
    min_width = 30
    max_width = 400
    rectangle_thickness = 4
    color = (rng.randint(120,130), rng.randint(120,130), rng.randint(120,130))
    minimum_distance_between_circles_centers = 80 

    contoursss, hierarchy = cv.findContours(temp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    all_circles = 0

    for j, con in enumerate(contoursss):
        x,y,w,h = cv.boundingRect(contoursss[j])

        perimeter  = cv.arcLength(con, True)
        approx = cv.approxPolyDP(con, 0.03 * perimeter , True)
        print(approx)
        
        if len(approx) != 4 or w < min_width or w > max_width or h < min_width or h > max_width:
            continue

        cropped =  temp[y:y+h,x:x+w]
        rezised = cv.resize(cropped, (400, 400))
    
        circles = cv.HoughCircles(rezised, cv.HOUGH_GRADIENT, 1,
                            minimum_distance_between_circles_centers,
                            param1=100, param2=14,
                            minRadius=15, maxRadius=80)
    
        if circles is None:
            continue 

        circles = np.uint16(np.around(circles))
        circles_count = len(circles[0, :])
        all_circles += circles_count
        cv.rectangle(image, (x, y), (x+w, y+h), color, rectangle_thickness)
        cv.putText(image, str(circles_count), (x+w+10, y+h+10), cv.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv.LINE_AA)

    # fig = plt.figure()

    # for i, img in enumerate(images):
    #     fig.add_subplot(len(images), 1, i+1)
    #     plt.imshow(img)

    cv.putText(image, f'Wszystkich kropek: {all_circles}', (60, 60), cv.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 100), 4, cv.LINE_AA)
    

    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    plt.show()
