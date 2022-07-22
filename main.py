from statistics import mode
from urllib import response
import numpy as np
import cv2
import os
from fast_detector import fast_algorithm
from orientation import corner_orientations
from skimage.transform import pyramid_gaussian
from brief import brief_descriptor_function
import math
from tempfile import TemporaryFile

os.system('clear')

path_1 = "images/set_1/"
path_2 = "images/set_2/"
path_3 = "images/set_3/"
DOWNSCALE = 2
N_LAYERS = 4

images_1 = []

for i in os.listdir(path_1):
    image_path = path_1 + i
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images_1.append(img)

ds1 = []
orb = cv2.ORB_create(50)

all_keypoints = []
all_descriptors = []
image_counter = 0

for image in images_1:
    image_counter = image_counter + 1
    print("Image counter: ", image_counter)
    image_pyramid_gaussian = []
    image_pyramid_gaussian.append(image)
    height, width = image.shape
    for i in range(1,4):
        new_height = math.floor(height/i)
        new_width = math.floor(width/i)
        # image_pyramid_gaussian.append(cv2.GaussianBlur(cv2.resize(cv2.resize(image,(new_height,new_width)),(height,width)),(5,5),0))
        image_pyramid_gaussian.append(cv2.resize(cv2.resize(image,(new_height,new_width)),(height,width)))
    image_keypoints = ()
    octave_of_image = 0
    for current_image in image_pyramid_gaussian:
        keypoints, scores = fast_algorithm(current_image,80, 50)

        orientations = corner_orientations(current_image,keypoints)
        print(len(keypoints))
        keypoints_list_temp = ()
        for i in range(0,len(keypoints)):
            image_keypoints = image_keypoints + (cv2.KeyPoint(x = float(keypoints[i][0]),y = float(keypoints[i][1]),size = 7, angle = orientations[1], response = scores[i], octave = octave_of_image, class_id = -1),)

        octave_of_image = octave_of_image + 1

    all_keypoints.append(image_keypoints)
    # kp, des = orb.detectAndCompute(image,None)
    # kp_image1 = cv2.drawKeypoints(image, keypoints_list, None)
    # cv2.imshow("Keypoints",kp_image1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
        # descriptor = brief_descriptor_function(current_image,keypoints,orientations,mode = 'uniform',n = 128)
        # # ds1.append(descriptor)
        # print(descriptor)
        # exit(0)
    # print(image_gaussian_pyramid_orientations)

    


