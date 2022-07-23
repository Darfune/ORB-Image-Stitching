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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils


if __name__ == '__main__':
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
            print("Image : ", image_counter, " octave of image : ", octave_of_image)
            keypoints, scores = fast_algorithm(current_image,80, 50)
            orientations = corner_orientations(current_image,keypoints)
            keypoints_list_temp = ()
            for i in range(0,len(keypoints)):
                image_keypoints = image_keypoints + (cv2.KeyPoint(x = float(keypoints[i][0]),y = float(keypoints[i][1]),size = 7, angle = orientations[1], response = scores[i], octave = octave_of_image, class_id = -1),)

            octave_of_image = octave_of_image + 1
        image_keypoints, image_descriptors = orb.compute(image, image_keypoints)

        all_keypoints.append(image_keypoints)
        all_descriptors.append(image_descriptors)



    ratio = 0.85
    min_match = 10

    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(all_descriptors[0], all_descriptors[1], k=2)
    good_points = []
    good_matches = []
    for m1, m2 in raw_matches:
        if m1.distance < ratio * m2.distance:
            good_points.append((m1.trainIdx, m1.queryIdx))
            good_matches.append([m1])
    matches = cv2.drawMatchesKnn(images_1[0], all_keypoints[0], images_1[1], all_keypoints[1], good_matches, None, flags=2)
    cv2.imshow("Matches",matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

        


