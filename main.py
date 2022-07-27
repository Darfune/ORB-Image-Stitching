from multiprocessing.connection import wait
from re import S
from statistics import mode
from urllib import response
from cv2 import ORB_create
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
import time
from scipy.spatial.distance import cdist
from skimage.feature import plot_matches
from find_matches import match
from best_keypoints import find_best_keypoints


if __name__ == '__main__':
    os.system('clear')

    path_1 = "images/set_1/"
    path_2 = "images/set_2/"
    path_3 = "images/set_3/"
    DOWNSCALE = 2
    N_LAYERS = 4
    percent = 0.000000000001
    images_1 = []


    for i in os.listdir(path_1):
        image_path = path_1 + i
        img = cv2.imread(image_path)
        # img = cv2.cvtColor(img)
        images_1.append(img)
        
    ds1 = []

    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    all_keypoints = []
    all_descriptors = []
    image_counter = 0
    all_scale_keypoints = []
    all_scale_descriptors = []
    for image in images_1:
        image_counter = image_counter + 1
        image_pyramid_gaussian = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_pyramid_gaussian.append(gray)
        height, width = gray.shape
        positions = []
        for i in range(1,4):

            # image_pyramid_gaussian.append(cv2.GaussianBlur(cv2.resize(cv2.resize(image,(new_height,new_width)),(height,width)),(5,5),0))
            image_pyramid_gaussian.append(cv2.pyrDown(gray))
        image_keypoints = ()
        octave_of_image = 0
        for current_image in image_pyramid_gaussian:
            print("Image : ", image_counter, " octave of image : ", octave_of_image)
            keypoints, scores = fast_algorithm(current_image,80)
            orientations = corner_orientations(current_image,keypoints)
            keypoints_list_temp = ()
            for i in range(0,len(keypoints)):
                # positions.append((float(keypoints[i][0]),float(keypoints[i][1])))
                
                image_keypoints = image_keypoints + (cv2.KeyPoint(x = float(keypoints[i][0]),y = float(keypoints[i][1]),size = 7 + (1.2 * octave_of_image), angle = orientations[i], response = scores[i], octave = octave_of_image, class_id = -1),)
            keypoint_image = cv2.drawKeypoints(current_image,image_keypoints,None,(0,255,0),4)
            cv2.imshow("Image : ", keypoint_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            octave_of_image = octave_of_image + 1
        # image_keypoints, image_descriptors = brief.compute(image, image_keypoints)
        # print(keypoints)


        image_keypoints = find_best_keypoints(image_keypoints,50)
        
        image_descriptors = brief_descriptor_function(gray, image_keypoints)

        # scale_keypoints = np.vstack(positions)
        # scale_descriptors = np.vstack(image_descriptors)

        all_keypoints.append(image_keypoints)
        all_descriptors.append(image_descriptors)
        # all_scale_keypoints.append(scale_keypoints)

    match(images_1[0],images_1[1],all_keypoints[0],all_keypoints[1], all_descriptors[0],all_descriptors[1])

    # matches = match(np.vstack(all_descriptors[0]),np.vstack(all_descriptors[1]))

    # fig = plt.figure(figsize=(20.0, 30.0))
    # ax = fig.add_subplot(1,1,1)
    # plot_matches(ax, images_1[0], images_1[1], np.flip(all_scale_keypoints[0], 1), np.flip(all_scale_keypoints[1], 1), matches[:20], 
    #             alignment='horizontal', only_matches=True)




    

    # matcher = cv2.BFMatcher()
    # raw_matches = matcher.knnMatch(all_descriptors[0], all_descriptors[1], k=2)
    # good_points = []
    # good_matches = []
    # for m1, m2 in raw_matches:
    #     if m1.distance < ratio * m2.distance:
    #         good_points.append((m1.trainIdx, m1.queryIdx))
    #         good_matches.append([m1])
    # matches = cv2.drawMatchesKnn(images_1[0], all_keypoints[0], images_1[1], all_keypoints[1], good_matches, None, flags=2)
    # cv2.imshow("Matches",matches)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    

        


