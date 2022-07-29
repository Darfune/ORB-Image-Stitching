from concurrent.futures import thread
from multiprocessing.connection import wait
from random import gauss
from re import S
import re
from statistics import mode
from turtle import down, up
from urllib import response
from cv2 import ORB_create
import numpy as np
import cv2
import os
from fast_detector import fast_detect
from orientation import corner_orientations
from skimage.transform import pyramid_gaussian
from brief import brief_descriptor_function
import math
from tempfile import TemporaryFile, tempdir
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
import time
from scipy.spatial.distance import cdist
from skimage.feature import plot_matches
from find_matches import match
from best_keypoints import find_best_keypoints
from harris_score import find_harris_corners
from find_homography import homography_stitching


def warp_image(image, homography):
    """Warps 'image' by 'homography'
    Arguments:
      image: a 3-channel image to be warped.
      homography: a 3x3 perspective projection matrix mapping points
                  in the frame of 'image' to a target frame.
    Returns:
      - a new 4-channel image containing the warped input, resized to contain
        the new image's bounds. Translation is offset so the image fits exactly
        within the bounds of the image. The fourth channel is an alpha channel
        which is zero anywhere that the warped input image does not map in the
        output, i.e. empty pixels.
      - an (x, y) tuple containing location of the warped image's upper-left
        corner in the target space of 'homography', which accounts for any
        offset translation component of the homography.
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    h, w, z = image.shape

    # Find min and max x, y of new image
    p = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]])
    p_prime = np.dot(homography, p)

    yrow = p_prime[1] / p_prime[2]
    xrow = p_prime[0] / p_prime[2]
    ymin = min(yrow)
    xmin = min(xrow)
    ymax = max(yrow)
    xmax = max(xrow)

    # Make new matrix that removes offset and multiply by homography
    new_mat = np.array([[1, 0, -1 * xmin], [0, 1, -1 * ymin], [0, 0, 1]])
    homography = np.dot(new_mat, homography)

    # height and width of new image frame
    height = int(round(ymax - ymin))
    width = int(round(xmax - xmin))
    size = (width, height)
    # Do the warp
    warped = cv2.warpPerspective(src=image, M=homography, dsize=size)

    return warped, (int(xmin), int(ymin))   



def sortScore(val):
    return val[2]

if __name__ == '__main__':
    os.system('clear')

    path_1 = "images/set_1/"
    path_2 = "images/set_2/"
    path_3 = "images/set_3/"
    DOWNSCALE = 2
    N_LAYERS = 8
    percent = 0.000000000001
    threshold = 80
    images = []

    all_keypoints = []
    all_descriptors = []

    for i in os.listdir(path_1):
        image_path = path_1 + i
        img = cv2.imread(image_path)
        # img = cv2.cvtColor(img)
        images.append(img)
    
    for image in images:
        # ###########################
        orb  = ORB_create()
        kp, des = orb.detectAndCompute(image, None)
        all_keypoints.append(kp)
        all_descriptors.append(des)
        # for keypoint in keypoints:
        #     print(type(keypoint.angle))
        #     #append the response and angle for every keypoint in a file
        #     with open("keypoints.txt", "a") as f:
        #         f.write(str(keypoint.angle) + " " + str(keypoint.response) + "\n")
        


        # exit()
        # ###########################
        gaussian_pyramid = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gaussian_pyramid.append(gray)
        layer = gray
        octave = 0
        image_keypoints = ()
        for i in range(1, N_LAYERS):
            downscale = cv2.pyrDown(layer)
            layer = downscale
            for j in range(i, 0, -1):
                downscale = cv2.pyrUp(downscale)
            gaussian_pyramid.append(downscale)
        for layer in gaussian_pyramid:
            keypoints_of_layer = fast_detect(layer, threshold)
            keypoints_of_layer, scores_of_layer = find_harris_corners(layer,100.0,keypoints_of_layer)
            
            orientations = corner_orientations(layer,keypoints_of_layer)
            for i in range(0,len(keypoints_of_layer)):
                image_keypoints = image_keypoints + (cv2.KeyPoint(
                    x = float(keypoints_of_layer[i][0]),
                    y = float(keypoints_of_layer[i][1]),
                    size = 7,
                    angle = orientations[i],
                    response = scores_of_layer[i],
                    octave = octave,
                    class_id = -1),)
            octave += 1
            layer = cv2.drawKeypoints(layer, image_keypoints, None, color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # cv2.imshow("image", layer)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        image_keypoints = find_best_keypoints(image_keypoints, 50)
        
        # image_descriptors = brief_descriptor_function(gray, all_keypoints)
        all_descriptors.append(brief_descriptor_function(gray, image_keypoints))
        # image_keypoints, descriptors = orb.compute(gray, image_keypoints)
        all_keypoints.append(image_keypoints)
        # all_descriptors.append(descriptors)
    good_matches = match(images[0],images[1],all_keypoints[0],all_keypoints[1], all_descriptors[0],all_descriptors[1])

    M = homography_stitching(all_keypoints[0], all_keypoints[1], good_matches, reprojThresh=4)

    if M is None:
        print("Error!")

    (matches, Homography_Matrix, status) = M

    print(Homography_Matrix)


    width = images[1].shape[1] + images[0].shape[1]
    print("width ", width) 
    # 2922 - Which is exactly the sum value of the width of 
    # my train.jpg and query.jpg


    height = max(images[1].shape[0], images[0].shape[0])

    # otherwise, apply a perspective warp to stitch the images together

    # Now just plug that "Homography_Matrix"  into cv::warpedPerspective and I shall have a warped image1 into image2 frame

    result = cv2.warpPerspective(images[1], Homography_Matrix,  (width, height))
    cv2.imshow("warpPerspective", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # The warpPerspective() function returns an image or video whose size is the same as the size of the original image or video. Hence set the pixels as per my query_photo

    result[0:images[0].shape[0], 0:images[0].shape[1]] = images[0]

    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        


