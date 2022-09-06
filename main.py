from concurrent.futures import thread
import multiprocessing
from multiprocessing import process
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
import concurrent.futures
from objects import keypoint
from pattern import generate_pattern


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
all_keypoints = []

def keypoint_details_processing(pyramid_details):
    octave = pyramid_details[0]
    threshold = pyramid_details[1]
    image = pyramid_details[2]
    
    keypoints_of_image = fast_detect(image, threshold, octave)
    keypoints_of_image = find_harris_corners(image,1000.0,keypoints_of_image)
    
    keypoints_of_image = corner_orientations(image,keypoints_of_image)
    print("Finished keypoint detection")
    return keypoints_of_image


def find_keypoints_across_octaves(keypoints, octaves):
    print("Finding keypoints across octaves")
    to_return_keypoints = ()
    for kp_one in keypoints:
        if kp_one.octave == 0:
            count = 0
            kp_family = []
            best_of_octave = kp_one
            for kp_two in keypoints:
                kp_family.append(kp_one)
                if kp_one.pt == kp_two.pt and kp_one.octave != kp_two.octave and kp_one.angle == kp_two.angle:
                    kp_family.append(kp_two)
                    count += 1
                    if kp_one.response < kp_two.response:
                        best_of_octave = kp_two
            if count == octaves:
                to_return_keypoints += (best_of_octave,)
                kp_temp = list(keypoints)
                for kp in kp_family:
                    kp_temp.remove(kp)
                kp_family.clear()
                keypoints = tuple(kp_temp)
    print("Finished finding keypoints across octaves")
    return to_return_keypoints
                


def topN_kepoints(keypoints, n):
    print("Finding top {} keypoints".format(n))
    size = len(keypoints)
    keypoints = list(keypoints)
    for index in range(size):
        min_index = index

        for j in range(index + 1, size):
            if keypoints[j].response > keypoints[min_index].response:
                min_index = j

        keypoints[index], keypoints[min_index] = keypoints[min_index], keypoints[index]
    print("Finished finding top {} keypoints".format(n))
    return tuple(keypoints[:n])


def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame



if __name__ == '__main__':
    os.system('clear')
    path_pada = "images/pada_images/"
    path_1 = "images/set_1/"
    path_2 = "images/set_2/"
    path_3 = "images/set_3/"
    DOWNSCALE = 2
    N_LAYERS = 4
    percent = 0.000000000001
    threshold = 20
    images = []

    pattern = generate_pattern()
    
    all_descriptors = []
    all_keypoints = []
    for i in os.listdir(path_pada):
        image_path = path_pada + i
        img = cv2.imread(image_path)
        # img = cv2.cvtColor(img)
        # img = cv2.resize(img, (math.floor(img.shape[1]/6),math.floor(img.shape[0]/6)))
        images.append(img)
    keypoints_and_octaves = []
    pack = []
    for image in images:
        image_keypoints = ()
        # ###########################
        orb  = ORB_create()
        # kp_orb, des = orb.detectAndCompute(image, None)
        # for k_point in kp_orb:
        #     print ("angle = {}".format((k_point.angle)))
        # # exit()
        # all_keypoints.append(kp)
        # all_descriptors.append(des)
        # # ###########################
        gaussian_pyramid = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gaussian_pyramid.append([0,threshold,gray])
        layer = gray
        for i in range(1, N_LAYERS):
            downscale = gray
            for j in range(i):
                downscale = cv2.pyrDown(downscale)
            for j in range(i):
                downscale = cv2.pyrUp(downscale)
            
            gaussian_pyramid.append([i, threshold,downscale])


        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(keypoint_details_processing, gp) for gp in gaussian_pyramid]
            for future in concurrent.futures.as_completed(results):
                
                
                for kp in future.result():
                    
                    image_keypoints = image_keypoints + (cv2.KeyPoint(
                        x = float(kp.x),
                        y = float(kp.y),
                        size = 7 * (2**kp.octave),
                        angle = kp.orientation,
                        response = kp.score,
                        octave = kp.octave,
                        class_id = -1),)
                
                
        find_keypoints_across_octaves(image_keypoints,N_LAYERS)
        image_keypoints = topN_kepoints(image_keypoints, 100)
        image_with_keypoints = cv2.drawKeypoints(image, image_keypoints, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow('image_1', image_with_keypoints)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # image_keypoints = orb.detect(image)
        # image_keypoints, des = orb.compute(image, image_keypoints)
        # better_image_with_keypoints = cv2.drawKeypoints(image, image_keypoints, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow('image_2', better_image_with_keypoints)
        # cv2.waitKey(0)
        
        # for k_p in image_keypoints:
        #     for kp2 in kp_orb:
        #         if k_p.pt == kp2.pt:
        #             print(f"MyORB: {k_p.angle} vs ORB: {kp2.angle}")
        # exit()
        all_keypoints.append(image_keypoints)
        all_descriptors.append(brief_descriptor_function(gray, image_keypoints, pattern = pattern))
        # all_descriptors.append(des)
        # ###########################

    
    good_matches = match(images[0],images[1],all_keypoints[0],all_keypoints[1],all_descriptors[0],all_descriptors[1])
    M = homography_stitching(all_keypoints[0], all_keypoints[1], good_matches, reprojThresh=4)

    if M is None:
        print("Error!")
            
    else:
        (matches, Homography_Matrix, mask) = M
    


        print(Homography_Matrix)


        width = images[1].shape[1] + images[0].shape[1]
        
        h, w = images[0].shape[:2]
        print(h, w)

        height = max(images[1].shape[0], images[0].shape[0])

        # otherwise, apply a perspective warp to stitch the images together

        # Now just plug that "Homography_Matrix"  into cv::warpedPerspective and I shall have a warped image1 into image2 frame



        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w-1, 0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, Homography_Matrix)

        
        result = cv2.warpPerspective(images[1], Homography_Matrix,  (width, height))


        i1x , i1y = images[0].shape[:2]
        i2x , i2y = result.shape[:2]
        for i in range(0, i1x):
            for j in range(0, i1y):
                try:
                    if(np.array_equal(images[0][j,i],np.array([0,0,0])) and  \
                        np.array_equal(result[j,i],np.array([0,0,0]))):
                        # print "BLACK"
                        # instead of just putting it with black, 
                        # take average of all nearby values and avg it.
                        result[j,i] = [0, 0, 0]
                    else:
                        if(np.array_equal(result[j,i],[0,0,0])):
                            # print "PIXEL"
                            result[j,i] = images[0][j,i]
                        else:
                            if not np.array_equal(images[0][j,i], [0,0,0]):
                                bl,gl,rl = images[0][j,i]                               
                                result[j, i] = [bl,gl,rl]
                except:
                    pass
        crop_result = trim(result)
        cv2.imshow("waRPED mix", crop_result)
        cv2.waitKey()
        exit()

        # alpha = 0.5
        cv2.imshow("warpPerspective", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        crop_result = trim(result)
        result[0:images[0].shape[0], 0:images[0].shape[1]] = images[0]

        temp_result =  result[:images[0].shape[0], crop_result.shape[1] - images[0].shape[1]:images[0].shape[1]]
        temp_img1 = images[0][:images[0].shape[0], crop_result.shape[1] - images[0].shape[1]:images[0].shape[1]]

        


        result[0:images[0].shape[0], 0:images[0].shape[1]] = images[0]
        
        cv2.imshow("final", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    


    # The warpPerspective() function returns an image or video whose size is the same as the size of the original image or video. Hence set the pixels as per my query_photo
    cv2.imshow("result", trim(result))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
        


