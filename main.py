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
import time
from scipy.spatial.distance import cdist
from skimage.feature import plot_matches

def match(descriptors1, descriptors2, max_distance=np.inf, cross_check=True, distance_ratio=None):
    distances = cdist(descriptors1, descriptors2, metric='hamming')   # distances.shape: [len(d1), len(d2)]
    
    indices1 = np.arange(descriptors1.shape[0])     # [0, 1, 2, 3, 4, 5, 6, 7, ..., len(d1)] "indices of d1"
    indices2 = np.argmin(distances, axis=1)         # [12, 465, 23, 111, 123, 45, 67, 2, 265, ..., len(d1)] "list of the indices of d2 points that are closest to d1 points"
                                                    # Each d1 point has a d2 point that is the most close to it.
    if cross_check:
        '''
        Cross check idea:
        what d1 matches with in d2 [indices2], should be equal to 
        what that point in d2 matches with in d1 [matches1]
        '''
        matches1 = np.argmin(distances, axis=0)     # [15, 37, 283, ..., len(d2)] "list of d1 points closest to d2 points"
                                                    # Each d2 point has a d1 point that is closest to it.
        # indices2 is the forward matches [d1 -> d2], while matches1 is the backward matches [d2 -> d1].
        mask = indices1 == matches1[indices2]       # len(mask) = len(d1)
        # we are basically asking does this point in d1 matches with a point in d2 that is also matching to the same point in d1 ?
        indices1 = indices1[mask]
        indices2 = indices2[mask]
    
    if max_distance < np.inf:
        mask = distances[indices1, indices2] < max_distance
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if distance_ratio is not None:
        '''
        the idea of distance_ratio is to use this ratio to remove ambigous matches.
        ambigous matches: matches where the closest match distance is similar to the second closest match distance
                          basically, the algorithm is confused about 2 points, and is not sure enough with the closest match.
        solution: if the ratio between the distance of the closest match and
                  that of the second closest match is more than the defined "distance_ratio",
                  we remove this match entirly. if not, we leave it as is.
        '''
        modified_dist = distances
        fc = np.min(modified_dist[indices1,:], axis=1)
        modified_dist[indices1, indices2] = np.inf
        fs = np.min(modified_dist[indices1,:], axis=1)
        mask = fc/fs <= 0.5
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    # sort matches using distances
    dist = distances[indices1, indices2]
    sorted_indices = dist.argsort()

    matches = np.column_stack((indices1[sorted_indices], indices2[sorted_indices]))
    return matches


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
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
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
        # image_keypoints, image_descriptors = brief.compute(image, image_keypoints)
        # print(keypoints)
        image_descriptors = brief_descriptor_function(image, keypoints, orientations)
        


        all_keypoints.append(image_keypoints)
        all_descriptors.append(image_descriptors)
    matches = match(all_descriptors[0],all_descriptors[1])

    fig = plt.figure(figsize=(20.0, 30.0))
    ax = fig.add_subplot(1,1,1)
    plot_matches(ax, images_1[0], images_1[1], np.flip(all_descriptors[0], 1), np.flip(all_descriptors[1], 1), matches[:20], 
                alignment='horizontal', only_matches=True)




    

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

    

        


