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



def create_mask(img1,img2,version):
    height_img1 = img1.shape[0]
    width_img1 = img1.shape[1]
    width_img2 = img2.shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 + width_img2
    offset = int(smoothing_window_size / 2)
    barrier = img1.shape[1] - int(smoothing_window_size / 2)
    mask = np.zeros((height_panorama, width_panorama))
    if version== 'left_image':
        mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
        mask[:, :barrier - offset] = 1
    else:

        mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
        mask[:, barrier + offset:] = 1
    return cv2.merge([mask, mask, mask])




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
            keypoints_list_temp = ()
            for i in range(0,len(keypoints)):
                image_keypoints = image_keypoints + (cv2.KeyPoint(x = float(keypoints[i][0]),y = float(keypoints[i][1]),size = 7, angle = orientations[1], response = scores[i], octave = octave_of_image, class_id = -1),)

            octave_of_image = octave_of_image + 1
        image_keypoints, image_descriptors = orb.compute(image, image_keypoints)

        # kp_image = cv2.drawKeypoints(image, image_keypoints, None)
        # cv2.imshow("Keypoints",kp_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

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

    if len(good_points) > min_match:
        image1_kp = np.float32(
            [all_keypoints[0][i].pt for (_, i) in good_points])
        image2_kp = np.float32(
            [all_keypoints[1][i].pt for (i, _) in good_points])
        H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)      # RANSAC


    smoothing_window_size = 32
    # H = registration(img1,img2)
    height_img1 = images_1[0].shape[0]
    width_img1 = images_1[0].shape[1]
    width_img2 = images_1[1].shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 + width_img2

    panorama1 = np.zeros((height_panorama, width_panorama, 3))
    mask1 = create_mask(images_1[0], images_1[1], version='right_image')
    panorama1[0:images_1[0].shape[0], 0:images_1[0].shape[1], :] = images_1[0]
    panorama1 *= mask1
    mask2 = create_mask(images_1[0],images_1[1],version='left_image')
    panorama2 = cv2.warpPerspective(images_1[1], H, (width_panorama, height_panorama))*mask2
    result = panorama1 + panorama2

    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    final_result = result[min_row:max_row, min_col:max_col, :]

    test = cv2.warpPerspective(images_1[1], H, (width_panorama, height_panorama))
    plt.figure(figsize=(15,10))
    plt.imshow(test/255.0)
    plt.figure(figsize=(15,10))
    plt.imshow(panorama1/255.0)
    plt.figure(figsize=(15,10))
    plt.imshow(panorama2/255.0)
    plt.figure(figsize=(15,10))
    plt.imshow(final_result/255.0)
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images_1)
    plt.figure(figsize=(15,10))
    plt.imshow(stitched)
    (status_1, stitched_1) = stitcher.stitch([images_1[0], images_1[1]])
    plt.figure(figsize=(15,10))
    plt.imshow(stitched_1)
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

        


