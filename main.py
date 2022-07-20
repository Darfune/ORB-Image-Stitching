from statistics import mode
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
    img = np.float32(img)
    images_1.append(img)

ds1 = []

for image in images_1:
    
    # image = np.array(image)
    image_pyramid_gaussian = []
    image_pyramid_gaussian.append(image)
    height, width = image.shape
    for i in range(4):
        height = math.floor(height/2)
        width = math.floor(width/2)
        image_pyramid_gaussian.append(cv2.resize(image,(height,width)))

    image_gaussian_pyramid_keypoints = []
    image_gaussian_pyramid_orientations = []

    for current_image in image_pyramid_gaussian:
        keypoints = fast_algorithm(current_image,80, 50)
        image_gaussian_pyramid_keypoints.append(keypoints)
        orientations = corner_orientations(current_image,keypoints)
        image_gaussian_pyramid_orientations.append(orientations)

        # descriptor = brief_descriptor_function(image[i],keypoints,orientations,mode = 'uniform',n = 128)
        # ds1.append(descriptor)
    # print(image_gaussian_pyramid_orientations)
    


