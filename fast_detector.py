from ast import If
import numpy as np
from harris_score import find_harris_corners
import cv2
import time
import math
from objects import keypoint


def fast_detect(image, thres, octave):

    print("in fast_detect")
    height, width = image.shape
    keypoints = []

    radian = math.floor((7 * (2**octave))/2)
    match octave:
        case 0:
            max = 9
        case 1:
            max = 18
        case 2:
            max = 32
        case 3:
            max = 64
        case 4:
            max = 128
        case 5:
            max = 256
        case 6:
            max = 512
        case 7:
            max = 1024
        


    for h in range(radian + 10,height-radian - 10):
        for w in range(radian + 10,width-radian - 10):
            
            above_thres = 0
            below_thres = 0
            # main 4 corners
            if image[h-radian, w] > image[h, w] + thres: above_thres = above_thres + 1
            elif image[h-radian, w] < image[h, w] - thres: below_thres = below_thres + 1
            if image[h+radian, w] > image[h, w] + thres: above_thres = above_thres + 1
            elif image[h+radian, w] < image[h, w] - thres: below_thres = below_thres + 1
            if above_thres == 2:
                if image[h, w-radian] > image[h, w] + thres: above_thres = above_thres + 1
                if image[h, w+radian] > image[h, w] + thres: above_thres = above_thres + 1
                if above_thres >= 3 :
                    if image[h-radian, w-1] > image[h, w] + thres: above_thres = above_thres + 1
                    if image[h-radian, w+1] > image[h, w] + thres: above_thres = above_thres + 1
                    if image[h+radian, w-1] > image[h, w] + thres: above_thres = above_thres + 1
                    if image[h+radian, w+1] > image[h, w] + thres: above_thres = above_thres + 1

                    if image[h-1, w-radian] > image[h, w] + thres: above_thres = above_thres + 1
                    if image[h+1, w-radian] > image[h, w] + thres: above_thres = above_thres + 1
                    if image[h-1, w+radian] > image[h, w] + thres: above_thres = above_thres + 1
                    if image[h+1, w+radian] > image[h, w] + thres: above_thres = above_thres + 1

                    for i in range(1,radian-1):
                        if image[h+(radian-i), w+i+1] > image[h, w] + thres: above_thres = above_thres + 1
                        if image[h-(radian-i), w+i+1] > image[h, w] + thres: above_thres = above_thres + 1
                        if image[h+(radian-i), w-i+1] > image[h, w] + thres: above_thres = above_thres + 1
                        if image[h-(radian-i), w-i+1] > image[h, w] + thres: above_thres = above_thres + 1
                    if above_thres >= max:
                        kp = keypoint(w,h, octave=octave)
                        keypoints.append(kp)

            elif below_thres == 2:
                if image[h, w-radian] < image[h, w] - thres: below_thres = below_thres + 1
                if image[h, w+radian] < image[h, w] - thres: below_thres = below_thres + 1
                if below_thres >= 3 :
                    if image[h-radian, w-1] < image[h, w] - thres: below_thres = below_thres + 1
                    if image[h-radian, w+1] < image[h, w] - thres: below_thres = below_thres + 1
                    if image[h+radian, w-1] < image[h, w] - thres: below_thres = below_thres + 1
                    if image[h+radian, w+1] < image[h, w] - thres: below_thres = below_thres + 1

                    if image[h-1, w-radian] < image[h, w] - thres: below_thres = below_thres + 1
                    if image[h+1, w-radian] < image[h, w] - thres: below_thres = below_thres + 1
                    if image[h-1, w+radian] < image[h, w] - thres: below_thres = below_thres + 1
                    if image[h+1, w+radian] < image[h, w] - thres: below_thres = below_thres + 1
                    for i in range(1,radian-1):
                        if image[h+(radian-i), w+i+1] < image[h, w] - thres: below_thres = below_thres + 1
                        if image[h-(radian-i), w+i+1] < image[h, w] - thres: below_thres = below_thres + 1
                        if image[h+(radian-i), w-i+1] < image[h, w] - thres: below_thres = below_thres + 1
                        if image[h-(radian-i), w-i+1] < image[h, w] - thres: below_thres = below_thres + 1

                    if below_thres >= max:
                        kp = keypoint(w,h, octave= octave)
                        keypoints.append(kp)

    print(f"Total features detected for layer {octave}: {len(keypoints)}")

    # keypoints = non_max_suppression(keypoints, image, thres, octave)

    return keypoints 

# def non_max_suppression(keypoints, image, thres, octave):
#     print("in non_max_suppression")
#     for kp in keypoints:
#         window = 
#     return keypoints
