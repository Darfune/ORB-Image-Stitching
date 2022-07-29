from ast import If
import numpy as np
from harris_score import find_harris_corners
import cv2
import time
def fast_detect(image, thres):
    height, width = image.shape
    keypoints = []


    for h in range(3,height-3):
        for w in range(3,width-3):
            
            above_thres = 0
            below_thres = 0
            # main 4 corners
            if image[h-3, w] > image[h, w] + thres: above_thres = above_thres + 1
            elif image[h-3, w] < image[h, w] - thres: below_thres = below_thres + 1
            if image[h+3, w] > image[h, w] + thres: above_thres = above_thres + 1
            elif image[h+3, w] < image[h, w] - thres: below_thres = below_thres + 1
            if image[h, w-3] > image[h, w] + thres: above_thres = above_thres + 1
            elif image[h,w-3] < image[h, w] - thres: below_thres = below_thres + 1
            if image[h, w+3] > image[h, w] + thres: above_thres = above_thres + 1
            elif image[h, w+3] > image[h, w] - thres: below_thres = below_thres + 1
            if above_thres >= 3 :
                if image[h-3, w+1] > image[h, w] + thres: above_thres = above_thres + 1
                if image[h-3, w-1] > image[h, w] + thres: above_thres = above_thres + 1
                if image[h-2, w+2] > image[h, w] + thres: above_thres = above_thres + 1
                if image[h-2, w-2] > image[h, w] + thres: above_thres = above_thres + 1
                if image[h-1, w+3] > image[h, w] + thres: above_thres = above_thres + 1
                if image[h+1, w+3] > image[h, w] + thres: above_thres = above_thres + 1
                if image[h-1, w-3] > image[h, w] + thres: above_thres = above_thres + 1
                if image[h+1, w-3] > image[h, w] + thres: above_thres = above_thres + 1
                if image[h+2, w+2] > image[h, w] + thres: above_thres = above_thres + 1
                if image[h+2, w-2] > image[h, w] + thres: above_thres = above_thres + 1
                if image[h+3, w+1] > image[h, w] + thres: above_thres = above_thres + 1
                if image[h+3, w-1] > image[h, w] + thres: above_thres = above_thres + 1
                if above_thres >= 9:
                    keypoints.append((w,h))
            elif below_thres >= 3 :
                if image[h-3, w+1] < image[h, w] - thres: below_thres = below_thres + 1
                if image[h-3, w-1] < image[h, w] - thres: below_thres = below_thres + 1
                if image[h-2, w+2] < image[h, w] - thres: below_thres = below_thres + 1
                if image[h-2, w-2] < image[h, w] - thres: below_thres = below_thres + 1
                if image[h-1, w+3] < image[h, w] - thres: below_thres = below_thres + 1
                if image[h+1, w+3] < image[h, w] - thres: below_thres = below_thres + 1
                if image[h-1, w-3] < image[h, w] - thres: below_thres = below_thres + 1
                if image[h+1, w-3] < image[h, w] - thres: below_thres = below_thres + 1
                if image[h+2, w+2] < image[h, w] - thres: below_thres = below_thres + 1
                if image[h+2, w-2] < image[h, w] - thres: below_thres = below_thres + 1
                if image[h+3, w+1] < image[h, w] - thres: below_thres = below_thres + 1
                if image[h+3, w-1] < image[h, w] - thres: below_thres = below_thres + 1
                if below_thres >= 9:
                    keypoints.append((w, h),)

    return keypoints 

def is_corner(image,h,w,thres):
    above_thres = 0
    below_thres = 0
    if image[h-3, w] > image[h, w] + thres: above_thres = above_thres + 1
    elif image[h-3, w] < image[h, w] - thres: below_thres = below_thres + 1
    if image[h+3, w] > image[h, w] + thres: above_thres = above_thres + 1
    elif image[h+3, w] < image[h, w] - thres: below_thres = below_thres + 1
    if image[h, w-3] > image[h, w] + thres: above_thres = above_thres + 1
    elif image[h,w-3] < image[h, w] - thres: below_thres = below_thres + 1
    if image[h, w+3] > image[h, w] + thres: above_thres = above_thres + 1
    elif image[h, w+3] > image[h, w] - thres: below_thres = above_thres + 1
    if above_thres >= 3 :
        if image[h-3, w+1] > image[h, w] + thres: above_thres = above_thres + 1
        if image[h-3, w-1] > image[h, w] + thres: above_thres = above_thres + 1
        if image[h-2, w+2] > image[h, w] + thres: above_thres = above_thres + 1
        if image[h-2, w-2] > image[h, w] + thres: above_thres = above_thres + 1
        if image[h-1, w+3] > image[h, w] + thres: above_thres = above_thres + 1
        if image[h+1, w+3] > image[h, w] + thres: above_thres = above_thres + 1
        if image[h-1, w-3] > image[h, w] + thres: above_thres = above_thres + 1
        if image[h+1, w-3] > image[h, w] + thres: above_thres = above_thres + 1
        if image[h+2, w+2] > image[h, w] + thres: above_thres = above_thres + 1
        if image[h+2, w-2] > image[h, w] + thres: above_thres = above_thres + 1
        if image[h+3, w+1] > image[h, w] + thres: above_thres = above_thres + 1
        if image[h+3, w-1] > image[h, w] + thres: above_thres = above_thres + 1
    elif below_thres >= 3 :
        if image[h-3, w+1] < image[h, w] - thres: below_thres = below_thres + 1
        if image[h-3, w-1] < image[h, w] - thres: below_thres = below_thres + 1
        if image[h-2, w+2] < image[h, w] - thres: below_thres = below_thres + 1
        if image[h-2, w-2] < image[h, w] - thres: below_thres = below_thres + 1
        if image[h-1, w+3] < image[h, w] - thres: below_thres = below_thres + 1
        if image[h+1, w+3] < image[h, w] - thres: below_thres = below_thres + 1
        if image[h-1, w-3] < image[h, w] - thres: below_thres = below_thres + 1
        if image[h+1, w-3] < image[h, w] - thres: below_thres = below_thres + 1
        if image[h+2, w+2] < image[h, w] - thres: below_thres = below_thres + 1
        if image[h+2, w-2] < image[h, w] - thres: below_thres = below_thres + 1
        if image[h+3, w+1] < image[h, w] - thres: below_thres = below_thres + 1
        if image[h+3, w-1] < image[h, w] - thres: below_thres = below_thres + 1
    else: return 0
    if above_thres >= 9 or below_thres >= 9: return 1

def fast_score(image,h,w,thres):
    bmin = 0
    bmah = 255
    b = (bmah + bmin) / 2

    while True:
        if is_corner(image,h,w,b):
            bmin = b
        else: bmah = b

        if bmin >= bmah -1 or bmin == bmah:
            return bmin * 0.00001
        
        b = (bmah + bmin) / 2




