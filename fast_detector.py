import numpy as np
import cv2
import os


def fast_detect(image, thres):
    height, width = image.shape
    pixels_of_interest = 0
    score = 0
    
    keypoints = []
    for x in range(30,height-30):
        for y in range(30,width-30):
            keypoint = []
            above_thres = 0
            below_thres = 0
            if image[x-3, y+1] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x-3, y+1] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x-3, y] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x-3, y] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x-3, y-1] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x-3, y-1] < image[x, y] - thres: below_thres = below_thres + 1
            ##########################################################################
            if image[x-2, y+2] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x-2, y+2] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x-2, y-2] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x-2, y-2] < image[x, y] - thres: below_thres = below_thres + 1
            ##########################################################################
            if image[x-1, y+3] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x-1, y+3] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x, y+3] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x, y+3] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x+1, y+3] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x+1, y+3] < image[x, y] - thres: below_thres = below_thres + 1
            ##########################################################################
            if image[x-1, y-3] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x-1, y-3] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x, y-3] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x, y-3] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x+1, y-3] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x+1, y-3] < image[x, y] - thres: below_thres = below_thres + 1
            ##########################################################################
            if image[x+2, y+2] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x+2, y+2] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x+2, y-2] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x+2, y-2] < image[x, y] - thres: below_thres = below_thres + 1
            ##########################################################################
            if image[x+3, y+1] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x+3, y+1] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x+3, y] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x+3, y] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x+3, y-1] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x+3, y-1] < image[x, y] - thres: below_thres = below_thres + 1
            ##########################################################################
            if above_thres > 11 or below_thres > 11:
                score = (fast_score(image, x, y, thres))
                pixels_of_interest = ((x,y))
                keypoint.append(pixels_of_interest)
                keypoint.append(score)
                keypoints.append(keypoint)

    return keypoints


def is_corner(image,x,y,thres):
    above_thres = 0
    below_thres = 0
    if image[x-3, y+1] > image[x, y] + thres: above_thres = above_thres + 1
    elif image[x-3, y+1] < image[x, y] - thres: below_thres = below_thres + 1
    if image[x-3, y] > image[x, y] + thres: above_thres = above_thres + 1
    elif image[x-3, y] < image[x, y] - thres: below_thres = below_thres + 1
    if image[x-3, y-1] > image[x, y] + thres: above_thres = above_thres + 1
    elif image[x-3, y-1] < image[x, y] - thres: below_thres = below_thres + 1
    ##########################################################################
    if image[x-2, y+2] > image[x, y] + thres: above_thres = above_thres + 1
    elif image[x-2, y+2] < image[x, y] - thres: below_thres = below_thres + 1
    if image[x-2, y-2] > image[x, y] + thres: above_thres = above_thres + 1
    elif image[x-2, y-2] < image[x, y] - thres: below_thres = below_thres + 1
    ##########################################################################
    if image[x-1, y+3] > image[x, y] + thres: above_thres = above_thres + 1
    elif image[x-1, y+3] < image[x, y] - thres: below_thres = below_thres + 1
    if image[x, y+3] > image[x, y] + thres: above_thres = above_thres + 1
    elif image[x, y+3] < image[x, y] - thres: below_thres = below_thres + 1
    if image[x+1, y+3] > image[x, y] + thres: above_thres = above_thres + 1
    elif image[x+1, y+3] < image[x, y] - thres: below_thres = below_thres + 1
    ##########################################################################
    if image[x-1, y-3] > image[x, y] + thres: above_thres = above_thres + 1
    elif image[x-1, y-3] < image[x, y] - thres: below_thres = below_thres + 1
    if image[x, y-3] > image[x, y] + thres: above_thres = above_thres + 1
    elif image[x, y-3] < image[x, y] - thres: below_thres = below_thres + 1
    if image[x+1, y-3] > image[x, y] + thres: above_thres = above_thres + 1
    elif image[x+1, y-3] < image[x, y] - thres: below_thres = below_thres + 1
    ##########################################################################
    if image[x+2, y+2] > image[x, y] + thres: above_thres = above_thres + 1
    elif image[x+2, y+2] < image[x, y] - thres: below_thres = below_thres + 1
    if image[x+2, y-2] > image[x, y] + thres: above_thres = above_thres + 1
    elif image[x+2, y-2] < image[x, y] - thres: below_thres = below_thres + 1
    ##########################################################################
    if image[x+3, y+1] > image[x, y] + thres: above_thres = above_thres + 1
    elif image[x+3, y+1] < image[x, y] - thres: below_thres = below_thres + 1
    if image[x+3, y] > image[x, y] + thres: above_thres = above_thres + 1
    elif image[x+3, y] < image[x, y] - thres: below_thres = below_thres + 1
    if image[x+3, y-1] > image[x, y] + thres: above_thres = above_thres + 1
    elif image[x+3, y-1] < image[x, y] - thres: below_thres = below_thres + 1
    ##########################################################################
    if above_thres > 11 or below_thres > 11:
        return 1
    else: return 0

def fast_score(image,x,y,thres):
    bmin = 0
    bmax = 255
    b = (bmax + bmin) / 2

    while True:
        if is_corner(image,x,y,b):
            bmin = b
        else: bmax = b

        if bmin >= bmax -1 or bmin == bmax:
            return bmin * 0.00001
        
        b = (bmax + bmin) / 2


def fast_algorithm (image,thres = 80,maxpoints = 50):
    temp_keypoints = fast_detect(image,thres)
    best_scores = []
    if len(temp_keypoints) > 50:
        keypoints = []
        while len(keypoints) < maxpoints:
            max_score = 0
            for temp_keypoint in temp_keypoints:
                if temp_keypoint[1] > max_score:
                    max_score = temp_keypoint[1]
                    biggest_score = temp_keypoint
            
                    
            keypoints.append(biggest_score[0])
            best_scores.append(biggest_score[1])
            temp_keypoints.remove(biggest_score)
        return np.array(keypoints), best_scores
    else: 
        keypoints = []
        for temp_keypoint in temp_keypoints:
                keypoints.append(temp_keypoint[0])
                best_scores.append(temp_keypoint[1])
        return np.array(keypoints), best_scores