import numpy as np
import cv2
import os


def fast_detect(image, thres):
    height, width = image.shape

    keypoints = []
    
    for x in range(30,height-30):
        for y in range(30,width-30):
            
            above_thres = 0
            below_thres = 0
            if image[x-3, y+1] > image[x, y] + thres : above_thres = above_thres + 1
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
            if image[x-1, y-3] > image[x, y] + thres and above_thres < 9: above_thres = above_thres + 1
            elif above_thres >= 9:
                keypoints.append(point_found(image, x, y, thres))
                break
            elif image[x-1, y-3] < image[x, y] - thres and below_thres < 9: below_thres = below_thres + 1
            elif below_thres >= 9: 
                keypoints.append(point_found(image, x, y, thres))
                break
            if image[x, y-3] > image[x, y] + thres and above_thres < 9: above_thres = above_thres + 1
            elif above_thres >= 9:
                keypoints.append(point_found(image, x, y, thres))
                break
            elif image[x, y-3] < image[x, y] - thres  and below_thres < 9: below_thres = below_thres + 1
            elif below_thres >= 9:
                keypoints.append(point_found(image, x, y, thres))
                break
            if image[x+1, y-3] > image[x, y] + thres and above_thres < 9: above_thres = above_thres + 1
            elif above_thres >= 9:
                keypoints.append(point_found(image, x, y, thres))
                break
            elif image[x+1, y-3] < image[x, y] - thres  and below_thres < 9: below_thres = below_thres + 1
            elif below_thres >= 9:
                keypoints.append(point_found(image, x, y, thres))
                break
            ##########################################################################
            if image[x+2, y+2] > image[x, y] + thres and above_thres < 9: above_thres = above_thres + 1
            elif above_thres >= 9:
                keypoints.append(point_found(image, x, y, thres))
                break
            elif image[x+2, y+2] < image[x, y] - thres and below_thres < 9: below_thres = below_thres + 1
            elif below_thres >= 9:
                keypoints.append(point_found(image, x, y, thres))
                break
            if image[x+2, y-2] > image[x, y] + thres and above_thres < 9: above_thres = above_thres + 1
            elif above_thres >= 9:
                keypoints.append(point_found(image, x, y, thres))
                break
            elif image[x+2, y-2] < image[x, y] - thres  and below_thres < 9: below_thres = below_thres + 1
            elif below_thres >= 9:
                keypoints.append(point_found(image, x, y, thres))
                break
            ##########################################################################
            if image[x+3, y+1] > image[x, y] + thres and above_thres < 9: above_thres = above_thres + 1
            elif above_thres >= 9:
                keypoints.append(point_found(image, x, y, thres))
                break
            elif image[x+3, y+1] < image[x, y] - thres: below_thres = below_thres + 1
            elif below_thres >= 9:
                keypoints.append(point_found(image, x, y, thres))
                break
            if image[x+3, y] > image[x, y] + thres and above_thres < 9: above_thres = above_thres + 1
            elif above_thres >= 9:
                keypoints.append(point_found(image, x, y, thres))
                break
            elif image[x+3, y] < image[x, y] - thres: below_thres = below_thres + 1
            elif below_thres >= 9:
                keypoints.append(point_found(image, x, y, thres))
                break
            if image[x+3, y-1] > image[x, y] + thres and above_thres < 9: above_thres = above_thres + 1
            elif above_thres >= 9:
                keypoints.append(point_found(image, x, y, thres))
                break
            elif image[x+3, y-1] < image[x, y] - thres: below_thres = below_thres + 1
            elif above_thres >= 9:
                keypoints.append(point_found(image, x, y, thres))
                break
            ##########################################################################
    return keypoints 


def point_found(image, x, y, thres):
    keypoint = []
    score = (fast_score(image, x, y, thres))
    pixels_of_interest = ((x,y))
    keypoint.append(pixels_of_interest)
    keypoint.append(score)

    return keypoint

def is_corner(image,x,y,thres):
    above_thres = 0
    below_thres = 0
    if image[x-3, y+1] > image[x, y] + thres : above_thres = above_thres + 1
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
    if image[x-1, y-3] > image[x, y] + thres and above_thres < 9: above_thres = above_thres + 1
    elif above_thres >= 9: return 1
    elif image[x-1, y-3] < image[x, y] - thres and below_thres < 9: below_thres = below_thres + 1
    elif below_thres >= 9: return 1
    if image[x, y-3] > image[x, y] + thres and above_thres < 9: above_thres = above_thres + 1
    elif above_thres >= 9: return 1
    elif image[x, y-3] < image[x, y] - thres  and below_thres < 9: below_thres = below_thres + 1
    elif below_thres >= 9: return 1
    if image[x+1, y-3] > image[x, y] + thres and above_thres < 9: above_thres = above_thres + 1
    elif above_thres >= 9: return 1
    elif image[x+1, y-3] < image[x, y] - thres  and below_thres < 9: below_thres = below_thres + 1
    elif below_thres >= 9: return 1
    ##########################################################################
    if image[x+2, y+2] > image[x, y] + thres and above_thres < 9: above_thres = above_thres + 1
    elif above_thres >= 9: return 1
    elif image[x+2, y+2] < image[x, y] - thres and below_thres < 9: below_thres = below_thres + 1
    elif below_thres >= 9: return 1
    if image[x+2, y-2] > image[x, y] + thres and above_thres < 9: above_thres = above_thres + 1
    elif above_thres >= 9: return 1
    elif image[x+2, y-2] < image[x, y] - thres  and below_thres < 9: below_thres = below_thres + 1
    elif below_thres >= 9: return 1
    ##########################################################################
    if image[x+3, y+1] > image[x, y] + thres and above_thres < 9: above_thres = above_thres + 1
    elif above_thres >= 9: return 1
    elif image[x+3, y+1] < image[x, y] - thres: below_thres = below_thres + 1
    elif below_thres >= 9: return 1
    if image[x+3, y] > image[x, y] + thres and above_thres < 9: above_thres = above_thres + 1
    elif above_thres >= 9: return 1
    elif image[x+3, y] < image[x, y] - thres: below_thres = below_thres + 1
    elif below_thres >= 9: return 1
    if image[x+3, y-1] > image[x, y] + thres and above_thres < 9: above_thres = above_thres + 1
    elif above_thres >= 9: return 1
    elif image[x+3, y-1] < image[x, y] - thres: below_thres = below_thres + 1
    elif above_thres >= 9: return 1
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
    keypoints_scores = fast_detect(image,thres)
    keypoints = []
    best_scores = []
    if len(keypoints_scores) > 50:
        keypoints = []
        while len(keypoints) < maxpoints:
            max_score = 0
            for keypoint_score in keypoints_scores:
                if keypoint_score[1] > max_score:
                    max_score = keypoint_score[1]
                    biggest_score = keypoint_score
            
                    
            keypoints.append(biggest_score[0])
            best_scores.append(biggest_score[1])
            keypoints_scores.remove(biggest_score)
        return np.array(keypoints), best_scores
    else: 
        keypoints = []
        for keypoint_score in keypoints_scores:
                keypoints.append(keypoint_score[0])
                best_scores.append(keypoint_score[1])
        return keypoints, best_scores