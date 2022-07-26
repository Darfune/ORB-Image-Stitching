import re
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

            if image[x-3, y] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x-3, y] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x+3, y] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x+3, y] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x, y-3] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x,y-3] < image[x, y] - thres: below_thres = below_thres + 1
            if image[x, y+3] > image[x, y] + thres: above_thres = above_thres + 1
            elif image[x, y+3] > image[x, y] - thres: below_thres = above_thres + 1
            if above_thres >= 3 :
                if image[x-3, y+1] > image[x, y] + thres: above_thres = above_thres + 1
                if image[x-3, y-1] > image[x, y] + thres: above_thres = above_thres + 1
                if image[x-2, y+2] > image[x, y] + thres: above_thres = above_thres + 1
                if image[x-2, y-2] > image[x, y] + thres: above_thres = above_thres + 1
                if image[x-1, y+3] > image[x, y] + thres: above_thres = above_thres + 1
                if image[x+1, y+3] > image[x, y] + thres: above_thres = above_thres + 1
                if image[x-1, y-3] > image[x, y] + thres: above_thres = above_thres + 1
                if image[x+1, y-3] > image[x, y] + thres: above_thres = above_thres + 1
                if image[x+2, y+2] > image[x, y] + thres: above_thres = above_thres + 1
                if image[x+2, y-2] > image[x, y] + thres: above_thres = above_thres + 1
                if image[x+3, y+1] > image[x, y] + thres: above_thres = above_thres + 1
                if image[x+3, y-1] > image[x, y] + thres: above_thres = above_thres + 1
            elif below_thres >= 3 :
                if image[x-3, y+1] < image[x, y] - thres: below_thres = below_thres + 1
                if image[x-3, y-1] < image[x, y] - thres: below_thres = below_thres + 1
                if image[x-2, y+2] < image[x, y] - thres: below_thres = below_thres + 1
                if image[x-2, y-2] < image[x, y] - thres: below_thres = below_thres + 1
                if image[x-1, y+3] < image[x, y] - thres: below_thres = below_thres + 1
                if image[x+1, y+3] < image[x, y] - thres: below_thres = below_thres + 1
                if image[x-1, y-3] < image[x, y] - thres: below_thres = below_thres + 1
                if image[x+1, y-3] < image[x, y] - thres: below_thres = below_thres + 1
                if image[x+2, y+2] < image[x, y] - thres: below_thres = below_thres + 1
                if image[x+2, y-2] < image[x, y] - thres: below_thres = below_thres + 1
                if image[x+3, y+1] < image[x, y] - thres: below_thres = below_thres + 1
                if image[x+3, y-1] < image[x, y] - thres: below_thres = below_thres + 1

            if above_thres >= 7 or below_thres >= 7:
                # keypoints.append(point_found(image, x, y, thres))
                keypoints.append((x, y),)

    return keypoints 


# def point_found(image, x, y, thres):
#     keypoint = []
#     # score = (fast_score(image, x, y, thres))
#     # score = (harris_score(image, x, y, thres))
#     pixels_of_interest = ((x,y))
#     keypoint.append(pixels_of_interest)
#     keypoint.append(score)

    # return keypoint


def find_harris_corners(input_img, threshold):
    
    corner_list = []
    output_img = cv2.cvtColor(input_img.copy(), cv2.COLOR_GRAY2RGB)
    
    offset = int(5/2)
    y_range = input_img.shape[0] - offset
    x_range = input_img.shape[1] - offset
    
    
    dy, dx = np.gradient(input_img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    
    
    for y in range(offset, y_range):
        for x in range(offset, x_range):
            
            #Values of sliding window
            start_y = y - offset
            end_y = y + offset + 1
            start_x = x - offset
            end_x = x + offset + 1
            
            #The variable names are representative to 
            #the variable of the Harris corner equation
            windowIxx = Ixx[start_y : end_y, start_x : end_x]
            windowIxy = Ixy[start_y : end_y, start_x : end_x]
            windowIyy = Iyy[start_y : end_y, start_x : end_x]
            
            #Sum of squares of intensities of partial derevatives 
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            #Calculate determinant and trace of the matrix
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            
            #Calculate r for Harris Corner equation
            r = det - 0.06*(trace**2)
            
            if r > threshold:
                corner_list.append([x, y, r])

    return corner_list


def is_corner(image,x,y,thres):
    above_thres = 0
    below_thres = 0
    if image[x-3, y] > image[x, y] + thres: above_thres = above_thres + 1
    elif image[x-3, y] < image[x, y] - thres: below_thres = below_thres + 1
    if image[x+3, y] > image[x, y] + thres: above_thres = above_thres + 1
    elif image[x+3, y] < image[x, y] - thres: below_thres = below_thres + 1
    if image[x, y-3] > image[x, y] + thres: above_thres = above_thres + 1
    elif image[x,y-3] < image[x, y] - thres: below_thres = below_thres + 1
    if image[x, y+3] > image[x, y] + thres: above_thres = above_thres + 1
    elif image[x, y+3] > image[x, y] - thres: below_thres = above_thres + 1
    if above_thres >= 3 :
        if image[x-3, y+1] > image[x, y] + thres: above_thres = above_thres + 1
        if image[x-3, y-1] > image[x, y] + thres: above_thres = above_thres + 1
        if image[x-2, y+2] > image[x, y] + thres: above_thres = above_thres + 1
        if image[x-2, y-2] > image[x, y] + thres: above_thres = above_thres + 1
        if image[x-1, y+3] > image[x, y] + thres: above_thres = above_thres + 1
        if image[x+1, y+3] > image[x, y] + thres: above_thres = above_thres + 1
        if image[x-1, y-3] > image[x, y] + thres: above_thres = above_thres + 1
        if image[x+1, y-3] > image[x, y] + thres: above_thres = above_thres + 1
        if image[x+2, y+2] > image[x, y] + thres: above_thres = above_thres + 1
        if image[x+2, y-2] > image[x, y] + thres: above_thres = above_thres + 1
        if image[x+3, y+1] > image[x, y] + thres: above_thres = above_thres + 1
        if image[x+3, y-1] > image[x, y] + thres: above_thres = above_thres + 1
    elif below_thres >= 3 :
        if image[x-3, y+1] < image[x, y] - thres: below_thres = below_thres + 1
        if image[x-3, y-1] < image[x, y] - thres: below_thres = below_thres + 1
        if image[x-2, y+2] < image[x, y] - thres: below_thres = below_thres + 1
        if image[x-2, y-2] < image[x, y] - thres: below_thres = below_thres + 1
        if image[x-1, y+3] < image[x, y] - thres: below_thres = below_thres + 1
        if image[x+1, y+3] < image[x, y] - thres: below_thres = below_thres + 1
        if image[x-1, y-3] < image[x, y] - thres: below_thres = below_thres + 1
        if image[x+1, y-3] < image[x, y] - thres: below_thres = below_thres + 1
        if image[x+2, y+2] < image[x, y] - thres: below_thres = below_thres + 1
        if image[x+2, y-2] < image[x, y] - thres: below_thres = below_thres + 1
        if image[x+3, y+1] < image[x, y] - thres: below_thres = below_thres + 1
        if image[x+3, y-1] < image[x, y] - thres: below_thres = below_thres + 1
    else: return 0
    if above_thres >= 9 or below_thres >= 9: return 1

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

def sortScore(val):
    return val[2]

def fast_algorithm (image,thres = 80):
    fast_keypoints = fast_detect(image,thres)
    harris_corners = find_harris_corners(image,100000.0)
    harris_corners.sort(key=sortScore, reverse=True)
    top_50_kps = []
    top50_scores = []

    for fs_kp in fast_keypoints:
        for hr_kp in harris_corners:
                if hr_kp[0] == fs_kp[0] and hr_kp[1] == fs_kp[1]:
                    top_50_kps.append(fs_kp)
                    top50_scores.append(hr_kp[2])
                    break
    top_50_kps = top_50_kps[:50]
    top50_scores = top50_scores[:50]
    print("Top 50 points: ",top_50_kps)
    return top_50_kps, top50_scores

    keypoints_scores = fast_detect(image,thres)
    keypoints = []
    best_scores = []
    for keypoint_score in keypoints_scores:
                keypoints.append(keypoint_score[0])
                best_scores.append(keypoint_score[1])
    return keypoints, best_scores

