import numpy as np
from harris_score import find_harris_corners

def fast_detect(image, thres):
    height, width = image.shape

    keypoints = []
    
    for x in range(3,height-3):
        for y in range(3,width-3):
            
            above_thres = 0
            below_thres = 0
            # main 4 corners
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

            if above_thres >= 9 or below_thres >= 9:
                keypoints.append((x, y),)

    return keypoints 

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


def fast_algorithm (image,thres = 80):
    fast_keypoints = fast_detect(image,thres)
    return find_harris_corners(image,100000.0, fast_keypoints, 50)

