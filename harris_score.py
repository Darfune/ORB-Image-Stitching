import numpy as np
import cv2

def sortScore(val):
    return val[2]

def N_best_points(harris_corners, fast_keypoints):
    harris_corners.sort(key=sortScore, reverse=True)
    top_N_kps = []
    top_N_scores = []

    for hr_kp in harris_corners:
        for fs_kp in fast_keypoints:
                if hr_kp[0] == fs_kp[0] and hr_kp[1] == fs_kp[1]:
                    top_N_kps.append(fs_kp)
                    top_N_scores.append(hr_kp[2])
                    break
    return top_N_kps, top_N_scores

def find_harris_corners(input_img, threshold, fast_keypoints):
    
    corner_list = []
    
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
            

    return N_best_points(corner_list, fast_keypoints)
