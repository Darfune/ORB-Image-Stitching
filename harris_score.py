import numpy as np
from objects import keypoint

def sort(corner_list):
    size = len(corner_list)

    for index in range(size):
        min_index = index

        for j in range(index + 1, size):
            if corner_list[j].score > corner_list[min_index].score:
                min_index = j

        corner_list[index], corner_list[min_index] = corner_list[min_index], corner_list[index]

    return corner_list[:1000]

def find_harris_corners(input_img, threshold, fast_keypoints):
    print("in find_harris_corners")
    corner_list = []
    
    offset = int(5/2) 
    
    dy, dx = np.gradient(input_img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    for kp in fast_keypoints:  

        x = kp.x
        y = kp.y
        octave = kp.octave
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
            kp = keypoint(x,y, score = r, octave = octave)
            corner_list.append(kp)

    return sort(corner_list)
