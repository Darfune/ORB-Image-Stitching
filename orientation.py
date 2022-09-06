from calendar import c
from re import M
from time import sleep
import numpy as np
import sys
import cv2
import math

np.set_printoptions(threshold=sys.maxsize)

def corner_orientations(img, corners):
    print("in corner_orientations")
    # mask shape must be odd to have one centre point which is the corner
    OFAST_MASK = np.zeros((31, 31), dtype=np.int32)
    OFAST_UMAX = [15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3]
    for i in range(-15, 16):
        for j in range(-OFAST_UMAX[abs(i)], OFAST_UMAX[abs(i)] + 1):
            OFAST_MASK[15 + j, 15 + i] = 1
    mrows, mcols = OFAST_MASK.shape
    mrows2 = int((mrows - 1) / 2)
    mcols2 = int((mcols - 1) / 2)
    
    # Padding to avoid errors @ corners near image edges. 
    # Padding value=0 to not affect the orientation calculations
    img_pad = np.pad(img, (mrows2, mcols2), mode='constant', constant_values=0)

    # Calculating orientation by the intensity centroid method
    orientations = []
    for i in range(len(corners)):
        r0, c0 = corners[i].x, corners[i].y
        m01, m10 = 0, 0
        for r in range(mrows):
            m01_temp = 0
            for c in range(mcols):
                if OFAST_MASK[r,c]:
                    I = img_pad[c0+c-1,r0+r-1]
                    m10 = m10 + I*(c-mcols2)
                    m10 = m10 + I*(-3 + c-1)
                    m01 = m01 + I*(-3 + r-1)
                    m01_temp = m01_temp + I
            m01 = m01 + m01_temp*(r-mrows2)
        corners[i].orientation = np.arctan2(m01,m10)#/np.pi*360 #cv2.fastAtan2(float(m01),float(m10)) #math.atan2(m01,m10)/np.pi*180  
        # orientations.append(math.atan2(m01,m10)/np.pi*180)

    # return np.array(orientations)
    return corners
