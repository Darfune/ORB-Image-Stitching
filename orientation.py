from calendar import c
from re import M
from time import sleep
import numpy as np
import sys
import cv2
import math

np.set_printoptions(threshold=sys.maxsize)
# #Calculating orientation by the intensity centroid method
# def corner_orientations(image, corners):

#     mask = np.ones((7,7), dtype=np.uint8)
#     mask[0,0] = 0
#     mask[0,6] = 0
#     mask[6,0] = 0
#     mask[6,6] = 0
#     mask[0,1] = 0
#     mask[1,0] = 0
#     mask[6,1] = 0
#     mask[5,0] = 0
#     mask[6,5] = 0
#     mask[5,6] = 0
#     mask[1,6] = 0
#     angle = []
#     image_with_padding = np.zeros((image.shape[0]+3, image.shape[1]+3), dtype=np.uint8)

#     for i in range(len(corners)):
#         r0 = corners[i][0]
#         c0 = corners[i][1]
#         m01 = 0
#         m10 = 0
#         for r in range(0,7):
#             for c in range(0,7):
#                 if mask[r,c] == 1:
#                     pixel = image_with_padding[r0 + r-1][c0 + c-1]#pixel = Ip(c0 + k-1, r0 + j-1)

#                     m10 = m10 + pixel * (-3 + c - 1)
#                     m01 = m01 + pixel * (-3 + r - 1)
#                     # m01 = m01 + image[r0+r-3,c0+c-3]
#                     # m10 = m10 + image[r0+r-3,c0+c-3]

#         angle.append(math.atan2(m01, m10))
#     print(angle)
    # return angle
#  for j= 1:r
#         for k = 1:c
#             if mask(k,j) 
#                pixel = Ip(c0 + k-1, r0 + j-1);
#                m10 = m10 + pixel * (-radii + k - 1);
#                m01 = m01 + pixel * (-radii + j - 1);
#             end
#         end
#     end
#     angle(i) = atan2(m01,m10);
# if __name__ == '__main__':
#     img = cv2.imread("images/main_set/00.jpg")
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     corner_orientations(img,2)

def corner_orientations(img, corners):
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
        c0, r0 = corners[i]
        m01, m10 = 0, 0
        for r in range(mrows):
            m01_temp = 0
            for c in range(mcols):
                if OFAST_MASK[r,c]:
                    I = img_pad[c0+c-1, r0+r-1]
                    m10 = m10 + I*(c-mcols2)
                    m01_temp = m01_temp + I
            m01 = m01 + m01_temp*(r-mrows2)
        orientations.append(math.atan2(m01,m10)/np.pi*180)
    print(orientations)
    return np.array(orientations)
