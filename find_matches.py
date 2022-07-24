import numpy as np
from scipy.spatial.distance import cdist
import cv2
import matplotlib.pyplot as plt

def match(img1, img2, kp1, kp2, des1, des2):
    orb = cv2.ORB_create(42)
    bf = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    kp_1, des1_1 = orb.detectAndCompute(img1,None)
    kp_2, des2_1 = orb.detectAndCompute(img2,None)

    # print(des1.shape)
    # print(des2.shape)

    # if des1.shape[0] < des2.shape[0]:
    #     for i in range(des1.shape[0] + 1, des2.shape[0] + 1):
    #         des2 = des2[:des1.shape[0], ]
    # elif des2.shape[0] < des1.shape[0]:
    #     for i in range(des2.shape[0] + 1, des1.shape[0] + 1):
    #         des1 = des1[:des1.shape[0], ]

    # print(des1.shape)
    # print(des2.shape)

    matches = bf.match(des1,des2_1)
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp_2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("KILL ME",img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()