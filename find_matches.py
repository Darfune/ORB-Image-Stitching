import numpy as np
from scipy.spatial.distance import cdist
import cv2
import matplotlib.pyplot as plt




def match(img1, img2, kp1, kp2, des1, des2):
    orb = cv2.ORB_create(42)
    bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
    # kp1, des1 = orb.compute(img1,kp1)
    # kp2, des2 = orb.compute(img2,kp2)

    # # print(des1.shape)
    # # print(des2.shape)

    # if des1.shape[0] < des2.shape[0]:
    #     for i in range(des1.shape[0] + 1, des2.shape[0] + 1):
    #         des2 = des2[:des1.shape[0], ]
    # elif des2.shape[0] < des1.shape[0]:
    #     for i in range(des2.shape[0] + 1, des1.shape[0] + 1):
    #         des1 = des1[:des2.shape[0], ]


    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("KILL ME",img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()