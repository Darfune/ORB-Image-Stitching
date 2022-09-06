import numpy as np
from scipy.spatial.distance import cdist
import cv2
import matplotlib.pyplot as plt




def match(img1, img2, kp1, kp2, des1, des2):

    bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)


    matches = []

    # rawMatches = bf.knnMatch(des1,des2, k=2)


    # # rawMatches = sorted(rawMatches, key = lambda x:x.distance)
    # for m,n in rawMatches:
    #     # ensure the distance is within a certain ratio of each
    #     # other (i.e. Lowe's ratio test)
    #     if m.distance < n.distance * 0.75:
    #         matches.append(m)
    # # Draw first 10 matches.


    matches = bf.match(des1, des2)
    print(matches)
    # sorted by distance
    matches = sorted(matches, key=lambda x: x.distance)
    print(matches)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:15],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)



    cv2.imshow('img3',img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return matches[:15]